use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::{
    constants::{WAL_CHECKPOINT_PERIOD, WAL_CHECKPOINT_THRESHOLD},
    error::{MemvidError, Result},
    types::Header,
};

// Each WAL record header: [seq: u64][len: u32][reserved: 4 bytes][checksum: 32 bytes]
const ENTRY_HEADER_SIZE: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WalStats {
    pub region_size: u64,
    pub pending_bytes: u64,
    pub appends_since_checkpoint: u64,
    pub sequence: u64,
    pub skip_sync: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    pub sequence: u64,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
pub struct EmbeddedWal {
    file: File,
    region_offset: u64,
    region_size: u64,
    write_head: u64,
    checkpoint_head: u64,
    pending_bytes: u64,
    sequence: u64,
    checkpoint_sequence: u64,
    appends_since_checkpoint: u64,
    read_only: bool,
    skip_sync: bool,
}

impl EmbeddedWal {
    /// Mark a fully-checkpointed region as empty by writing a zero
    /// sentinel at its start. Required after the region is resized in
    /// place: stale record bytes from the previous geometry remain in
    /// the retained head, and a record straddling the new boundary
    /// makes the next open's scan misparse them as corruption.
    pub fn write_empty_sentinel(file: &mut File, header: &Header) -> Result<()> {
        file.seek(SeekFrom::Start(header.wal_offset))?;
        file.write_all(&[0u8; ENTRY_HEADER_SIZE])?;
        Ok(())
    }

    pub fn open(file: &File, header: &Header) -> Result<Self> {
        Self::open_internal(file, header, false)
    }

    pub fn open_read_only(file: &File, header: &Header) -> Result<Self> {
        Self::open_internal(file, header, true)
    }

    fn open_internal(file: &File, header: &Header, read_only: bool) -> Result<Self> {
        if header.wal_size == 0 {
            return Err(MemvidError::InvalidHeader {
                reason: "wal_size must be non-zero".into(),
            });
        }
        let mut clone = file.try_clone()?;
        let region_offset = header.wal_offset;
        let region_size = header.wal_size;
        let checkpoint_sequence = header.wal_sequence;

        let (entries, next_head) = Self::scan_records(&mut clone, region_offset, region_size)?;

        let pending_bytes = entries
            .iter()
            .filter(|entry| entry.sequence > checkpoint_sequence)
            .map(|entry| entry.total_size)
            .sum();
        let sequence = entries
            .last()
            .map_or(checkpoint_sequence, |entry| entry.sequence);

        let mut wal = Self {
            file: clone,
            region_offset,
            region_size,
            write_head: next_head % region_size,
            checkpoint_head: header.wal_checkpoint_pos % region_size,
            pending_bytes,
            sequence,
            checkpoint_sequence,
            appends_since_checkpoint: 0,
            read_only,
            skip_sync: false,
        };

        if !wal.read_only {
            wal.initialise_sentinel()?;
        }
        Ok(wal)
    }

    fn assert_writable(&self) -> Result<()> {
        if self.read_only {
            return Err(MemvidError::Lock(
                "wal is read-only; reopen memory with write access".into(),
            ));
        }
        Ok(())
    }

    pub fn append_entry(&mut self, payload: &[u8]) -> Result<u64> {
        self.assert_writable()?;
        let payload_len = payload.len();
        if payload_len > u32::MAX as usize {
            return Err(MemvidError::CheckpointFailed {
                reason: "WAL payload too large".into(),
            });
        }

        let entry_size = ENTRY_HEADER_SIZE as u64 + payload_len as u64;
        if entry_size > self.region_size {
            return Err(MemvidError::CheckpointFailed {
                reason: "embedded WAL region too small for entry".into(),
            });
        }
        if self.pending_bytes + entry_size > self.region_size {
            return Err(MemvidError::CheckpointFailed {
                reason: "embedded WAL region full".into(),
            });
        }

        // Check if we need to wrap around
        let wrapping = self.write_head + entry_size > self.region_size;
        if wrapping {
            // If wrapping would overwrite uncommitted data, return "WAL full" error
            // instead of silently overwriting. This triggers WAL growth.
            // The checkpoint_head marks where committed data starts - if we wrap and would
            // write over any pending (uncommitted) data, we must grow the WAL instead.
            if self.pending_bytes > 0 {
                return Err(MemvidError::CheckpointFailed {
                    reason: "embedded WAL region full".into(),
                });
            }
            self.write_head = 0;
        }

        let next_sequence = self.sequence + 1;
        tracing::debug!(
            wal.write_head = self.write_head,
            wal.sequence = next_sequence,
            wal.payload_len = payload_len,
            "wal append entry"
        );
        self.write_record(self.write_head, next_sequence, payload)?;

        self.write_head = (self.write_head + entry_size) % self.region_size;
        self.pending_bytes += entry_size;
        self.sequence = self.sequence.wrapping_add(1);
        self.appends_since_checkpoint = self.appends_since_checkpoint.saturating_add(1);

        self.maybe_write_sentinel()?;

        Ok(self.sequence)
    }

    #[must_use]
    pub fn should_checkpoint(&self) -> bool {
        if self.read_only || self.region_size == 0 {
            return false;
        }
        let occupancy = self.pending_bytes as f64 / self.region_size as f64;
        occupancy >= WAL_CHECKPOINT_THRESHOLD
            || self.appends_since_checkpoint >= WAL_CHECKPOINT_PERIOD
    }

    pub fn record_checkpoint(&mut self, header: &mut Header) -> Result<()> {
        self.assert_writable()?;
        self.checkpoint_head = self.write_head;
        self.pending_bytes = 0;
        self.appends_since_checkpoint = 0;
        self.checkpoint_sequence = self.sequence;
        header.wal_checkpoint_pos = self.checkpoint_head;
        header.wal_sequence = self.checkpoint_sequence;
        self.maybe_write_sentinel()
    }

    pub fn pending_records(&mut self) -> Result<Vec<WalRecord>> {
        self.records_after(self.checkpoint_sequence)
    }

    pub fn records_after(&mut self, sequence: u64) -> Result<Vec<WalRecord>> {
        let (entries, next_head) =
            Self::scan_records(&mut self.file, self.region_offset, self.region_size)?;

        self.sequence = entries.last().map_or(self.sequence, |entry| entry.sequence);
        self.pending_bytes = entries
            .iter()
            .filter(|entry| entry.sequence > self.checkpoint_sequence)
            .map(|entry| entry.total_size)
            .sum();
        self.write_head = next_head % self.region_size;
        if !self.read_only {
            self.initialise_sentinel()?;
        }

        Ok(entries
            .into_iter()
            .filter(|entry| entry.sequence > sequence)
            .map(|entry| WalRecord {
                sequence: entry.sequence,
                payload: entry.payload,
            })
            .collect())
    }

    #[must_use]
    pub fn stats(&self) -> WalStats {
        WalStats {
            region_size: self.region_size,
            pending_bytes: self.pending_bytes,
            appends_since_checkpoint: self.appends_since_checkpoint,
            sequence: self.sequence,
            skip_sync: self.skip_sync,
        }
    }

    #[must_use]
    pub fn region_offset(&self) -> u64 {
        self.region_offset
    }

    #[must_use]
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Enable or disable per-entry fsync.
    ///
    /// When `skip` is `true`, `write_record()` will not call `sync_all()` after
    /// each WAL append. The caller **must** call [`flush()`](Self::flush) after
    /// the batch to ensure durability.
    pub fn set_skip_sync(&mut self, skip: bool) {
        self.skip_sync = skip;
    }

    /// Force an `fsync` on the underlying WAL file.
    ///
    /// Call this after a batch of appends performed with `skip_sync = true`
    /// to ensure all data is durable on disk.
    pub fn flush(&mut self) -> Result<()> {
        self.file.sync_all().map_err(Into::into)
    }

    fn initialise_sentinel(&mut self) -> Result<()> {
        self.maybe_write_sentinel()
    }

    fn write_record(&mut self, position: u64, sequence: u64, payload: &[u8]) -> Result<()> {
        self.assert_writable()?;
        let digest = blake3::hash(payload);
        let mut header = [0u8; ENTRY_HEADER_SIZE];
        header[..8].copy_from_slice(&sequence.to_le_bytes());
        header[8..12]
            .copy_from_slice(&(u32::try_from(payload.len()).unwrap_or(u32::MAX)).to_le_bytes());
        header[16..48].copy_from_slice(digest.as_bytes());

        // Atomic write: combine header and payload into single buffer
        // This prevents corruption if the file is closed mid-write
        let mut combined = Vec::with_capacity(ENTRY_HEADER_SIZE + payload.len());
        combined.extend_from_slice(&header);
        combined.extend_from_slice(payload);

        self.seek_and_write(position, &combined)?;
        if tracing::enabled!(tracing::Level::DEBUG) {
            if let Err(err) = self.debug_verify_header(position, sequence, payload.len()) {
                tracing::warn!(error = %err, "wal header verify failed");
            }
        }

        // Force fsync to ensure data is durable before returning
        // Critical for preventing corruption during rapid file operations
        // In batch mode (skip_sync=true), fsync is deferred to flush() for performance
        if !self.skip_sync {
            self.file.sync_all()?;
        }

        Ok(())
    }

    fn write_zero_header(&mut self, position: u64) -> Result<u64> {
        self.assert_writable()?;
        if self.region_size == 0 {
            return Ok(0);
        }
        let mut pos = position % self.region_size;
        let remaining = self.region_size - pos;
        if remaining < ENTRY_HEADER_SIZE as u64 {
            if remaining > 0 {
                // Safe: remaining < ENTRY_HEADER_SIZE (48) so always fits in usize
                #[allow(clippy::cast_possible_truncation)]
                let zero_tail = vec![0u8; remaining as usize];
                self.seek_and_write(pos, &zero_tail)?;
            }
            pos = 0;
        }
        let zero = [0u8; ENTRY_HEADER_SIZE];
        self.seek_and_write(pos, &zero)?;
        Ok(pos)
    }

    fn seek_and_write(&mut self, position: u64, bytes: &[u8]) -> Result<()> {
        self.assert_writable()?;
        let pos = position % self.region_size;
        let absolute = self.region_offset + pos;
        self.file.seek(SeekFrom::Start(absolute))?;
        self.file.write_all(bytes)?;
        Ok(())
    }

    fn maybe_write_sentinel(&mut self) -> Result<()> {
        if self.read_only || self.region_size == 0 {
            return Ok(());
        }
        if self.pending_bytes >= self.region_size {
            return Ok(());
        }
        // Sentinel marks end of valid entries - always keep write_head in sync
        let next = self.write_zero_header(self.write_head)?;
        self.write_head = next;
        Ok(())
    }

    fn scan_records(file: &mut File, offset: u64, size: u64) -> Result<(Vec<ScannedRecord>, u64)> {
        let mut records = Vec::new();
        let mut cursor = 0u64;
        while cursor + ENTRY_HEADER_SIZE as u64 <= size {
            file.seek(SeekFrom::Start(offset + cursor))?;
            let mut header = [0u8; ENTRY_HEADER_SIZE];
            file.read_exact(&mut header)?;

            let sequence = u64::from_le_bytes(header[..8].try_into().map_err(|_| {
                MemvidError::WalCorruption {
                    offset: cursor,
                    reason: "invalid wal sequence header".into(),
                }
            })?);
            let length = u64::from(u32::from_le_bytes(header[8..12].try_into().map_err(
                |_| MemvidError::WalCorruption {
                    offset: cursor,
                    reason: "invalid wal length header".into(),
                },
            )?));
            let checksum = &header[16..48];

            if sequence == 0 && length == 0 {
                break;
            }
            if length == 0 || cursor + ENTRY_HEADER_SIZE as u64 + length > size {
                tracing::error!(
                    wal.scan_offset = cursor,
                    wal.sequence = sequence,
                    wal.length = length,
                    wal.region_size = size,
                    "wal record length invalid"
                );
                return Err(MemvidError::WalCorruption {
                    offset: cursor,
                    reason: "wal record length invalid".into(),
                });
            }

            // Safe: length comes from u32::from_le_bytes above, so max is u32::MAX
            // which fits in usize on all supported platforms (32-bit and 64-bit)
            let length_usize = usize::try_from(length).map_err(|_| MemvidError::WalCorruption {
                offset: cursor,
                reason: "wal record length too large for platform".into(),
            })?;
            let mut payload = vec![0u8; length_usize];
            file.read_exact(&mut payload)?;
            let expected = blake3::hash(&payload);
            if expected.as_bytes() != checksum {
                return Err(MemvidError::WalCorruption {
                    offset: cursor,
                    reason: "wal record checksum mismatch".into(),
                });
            }

            records.push(ScannedRecord {
                sequence,
                payload,
                total_size: ENTRY_HEADER_SIZE as u64 + length,
            });

            cursor += ENTRY_HEADER_SIZE as u64 + length;
        }

        Ok((records, cursor))
    }
}

#[derive(Debug)]
struct ScannedRecord {
    sequence: u64,
    payload: Vec<u8>,
    total_size: u64,
}

impl EmbeddedWal {
    fn debug_verify_header(
        &mut self,
        position: u64,
        expected_sequence: u64,
        expected_len: usize,
    ) -> Result<()> {
        if self.region_size == 0 {
            return Ok(());
        }
        let pos = position % self.region_size;
        let absolute = self.region_offset + pos;
        let mut buf = [0u8; ENTRY_HEADER_SIZE];
        self.file.seek(SeekFrom::Start(absolute))?;
        self.file.read_exact(&mut buf)?;

        // Safe byte extraction - return early if malformed (debug function)
        let seq = buf
            .get(..8)
            .and_then(|s| <[u8; 8]>::try_from(s).ok())
            .map_or(0, u64::from_le_bytes);
        let len = buf
            .get(8..12)
            .and_then(|s| <[u8; 4]>::try_from(s).ok())
            .map_or(0, u32::from_le_bytes);

        tracing::debug!(
            wal.verify_position = pos,
            wal.verify_sequence = seq,
            wal.expected_sequence = expected_sequence,
            wal.verify_length = len,
            wal.expected_length = expected_len,
            "wal header verify"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::WAL_OFFSET;
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::tempfile;

    fn header_for(size: u64) -> Header {
        Header {
            magic: *b"MV2\0",
            version: 0x0201,
            footer_offset: 0,
            wal_offset: WAL_OFFSET,
            wal_size: size,
            wal_checkpoint_pos: 0,
            wal_sequence: 0,
            toc_checksum: [0u8; 32],
        }
    }

    fn prepare_wal(size: u64) -> (File, Header) {
        let file = tempfile().expect("temp file");
        file.set_len(WAL_OFFSET + size).expect("set_len");
        let header = header_for(size);
        (file, header)
    }

    #[test]
    fn append_and_recover() {
        let (file, header) = prepare_wal(1024);
        let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");

        wal.append_entry(b"first").expect("append first");
        wal.append_entry(b"second").expect("append second");

        let records = wal.records_after(0).expect("records");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].payload, b"first");
        assert_eq!(records[0].sequence, 1);
        assert_eq!(records[1].payload, b"second");
        assert_eq!(records[1].sequence, 2);
    }

    #[test]
    fn wrap_and_checkpoint() {
        let size = (ENTRY_HEADER_SIZE as u64 * 2) + 64;
        let (file, mut header) = prepare_wal(size);
        let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");

        wal.append_entry(&[0xAA; 32]).expect("append a");
        wal.append_entry(&[0xBB; 32]).expect("append b");
        wal.record_checkpoint(&mut header).expect("checkpoint");

        assert!(wal.pending_records().expect("pending").is_empty());

        wal.append_entry(&[0xCC; 32]).expect("append c");
        let records = wal.pending_records().expect("after append");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].payload, vec![0xCC; 32]);
    }

    #[test]
    fn corrupted_record_reports_offset() {
        let (mut file, header) = prepare_wal(64);
        // Write a record header that claims an impossible length so scan_records trips.
        file.seek(SeekFrom::Start(header.wal_offset)).expect("seek");
        let mut record = [0u8; ENTRY_HEADER_SIZE];
        record[..8].copy_from_slice(&1u64.to_le_bytes()); // sequence
        record[8..12].copy_from_slice(&(u32::MAX).to_le_bytes()); // absurd length
        file.write_all(&record).expect("write corrupt header");
        file.sync_all().expect("sync");

        let err = EmbeddedWal::open(&file, &header).expect_err("open should fail");
        match err {
            MemvidError::WalCorruption { offset, reason } => {
                assert_eq!(offset, 0);
                assert!(reason.contains("length"), "reason should mention length");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
