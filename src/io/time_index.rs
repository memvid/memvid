use std::io::{Read, Seek, SeekFrom, Write};

use blake3::Hasher;

use crate::{
    constants::TIME_INDEX_MAGIC,
    error::{MemvidError, Result},
};

/// Safety ceiling on declared time-index track length. A malformed or hostile
/// `.mv2` can declare an arbitrarily large `bytes_length` in its TOC; without
/// this cap, `read_track` would allocate proportional capacity before the
/// later `read_exact` would fail. 16 GiB matches the temporal-index ceiling.
const MAX_TIME_INDEX_BYTES: u64 = 1 << 34;

/// Raw entry used to build the time index track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeIndexEntry {
    pub timestamp: i64,
    pub frame_id: u64,
}

impl TimeIndexEntry {
    #[must_use]
    pub fn new(timestamp: i64, frame_id: u64) -> Self {
        Self {
            timestamp,
            frame_id,
        }
    }
}

/// Appends entries to the time index track, returning `(offset, length, checksum)`.
/// Entries are sorted by `(timestamp, frame_id)` prior to writing.
pub fn append_track<W: Write + Seek>(
    writer: &mut W,
    entries: &mut [TimeIndexEntry],
) -> Result<(u64, u64, [u8; 32])> {
    entries.sort_by_key(|entry| (entry.timestamp, entry.frame_id));

    let offset = writer.stream_position()?;
    let mut hasher = Hasher::new();

    writer.write_all(&TIME_INDEX_MAGIC)?;
    hasher.update(&TIME_INDEX_MAGIC);

    let count = entries.len() as u64;
    let count_bytes = count.to_le_bytes();
    writer.write_all(&count_bytes)?;
    hasher.update(&count_bytes);

    for entry in entries.iter() {
        let ts_bytes = entry.timestamp.to_le_bytes();
        let id_bytes = entry.frame_id.to_le_bytes();
        writer.write_all(&ts_bytes)?;
        writer.write_all(&id_bytes)?;
        hasher.update(&ts_bytes);
        hasher.update(&id_bytes);
    }

    let end = writer.stream_position()?;
    let length = end - offset;
    Ok((offset, length, *hasher.finalize().as_bytes()))
}

/// Reads the time index entries located at `(offset, length)` and validates ordering.
pub fn read_track<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    length: u64,
) -> Result<Vec<TimeIndexEntry>> {
    if length > MAX_TIME_INDEX_BYTES {
        return Err(MemvidError::InvalidTimeIndex {
            reason: "length exceeds supported limit".into(),
        });
    }

    reader.seek(SeekFrom::Start(offset))?;

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != TIME_INDEX_MAGIC {
        return Err(MemvidError::InvalidTimeIndex {
            reason: "magic mismatch".into(),
        });
    }

    let mut count_buf = [0u8; 8];
    reader.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf);

    let header_len = 4u64 + 8;
    if length < header_len {
        return Err(MemvidError::InvalidTimeIndex {
            reason: "length shorter than header".into(),
        });
    }
    let payload_bytes = length - header_len;
    let expected_payload = count
        .checked_mul((std::mem::size_of::<i64>() + std::mem::size_of::<u64>()) as u64)
        .ok_or(MemvidError::InvalidTimeIndex {
            reason: "entry count overflow".into(),
        })?;
    if payload_bytes != expected_payload {
        return Err(MemvidError::InvalidTimeIndex {
            reason: "length does not match declared count".into(),
        });
    }

    // Safe: count validated by checked_mul and payload_bytes comparison above
    #[allow(clippy::cast_possible_truncation)]
    let mut entries = Vec::with_capacity(count as usize);
    let mut prev: Option<TimeIndexEntry> = None;
    for _ in 0..count {
        let mut ts_buf = [0u8; 8];
        reader.read_exact(&mut ts_buf)?;
        let timestamp = i64::from_le_bytes(ts_buf);

        let mut id_buf = [0u8; 8];
        reader.read_exact(&mut id_buf)?;
        let frame_id = u64::from_le_bytes(id_buf);

        let entry = TimeIndexEntry {
            timestamp,
            frame_id,
        };
        if let Some(prev_entry) = prev {
            if entry.timestamp < prev_entry.timestamp
                || (entry.timestamp == prev_entry.timestamp && entry.frame_id < prev_entry.frame_id)
            {
                return Err(MemvidError::InvalidTimeIndex {
                    reason: "entries not sorted".into(),
                });
            }
        }
        prev = Some(entry);
        entries.push(entry);
    }

    Ok(entries)
}

/// Calculates the checksum for the provided entries in canonical order.
#[must_use]
pub fn calculate_checksum(entries: &[TimeIndexEntry]) -> [u8; 32] {
    let mut sorted = entries.to_vec();
    sorted.sort_by_key(|entry| (entry.timestamp, entry.frame_id));

    let mut hasher = Hasher::new();
    hasher.update(&TIME_INDEX_MAGIC);
    hasher.update(&(sorted.len() as u64).to_le_bytes());
    for entry in &sorted {
        hasher.update(&entry.timestamp.to_le_bytes());
        hasher.update(&entry.frame_id.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::tempfile;

    #[test]
    fn append_and_read_roundtrip() {
        let mut file = tempfile().expect("temp file");
        let mut entries = vec![
            TimeIndexEntry::new(30, 2),
            TimeIndexEntry::new(10, 0),
            TimeIndexEntry::new(20, 1),
        ];

        let (offset, length, checksum) =
            append_track(&mut file, &mut entries).expect("append track");
        assert_eq!(entries[0].timestamp, 10); // sorted in place
        let read_entries = read_track(&mut file, offset, length).expect("read track");
        assert_eq!(read_entries.len(), 3);
        assert!(
            read_entries
                .windows(2)
                .all(|w| w[0].timestamp <= w[1].timestamp)
        );

        let expected_checksum = calculate_checksum(&read_entries);
        assert_eq!(checksum, expected_checksum);
    }

    #[test]
    fn read_rejects_unsorted_entries() {
        let mut file = tempfile().expect("temp file");
        // Craft an invalid track where entries descend.
        file.write_all(&TIME_INDEX_MAGIC).unwrap();
        file.write_all(&(2u64).to_le_bytes()).unwrap();
        file.write_all(&50i64.to_le_bytes()).unwrap();
        file.write_all(&5u64.to_le_bytes()).unwrap();
        file.write_all(&40i64.to_le_bytes()).unwrap();
        file.write_all(&4u64.to_le_bytes()).unwrap();

        let length = file.seek(SeekFrom::End(0)).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        let err = read_track(&mut file, 0, length).expect_err("unsorted entries must fail");
        matches!(err, MemvidError::InvalidTimeIndex { .. });
    }

    #[test]
    fn read_rejects_oversized_declared_length() {
        // Hostile or malformed TOC declaring a length above the safety ceiling
        // must be rejected before any capacity allocation happens.
        let mut file = tempfile().expect("temp file");
        let err = read_track(&mut file, 0, MAX_TIME_INDEX_BYTES + 1)
            .expect_err("oversized length must be rejected");
        match err {
            MemvidError::InvalidTimeIndex { reason } => {
                assert!(reason.contains("length"), "unexpected reason: {reason}");
            }
            other => panic!("expected InvalidTimeIndex, got {other:?}"),
        }
    }

    #[test]
    fn calculate_checksum_is_deterministic() {
        let entries = vec![
            TimeIndexEntry::new(5, 10),
            TimeIndexEntry::new(1, 2),
            TimeIndexEntry::new(5, 9),
        ];
        let checksum_a = calculate_checksum(&entries);
        let checksum_b = calculate_checksum(&entries);
        assert_eq!(checksum_a, checksum_b);
    }
}
