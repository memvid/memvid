//! Binary encode/decode helpers for the temporal mentions track (feature gated).

use std::cmp::Ordering;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;

use blake3::Hasher;

use crate::constants::{TEMPORAL_TRACK_MAGIC, TEMPORAL_TRACK_VERSION};
use crate::error::{MemvidError, Result};
use crate::types::{
    AnchorSource, TemporalAnchor, TemporalMention, TemporalMentionFlags, TemporalMentionKind,
    TemporalTrack,
};

const HEADER_SIZE: usize = 56;
const CHECKSUM_OFFSET: usize = 24;
const MENTION_RECORD_SIZE: usize = 32; // padded to 32 bytes for alignment
const ANCHOR_RECORD_SIZE: usize = 24;
const MAX_TEMPORAL_TRACK_BYTES: u64 = 1 << 34; // 16 GiB safety ceiling

/// Initial capacity for decoded mention/anchor vectors. Bounds the *eager*
/// allocation driven by the on-disk header counts — both vectors still grow
/// to fit all records the reader delivers, but a hostile count no longer
/// reserves count-proportional memory before any record is read.
const INITIAL_RECORDS_CAPACITY: usize = 1024;

#[derive(Debug, Clone, Copy)]
struct RawMention {
    ts_utc: i64,
    frame_id: u64,
    byte_start: u32,
    byte_len: u32,
    kind: u8,
    confidence: u16,
    tz_hint_minutes: i16,
    flags: u8,
    reserved: u16,
}

impl RawMention {
    fn from_high_level(mention: &TemporalMention) -> Self {
        Self {
            ts_utc: mention.ts_utc,
            frame_id: mention.frame_id,
            byte_start: mention.byte_start,
            byte_len: mention.byte_len,
            kind: mention.kind.to_u8(),
            confidence: mention.confidence,
            tz_hint_minutes: mention.tz_hint_minutes,
            flags: mention.flags.0,
            reserved: 0,
        }
    }

    fn encode(self) -> [u8; MENTION_RECORD_SIZE] {
        let mut out = [0u8; MENTION_RECORD_SIZE];
        out[0..8].copy_from_slice(&self.ts_utc.to_le_bytes());
        out[8..16].copy_from_slice(&self.frame_id.to_le_bytes());
        out[16..20].copy_from_slice(&self.byte_start.to_le_bytes());
        out[20..24].copy_from_slice(&self.byte_len.to_le_bytes());
        out[24] = self.kind;
        out[25..27].copy_from_slice(&self.confidence.to_le_bytes());
        out[27..29].copy_from_slice(&self.tz_hint_minutes.to_le_bytes());
        out[29] = self.flags;
        out[30..32].copy_from_slice(&self.reserved.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let ts_utc = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let frame_id = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let byte_start = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let byte_len = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let kind = bytes[24];
        let confidence = u16::from_le_bytes(bytes[25..27].try_into().unwrap());
        let tz_hint_minutes = i16::from_le_bytes(bytes[27..29].try_into().unwrap());
        let flags = bytes[29];
        let reserved = u16::from_le_bytes(bytes[30..32].try_into().unwrap());
        Ok(Self {
            ts_utc,
            frame_id,
            byte_start,
            byte_len,
            kind,
            confidence,
            tz_hint_minutes,
            flags,
            reserved,
        })
    }

    fn into_high_level(self) -> Result<TemporalMention> {
        let kind =
            TemporalMentionKind::from_u8(self.kind).ok_or(MemvidError::InvalidTemporalTrack {
                reason: "unknown mention kind".into(),
            })?;
        Ok(TemporalMention::new(
            self.ts_utc,
            self.frame_id,
            self.byte_start,
            self.byte_len,
            kind,
            self.confidence,
            self.tz_hint_minutes,
            TemporalMentionFlags(self.flags),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
struct RawAnchor {
    frame_id: u64,
    anchor_ts: i64,
    source: u8,
    reserved: [u8; 7],
}

impl RawAnchor {
    fn from_high_level(anchor: &TemporalAnchor) -> Self {
        Self {
            frame_id: anchor.frame_id,
            anchor_ts: anchor.anchor_ts,
            source: anchor.source as u8,
            reserved: [0u8; 7],
        }
    }

    fn encode(self) -> [u8; ANCHOR_RECORD_SIZE] {
        let mut out = [0u8; ANCHOR_RECORD_SIZE];
        out[0..8].copy_from_slice(&self.frame_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.anchor_ts.to_le_bytes());
        out[16] = self.source;
        out[17..24].copy_from_slice(&self.reserved);
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let frame_id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let anchor_ts = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let source = bytes[16];
        let mut reserved = [0u8; 7];
        reserved.copy_from_slice(&bytes[17..24]);
        Ok(Self {
            frame_id,
            anchor_ts,
            source,
            reserved,
        })
    }

    fn into_high_level(self) -> Result<TemporalAnchor> {
        let source = match self.source {
            0 => AnchorSource::Explicit,
            1 => AnchorSource::FrameTimestamp,
            2 => AnchorSource::Metadata,
            3 => AnchorSource::IngestionClock,
            _ => {
                return Err(MemvidError::InvalidTemporalTrack {
                    reason: "unknown anchor source".into(),
                });
            }
        };
        Ok(TemporalAnchor::new(self.frame_id, self.anchor_ts, source))
    }
}

fn write_header<W: Write + Seek>(
    writer: &mut W,
    entry_count: u64,
    anchor_count: u64,
    flags: u16,
) -> Result<(u64, [u8; HEADER_SIZE])> {
    let offset = writer.seek(SeekFrom::Current(0))?;
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&TEMPORAL_TRACK_MAGIC);
    header[4..6].copy_from_slice(&TEMPORAL_TRACK_VERSION.to_le_bytes());
    header[6..8].copy_from_slice(&flags.to_le_bytes());
    header[8..16].copy_from_slice(&entry_count.to_le_bytes());
    header[16..24].copy_from_slice(&anchor_count.to_le_bytes());
    // checksum stays zero until finalisation
    writer.write_all(&header)?;
    Ok((offset, header))
}

fn patch_checksum<W: Write + Seek>(
    writer: &mut W,
    header_offset: u64,
    checksum: &[u8; 32],
    restore_pos: u64,
) -> Result<()> {
    writer.seek(SeekFrom::Start(header_offset + CHECKSUM_OFFSET as u64))?;
    writer.write_all(checksum)?;
    writer.seek(SeekFrom::Start(restore_pos))?;
    Ok(())
}

/// Serialises the provided mentions + anchors into the temporal track region.
pub fn append_track<W: Write + Seek>(
    writer: &mut W,
    mentions: &mut [TemporalMention],
    anchors: &mut [TemporalAnchor],
    flags: u32,
) -> Result<(u64, u64, [u8; 32])> {
    #[cfg(test)]
    println!(
        "append_track: mentions={}, anchors={}, flags={}",
        mentions.len(),
        anchors.len(),
        flags
    );
    mentions.sort_by(mention_cmp);
    anchors.sort_by_key(|anchor| anchor.frame_id);
    validate_mentions_sorted(mentions)?;
    validate_anchors_sorted(anchors)?;

    let entry_count = mentions.len() as u64;
    let anchor_count = anchors.len() as u64;
    let header_flags = flags as u16;

    let (header_offset, mut header) =
        write_header(writer, entry_count, anchor_count, header_flags)?;
    header[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 32].fill(0);

    let mut hasher = Hasher::new();
    hasher.update(&header);

    for mention in mentions.iter().map(RawMention::from_high_level) {
        let encoded = mention.encode();
        writer.write_all(&encoded)?;
        hasher.update(&encoded);
    }

    for anchor in anchors.iter().map(RawAnchor::from_high_level) {
        let encoded = anchor.encode();
        writer.write_all(&encoded)?;
        hasher.update(&encoded);
    }

    let checksum = *hasher.finalize().as_bytes();
    let end = writer.seek(SeekFrom::Current(0))?;
    patch_checksum(writer, header_offset, &checksum, end)?;

    Ok((header_offset, end - header_offset, checksum))
}

/// Reads and validates a temporal track from disk.
pub fn read_track<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    length: u64,
) -> Result<TemporalTrack> {
    if length > MAX_TEMPORAL_TRACK_BYTES {
        return Err(MemvidError::InvalidTemporalTrack {
            reason: "length exceeds supported limit".into(),
        });
    }

    if length < HEADER_SIZE as u64 {
        return Err(MemvidError::InvalidTemporalTrack {
            reason: "length shorter than header".into(),
        });
    }

    reader.seek(SeekFrom::Start(offset))?;
    let mut header = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header)?;

    if header[0..4] != TEMPORAL_TRACK_MAGIC {
        return Err(MemvidError::InvalidTemporalTrack {
            reason: "magic mismatch".into(),
        });
    }

    let version = u16::from_le_bytes(header[4..6].try_into().unwrap());
    if version > TEMPORAL_TRACK_VERSION {
        return Err(MemvidError::InvalidTemporalTrack {
            reason: "unsupported track version".into(),
        });
    }

    let flags = u16::from_le_bytes(header[6..8].try_into().unwrap()) as u32;
    let entry_count = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let anchor_count = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let checksum: [u8; 32] = header[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 32]
        .try_into()
        .unwrap();

    let expected_entries_bytes = entry_count.checked_mul(MENTION_RECORD_SIZE as u64).ok_or(
        MemvidError::InvalidTemporalTrack {
            reason: "entry count overflow".into(),
        },
    )?;
    let expected_anchor_bytes = anchor_count.checked_mul(ANCHOR_RECORD_SIZE as u64).ok_or(
        MemvidError::InvalidTemporalTrack {
            reason: "anchor count overflow".into(),
        },
    )?;

    let total_expected = HEADER_SIZE as u64 + expected_entries_bytes + expected_anchor_bytes;
    if total_expected != length {
        return Err(MemvidError::InvalidTemporalTrack {
            reason: "length does not match declared counts".into(),
        });
    }

    // Stream records one at a time rather than slurping the whole body into a
    // single `Vec<u8>`. A hostile but self-consistent header with
    // `length ≈ MAX_TEMPORAL_TRACK_BYTES` (and matching counts) would
    // otherwise force a ~16 GiB body allocation before any payload byte is
    // validated; here, both the hasher and the decoded vectors grow only as
    // `read_exact` succeeds.
    let mut header_for_hash = header;
    header_for_hash[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 32].fill(0);
    let mut hasher = Hasher::new();
    hasher.update(&header_for_hash);

    // Safe: counts validated by checked_mul and total_expected == length check
    // above. The eager Vec capacity is clamped to bound up-front allocation;
    // the vector grows as records are decoded. The result of the `min` always
    // fits in usize because INITIAL_RECORDS_CAPACITY is a usize literal.
    #[allow(clippy::cast_possible_truncation)]
    let mentions_initial = entry_count.min(INITIAL_RECORDS_CAPACITY as u64) as usize;
    #[allow(clippy::cast_possible_truncation)]
    let anchors_initial = anchor_count.min(INITIAL_RECORDS_CAPACITY as u64) as usize;
    let mut mentions = Vec::with_capacity(mentions_initial);
    let mut anchors = Vec::with_capacity(anchors_initial);

    let mut mention_buf = [0u8; MENTION_RECORD_SIZE];
    for _ in 0..entry_count {
        reader.read_exact(&mut mention_buf)?;
        hasher.update(&mention_buf);
        let raw = RawMention::decode(&mention_buf)?;
        mentions.push(raw.into_high_level()?);
    }

    let mut anchor_buf = [0u8; ANCHOR_RECORD_SIZE];
    for _ in 0..anchor_count {
        reader.read_exact(&mut anchor_buf)?;
        hasher.update(&anchor_buf);
        let raw = RawAnchor::decode(&anchor_buf)?;
        anchors.push(raw.into_high_level()?);
    }

    let computed = *hasher.finalize().as_bytes();
    if computed != checksum {
        return Err(MemvidError::InvalidTemporalTrack {
            reason: "checksum mismatch".into(),
        });
    }

    validate_mentions_sorted(&mentions)?;
    validate_anchors_sorted(&anchors)?;

    Ok(TemporalTrack {
        mentions,
        anchors,
        flags,
    })
}

/// Computes the deterministic checksum used for manifest verification.
pub fn calculate_checksum(
    mentions: &[TemporalMention],
    anchors: &[TemporalAnchor],
    flags: u32,
) -> [u8; 32] {
    let mut sorted_mentions = mentions.to_vec();
    sorted_mentions.sort_by(mention_cmp);
    let mut sorted_anchors = anchors.to_vec();
    sorted_anchors.sort_by_key(|anchor| anchor.frame_id);

    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&TEMPORAL_TRACK_MAGIC);
    header[4..6].copy_from_slice(&TEMPORAL_TRACK_VERSION.to_le_bytes());
    header[6..8].copy_from_slice(&(flags as u16).to_le_bytes());
    header[8..16].copy_from_slice(&(sorted_mentions.len() as u64).to_le_bytes());
    header[16..24].copy_from_slice(&(sorted_anchors.len() as u64).to_le_bytes());

    let mut hasher = Hasher::new();
    hasher.update(&header);

    for mention in sorted_mentions.iter().map(RawMention::from_high_level) {
        hasher.update(&mention.encode());
    }

    for anchor in sorted_anchors.iter().map(RawAnchor::from_high_level) {
        hasher.update(&anchor.encode());
    }

    *hasher.finalize().as_bytes()
}

/// Returns mention indices within `[start_utc, end_utc]` (inclusive).
#[must_use]
pub fn window(mentions: &[TemporalMention], start_utc: i64, end_utc: i64) -> Range<usize> {
    if mentions.is_empty() || start_utc > end_utc {
        return 0..0;
    }

    let lower = mentions.partition_point(|mention| mention.ts_utc < start_utc);
    let upper = mentions.partition_point(|mention| mention.ts_utc <= end_utc);
    lower..upper
}

fn mention_cmp(a: &TemporalMention, b: &TemporalMention) -> Ordering {
    a.ts_utc
        .cmp(&b.ts_utc)
        .then_with(|| a.frame_id.cmp(&b.frame_id))
        .then_with(|| a.byte_start.cmp(&b.byte_start))
}

fn validate_mentions_sorted(mentions: &[TemporalMention]) -> Result<()> {
    if mentions
        .windows(2)
        .any(|pair| mention_cmp(&pair[0], &pair[1]) == Ordering::Greater)
    {
        return Err(MemvidError::InvalidTemporalTrack {
            reason: "mentions not sorted".into(),
        });
    }
    Ok(())
}

fn validate_anchors_sorted(anchors: &[TemporalAnchor]) -> Result<()> {
    if anchors
        .windows(2)
        .any(|pair| pair[0].frame_id >= pair[1].frame_id)
    {
        return Err(MemvidError::InvalidTemporalTrack {
            reason: "anchors not strictly increasing".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempfile;

    #[test]
    fn append_and_read_roundtrip() {
        let mut mentions = vec![
            TemporalMention::new(
                10,
                1,
                0,
                5,
                TemporalMentionKind::Date,
                900,
                0,
                TemporalMentionFlags::empty(),
            ),
            TemporalMention::new(
                20,
                1,
                6,
                4,
                TemporalMentionKind::DateTime,
                800,
                60,
                TemporalMentionFlags::empty(),
            ),
            TemporalMention::new(
                20,
                2,
                0,
                3,
                TemporalMentionKind::RangeStart,
                750,
                0,
                TemporalMentionFlags(TemporalMentionFlags::HAS_RANGE),
            ),
        ];
        let mut anchors = vec![
            TemporalAnchor::new(1, 100, AnchorSource::FrameTimestamp),
            TemporalAnchor::new(3, 200, AnchorSource::Metadata),
        ];

        let mut file = tempfile().expect("temp");
        let (offset, length, checksum) =
            append_track(&mut file, &mut mentions, &mut anchors, 0).expect("append track");

        let track = read_track(&mut file, offset, length).expect("read track");
        assert_eq!(track.mentions.len(), 3);
        assert_eq!(track.anchors.len(), 2);
        assert_eq!(track.flags, 0);

        let expected_checksum = calculate_checksum(&track.mentions, &track.anchors, track.flags);
        assert_eq!(checksum, expected_checksum);
    }

    #[test]
    fn window_filters_correctly() {
        let mentions = vec![
            TemporalMention::new(
                10,
                1,
                0,
                1,
                TemporalMentionKind::Date,
                1000,
                0,
                TemporalMentionFlags::empty(),
            ),
            TemporalMention::new(
                12,
                1,
                0,
                1,
                TemporalMentionKind::Date,
                1000,
                0,
                TemporalMentionFlags::empty(),
            ),
            TemporalMention::new(
                15,
                1,
                0,
                1,
                TemporalMentionKind::Date,
                1000,
                0,
                TemporalMentionFlags::empty(),
            ),
        ];
        assert_eq!(window(&mentions, 9, 11), 0..1);
        assert_eq!(window(&mentions, 10, 15), 0..3);
        assert_eq!(window(&mentions, 16, 20), 3..3);
    }

    #[test]
    fn reject_unsorted_mentions() {
        let mentions = vec![
            TemporalMention::new(
                15,
                1,
                0,
                1,
                TemporalMentionKind::Date,
                1000,
                0,
                TemporalMentionFlags::empty(),
            ),
            TemporalMention::new(
                10,
                1,
                0,
                1,
                TemporalMentionKind::Date,
                1000,
                0,
                TemporalMentionFlags::empty(),
            ),
        ];
        assert!(matches!(
            validate_mentions_sorted(&mentions),
            Err(MemvidError::InvalidTemporalTrack { .. })
        ));
    }

    #[test]
    fn reject_unsorted_anchors() {
        let anchors = vec![
            TemporalAnchor::new(2, 100, AnchorSource::FrameTimestamp),
            TemporalAnchor::new(2, 110, AnchorSource::Metadata),
        ];
        assert!(matches!(
            validate_anchors_sorted(&anchors),
            Err(MemvidError::InvalidTemporalTrack { .. })
        ));
    }

    #[test]
    fn read_rejects_oversized_declared_length() {
        // Anything strictly above the safety ceiling must be rejected before
        // a body buffer is allocated.
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        let err = read_track(&mut cursor, 0, MAX_TEMPORAL_TRACK_BYTES + 1)
            .expect_err("oversized declared length must be rejected");
        match err {
            MemvidError::InvalidTemporalTrack { reason } => {
                assert!(reason.contains("length"), "unexpected reason: {reason}");
            }
            other => panic!("expected InvalidTemporalTrack, got {other:?}"),
        }
    }

    #[test]
    fn read_at_cap_does_not_eagerly_allocate_count_proportional_memory() {
        // Construct a header whose declared length sits just under
        // MAX_TEMPORAL_TRACK_BYTES with matching counts. This is the largest
        // value that passes the ceiling check and the count consistency check,
        // and is the case that previously allocated ~16 GiB body buffer plus
        // count-proportional Vec capacity before any record was decoded.
        //
        // With streaming reads + clamped Vec capacity, the function must fail
        // fast at the first body `read_exact` because only the header bytes
        // are present.
        let total_body = MAX_TEMPORAL_TRACK_BYTES - HEADER_SIZE as u64;
        let entry_count = total_body / MENTION_RECORD_SIZE as u64;
        let anchor_count = 0u64;
        let declared_length = HEADER_SIZE as u64 + entry_count * MENTION_RECORD_SIZE as u64;
        assert!(declared_length <= MAX_TEMPORAL_TRACK_BYTES);

        let flags: u16 = 0;
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&TEMPORAL_TRACK_MAGIC);
        header[4..6].copy_from_slice(&TEMPORAL_TRACK_VERSION.to_le_bytes());
        header[6..8].copy_from_slice(&flags.to_le_bytes());
        header[8..16].copy_from_slice(&entry_count.to_le_bytes());
        header[16..24].copy_from_slice(&anchor_count.to_le_bytes());
        // checksum bytes stay zero — the loop won't reach the checksum check
        // because the first body read_exact will fail first.

        let mut cursor = std::io::Cursor::new(header.to_vec());
        let err = read_track(&mut cursor, 0, declared_length)
            .expect_err("near-cap aligned length must fail at body read, not succeed");
        assert!(
            matches!(
                &err,
                MemvidError::Io { source, .. } if source.kind() == std::io::ErrorKind::UnexpectedEof
            ),
            "expected UnexpectedEof, got {err:?}",
        );
    }
}
