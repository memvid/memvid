//! Persistent manifest structures describing segments, indices, and TOC.

use serde::{
    Deserialize, Serialize,
    de::{self, SeqAccess, Visitor},
    ser::SerializeStruct,
};

use super::{common::FrameId, frame::Frame, ticket::TicketRef};

use std::{fmt, marker::PhantomData};

const MAX_TOC_SEGMENTS: usize = 1_000_000;
const MAX_TOC_FRAMES: usize = 10_000_000;
const MAX_SEGMENT_CATALOG_ENTRIES: usize = 1_000_000;

fn deserialize_vec_bounded<'de, D, T, const LIMIT: usize>(
    deserializer: D,
) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct BoundedVisitor<T, const LIMIT: usize>(PhantomData<T>);

    impl<'de, T, const LIMIT: usize> Visitor<'de> for BoundedVisitor<T, LIMIT>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence with a reasonable length")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            if let Some(size) = seq.size_hint() {
                if size > LIMIT {
                    return Err(de::Error::custom(format!(
                        "sequence length {size} exceeds bound {LIMIT}"
                    )));
                }
                let mut values = Vec::with_capacity(size.min(LIMIT));
                for i in 0..size {
                    let element = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                    values.push(element);
                }
                Ok(values)
            } else {
                let mut values = Vec::new();
                while let Some(element) = seq.next_element()? {
                    if values.len() == LIMIT {
                        return Err(de::Error::custom(format!(
                            "sequence length exceeds bound {LIMIT}"
                        )));
                    }
                    values.push(element);
                }
                Ok(values)
            }
        }
    }

    deserializer.deserialize_seq(BoundedVisitor::<T, LIMIT>(PhantomData))
}

fn deserialize_toc_segments<'de, D>(deserializer: D) -> Result<Vec<SegmentMeta>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_vec_bounded::<D, SegmentMeta, MAX_TOC_SEGMENTS>(deserializer)
}

fn deserialize_toc_frames<'de, D>(deserializer: D) -> Result<Vec<Frame>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_vec_bounded::<D, Frame, MAX_TOC_FRAMES>(deserializer)
}

fn deserialize_catalog_lex<'de, D>(deserializer: D) -> Result<Vec<LexSegmentDescriptor>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_vec_bounded::<D, LexSegmentDescriptor, MAX_SEGMENT_CATALOG_ENTRIES>(deserializer)
}

fn deserialize_catalog_vec<'de, D>(deserializer: D) -> Result<Vec<VecSegmentDescriptor>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_vec_bounded::<D, VecSegmentDescriptor, MAX_SEGMENT_CATALOG_ENTRIES>(deserializer)
}

fn deserialize_catalog_time<'de, D>(deserializer: D) -> Result<Vec<TimeSegmentDescriptor>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_vec_bounded::<D, TimeSegmentDescriptor, MAX_SEGMENT_CATALOG_ENTRIES>(deserializer)
}

fn deserialize_catalog_temporal<'de, D>(
    deserializer: D,
) -> Result<Vec<TemporalSegmentDescriptor>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_vec_bounded::<D, TemporalSegmentDescriptor, MAX_SEGMENT_CATALOG_ENTRIES>(
        deserializer,
    )
}

fn deserialize_catalog_tantivy<'de, D>(
    deserializer: D,
) -> Result<Vec<TantivySegmentDescriptor>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_vec_bounded::<D, TantivySegmentDescriptor, MAX_SEGMENT_CATALOG_ENTRIES>(
        deserializer,
    )
}

fn deserialize_catalog_index<'de, D>(deserializer: D) -> Result<Vec<IndexSegmentRef>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_vec_bounded::<D, IndexSegmentRef, MAX_SEGMENT_CATALOG_ENTRIES>(deserializer)
}

/// Incremental segment metadata catalogued within the TOC footer.
/// binary format compatibility. Feature flags control functionality, NOT structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentCatalog {
    /// Monotonically increasing identifier assigned to newly published segments.
    #[serde(default)]
    pub next_segment_id: u64,
    /// Optional catalog version for future schema evolution.
    #[serde(default)]
    pub version: u32,
    /// Flag indicating if lexical search has been enabled (persists even with no segments).
    #[serde(default)]
    pub lex_enabled: bool,
    /// Lexical search segments emitted by incremental commits.
    #[serde(default, deserialize_with = "deserialize_catalog_lex")]
    pub lex_segments: Vec<LexSegmentDescriptor>,
    /// Vector search segments emitted by incremental commits.
    #[serde(default, deserialize_with = "deserialize_catalog_vec")]
    pub vec_segments: Vec<VecSegmentDescriptor>,
    /// Time index segments emitted by incremental commits.
    #[serde(default, deserialize_with = "deserialize_catalog_time")]
    pub time_segments: Vec<TimeSegmentDescriptor>,
    /// Temporal mention segments emitted by incremental commits.
    /// ALWAYS present in struct - feature only controls if code uses it.
    #[serde(default, deserialize_with = "deserialize_catalog_temporal")]
    pub temporal_segments: Vec<TemporalSegmentDescriptor>,
    /// Tantivy (lexical) segments emitted by incremental commits.
    #[serde(default, deserialize_with = "deserialize_catalog_tantivy")]
    pub tantivy_segments: Vec<TantivySegmentDescriptor>,
    /// Unified append-only segment references (parallel builder).
    /// ALWAYS present in struct - feature only controls if code uses it.
    #[serde(default, deserialize_with = "deserialize_catalog_index")]
    pub index_segments: Vec<IndexSegmentRef>,
}

impl Default for SegmentCatalog {
    fn default() -> Self {
        Self {
            next_segment_id: 0,
            version: 1,
            lex_enabled: false,
            lex_segments: Vec::new(),
            vec_segments: Vec::new(),
            time_segments: Vec::new(),
            temporal_segments: Vec::new(),
            tantivy_segments: Vec::new(),
            index_segments: Vec::new(),
        }
    }
}

/// Shared metadata stored for an individual segment artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentCommon {
    pub segment_id: u64,
    pub bytes_offset: u64,
    pub bytes_length: u64,
    pub checksum: [u8; 32],
    #[serde(default)]
    pub build_sequence: u64,
    #[serde(default)]
    pub codec_version: u16,
    #[serde(default)]
    pub compression: SegmentCompression,
    #[serde(default)]
    pub span: Option<SegmentSpan>,
}

impl SegmentCommon {
    #[must_use]
    pub fn new(segment_id: u64, bytes_offset: u64, bytes_length: u64, checksum: [u8; 32]) -> Self {
        Self {
            segment_id,
            bytes_offset,
            bytes_length,
            checksum,
            build_sequence: 0,
            codec_version: 0,
            compression: SegmentCompression::None,
            span: None,
        }
    }
}

/// Logical span covered by a sealed segment.
#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SegmentSpan {
    pub frame_start: FrameId,
    pub frame_end: FrameId,
    #[serde(default)]
    pub page_start: u32,
    #[serde(default)]
    pub page_end: u32,
    #[serde(default)]
    pub token_start: u64,
    #[serde(default)]
    pub token_end: u64,
}

/// Lightweight descriptor stored in the manifest WAL for append-only segments.
/// Always defined for backwards compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSegmentRef {
    pub kind: SegmentKind,
    pub common: SegmentCommon,
    #[serde(default)]
    pub stats: SegmentStats,
}

/// Segment category emitted by the parallel builder.
/// Always defined for backwards compatibility.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SegmentKind {
    #[default]
    Lexical,
    Vector,
    Time,
    Temporal,
    Tantivy,
}

/// Build-time metrics captured for a sealed segment.
/// Always defined for backwards compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SegmentStats {
    pub doc_count: u64,
    pub vector_count: u64,
    pub time_entries: u64,
    pub bytes_uncompressed: u64,
    pub build_micros: u64,
}

/// Manifest entry describing a lexical index segment.
#[derive(Debug, Clone)]
pub struct LexSegmentDescriptor {
    pub common: SegmentCommon,
    pub doc_count: u64,
}

impl LexSegmentDescriptor {
    #[must_use]
    pub fn from_common(common: SegmentCommon, doc_count: u64) -> Self {
        Self { common, doc_count }
    }
}

impl Serialize for LexSegmentDescriptor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("LexSegmentDescriptor", 9)?;
        state.serialize_field("segment_id", &self.common.segment_id)?;
        state.serialize_field("bytes_offset", &self.common.bytes_offset)?;
        state.serialize_field("bytes_length", &self.common.bytes_length)?;
        state.serialize_field("checksum", &self.common.checksum)?;
        state.serialize_field("build_sequence", &self.common.build_sequence)?;
        state.serialize_field("codec_version", &self.common.codec_version)?;
        state.serialize_field("compression", &self.common.compression)?;
        state.serialize_field("span", &self.common.span)?;
        state.serialize_field("doc_count", &self.doc_count)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for LexSegmentDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            segment_id: u64,
            bytes_offset: u64,
            bytes_length: u64,
            checksum: [u8; 32],
            #[serde(default)]
            build_sequence: u64,
            #[serde(default)]
            codec_version: u16,
            #[serde(default)]
            compression: SegmentCompression,
            #[serde(default)]
            span: Option<SegmentSpan>,
            doc_count: u64,
        }

        let repr = Repr::deserialize(deserializer)?;
        let mut common = SegmentCommon::new(
            repr.segment_id,
            repr.bytes_offset,
            repr.bytes_length,
            repr.checksum,
        );
        common.build_sequence = repr.build_sequence;
        common.codec_version = repr.codec_version;
        common.compression = repr.compression;
        common.span = repr.span;
        Ok(Self {
            common,
            doc_count: repr.doc_count,
        })
    }
}

/// Manifest entry describing a vector index segment.
#[derive(Debug, Clone)]
pub struct VecSegmentDescriptor {
    pub common: SegmentCommon,
    pub vector_count: u64,
    pub dimension: u32,
    pub vector_compression: VectorCompression,
}

impl VecSegmentDescriptor {
    #[must_use]
    pub fn from_common(
        common: SegmentCommon,
        vector_count: u64,
        dimension: u32,
        vector_compression: VectorCompression,
    ) -> Self {
        Self {
            common,
            vector_count,
            dimension,
            vector_compression,
        }
    }
}

impl Serialize for VecSegmentDescriptor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("VecSegmentDescriptor", 11)?;
        state.serialize_field("segment_id", &self.common.segment_id)?;
        state.serialize_field("bytes_offset", &self.common.bytes_offset)?;
        state.serialize_field("bytes_length", &self.common.bytes_length)?;
        state.serialize_field("checksum", &self.common.checksum)?;
        state.serialize_field("build_sequence", &self.common.build_sequence)?;
        state.serialize_field("codec_version", &self.common.codec_version)?;
        state.serialize_field("compression", &self.common.compression)?;
        state.serialize_field("span", &self.common.span)?;
        state.serialize_field("vector_count", &self.vector_count)?;
        state.serialize_field("dimension", &self.dimension)?;
        state.serialize_field("vector_compression", &self.vector_compression)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for VecSegmentDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            segment_id: u64,
            bytes_offset: u64,
            bytes_length: u64,
            checksum: [u8; 32],
            #[serde(default)]
            build_sequence: u64,
            #[serde(default)]
            codec_version: u16,
            #[serde(default)]
            compression: SegmentCompression,
            #[serde(default)]
            span: Option<SegmentSpan>,
            vector_count: u64,
            dimension: u32,
            #[serde(default)]
            vector_compression: VectorCompression,
        }

        let repr = Repr::deserialize(deserializer)?;
        let mut common = SegmentCommon::new(
            repr.segment_id,
            repr.bytes_offset,
            repr.bytes_length,
            repr.checksum,
        );
        common.build_sequence = repr.build_sequence;
        common.codec_version = repr.codec_version;
        common.compression = repr.compression;
        common.span = repr.span;
        Ok(Self {
            common,
            vector_count: repr.vector_count,
            dimension: repr.dimension,
            vector_compression: repr.vector_compression,
        })
    }
}

/// Manifest entry describing a time index segment.
#[derive(Debug, Clone)]
pub struct TimeSegmentDescriptor {
    pub common: SegmentCommon,
    pub entry_count: u64,
}

impl TimeSegmentDescriptor {
    #[must_use]
    pub fn from_common(common: SegmentCommon, entry_count: u64) -> Self {
        Self {
            common,
            entry_count,
        }
    }
}

impl Serialize for TimeSegmentDescriptor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("TimeSegmentDescriptor", 9)?;
        state.serialize_field("segment_id", &self.common.segment_id)?;
        state.serialize_field("bytes_offset", &self.common.bytes_offset)?;
        state.serialize_field("bytes_length", &self.common.bytes_length)?;
        state.serialize_field("checksum", &self.common.checksum)?;
        state.serialize_field("build_sequence", &self.common.build_sequence)?;
        state.serialize_field("codec_version", &self.common.codec_version)?;
        state.serialize_field("compression", &self.common.compression)?;
        state.serialize_field("span", &self.common.span)?;
        state.serialize_field("entry_count", &self.entry_count)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TimeSegmentDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            segment_id: u64,
            bytes_offset: u64,
            bytes_length: u64,
            checksum: [u8; 32],
            #[serde(default)]
            build_sequence: u64,
            #[serde(default)]
            codec_version: u16,
            #[serde(default)]
            compression: SegmentCompression,
            #[serde(default)]
            span: Option<SegmentSpan>,
            entry_count: u64,
        }

        let repr = Repr::deserialize(deserializer)?;
        let mut common = SegmentCommon::new(
            repr.segment_id,
            repr.bytes_offset,
            repr.bytes_length,
            repr.checksum,
        );
        common.build_sequence = repr.build_sequence;
        common.codec_version = repr.codec_version;
        common.compression = repr.compression;
        common.span = repr.span;
        Ok(Self {
            common,
            entry_count: repr.entry_count,
        })
    }
}

/// Always defined for backwards compatibility.
#[derive(Debug, Clone)]
pub struct TemporalSegmentDescriptor {
    pub common: SegmentCommon,
    pub entry_count: u64,
    pub anchor_count: u64,
    pub flags: u32,
}

impl TemporalSegmentDescriptor {
    #[must_use]
    pub fn from_common(
        common: SegmentCommon,
        entry_count: u64,
        anchor_count: u64,
        flags: u32,
    ) -> Self {
        Self {
            common,
            entry_count,
            anchor_count,
            flags,
        }
    }
}

impl Serialize for TemporalSegmentDescriptor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("TemporalSegmentDescriptor", 11)?;
        state.serialize_field("segment_id", &self.common.segment_id)?;
        state.serialize_field("bytes_offset", &self.common.bytes_offset)?;
        state.serialize_field("bytes_length", &self.common.bytes_length)?;
        state.serialize_field("checksum", &self.common.checksum)?;
        state.serialize_field("build_sequence", &self.common.build_sequence)?;
        state.serialize_field("codec_version", &self.common.codec_version)?;
        state.serialize_field("compression", &self.common.compression)?;
        state.serialize_field("span", &self.common.span)?;
        state.serialize_field("entry_count", &self.entry_count)?;
        state.serialize_field("anchor_count", &self.anchor_count)?;
        state.serialize_field("flags", &self.flags)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TemporalSegmentDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            segment_id: u64,
            bytes_offset: u64,
            bytes_length: u64,
            checksum: [u8; 32],
            #[serde(default)]
            build_sequence: u64,
            #[serde(default)]
            codec_version: u16,
            #[serde(default)]
            compression: SegmentCompression,
            #[serde(default)]
            span: Option<SegmentSpan>,
            entry_count: u64,
            anchor_count: u64,
            #[serde(default)]
            flags: u32,
        }

        let repr = Repr::deserialize(deserializer)?;
        let mut common = SegmentCommon::new(
            repr.segment_id,
            repr.bytes_offset,
            repr.bytes_length,
            repr.checksum,
        );
        common.build_sequence = repr.build_sequence;
        common.codec_version = repr.codec_version;
        common.compression = repr.compression;
        common.span = repr.span;
        Ok(Self {
            common,
            entry_count: repr.entry_count,
            anchor_count: repr.anchor_count,
            flags: repr.flags,
        })
    }
}

/// Manifest entry describing an embedded Tantivy (lexical) segment file.
#[derive(Debug, Clone)]
pub struct TantivySegmentDescriptor {
    pub common: SegmentCommon,
    pub path: String,
}

impl TantivySegmentDescriptor {
    #[must_use]
    pub fn from_common(common: SegmentCommon, path: String) -> Self {
        Self { common, path }
    }
}

impl Serialize for TantivySegmentDescriptor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("TantivySegmentDescriptor", 9)?;
        state.serialize_field("segment_id", &self.common.segment_id)?;
        state.serialize_field("bytes_offset", &self.common.bytes_offset)?;
        state.serialize_field("bytes_length", &self.common.bytes_length)?;
        state.serialize_field("checksum", &self.common.checksum)?;
        state.serialize_field("build_sequence", &self.common.build_sequence)?;
        state.serialize_field("codec_version", &self.common.codec_version)?;
        state.serialize_field("compression", &self.common.compression)?;
        state.serialize_field("span", &self.common.span)?;
        state.serialize_field("path", &self.path)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TantivySegmentDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            segment_id: u64,
            bytes_offset: u64,
            bytes_length: u64,
            checksum: [u8; 32],
            #[serde(default)]
            build_sequence: u64,
            #[serde(default)]
            codec_version: u16,
            #[serde(default)]
            compression: SegmentCompression,
            #[serde(default)]
            span: Option<SegmentSpan>,
            path: String,
        }

        let repr = Repr::deserialize(deserializer)?;
        let mut common = SegmentCommon::new(
            repr.segment_id,
            repr.bytes_offset,
            repr.bytes_length,
            repr.checksum,
        );
        common.build_sequence = repr.build_sequence;
        common.codec_version = repr.codec_version;
        common.compression = repr.compression;
        common.span = repr.span;
        Ok(Self {
            common,
            path: repr.path,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexManifests {
    pub lex: Option<LexIndexManifest>,
    #[serde(default)]
    pub lex_segments: Vec<LexSegmentManifest>,
    pub vec: Option<VecIndexManifest>,
    /// CLIP visual embeddings index (separate from text vec index due to different dimensions)
    #[serde(default)]
    pub clip: Option<crate::clip::ClipIndexManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexIndexManifest {
    pub doc_count: u64,
    #[serde(default)]
    pub generation: u64,
    pub bytes_offset: u64,
    pub bytes_length: u64,
    pub checksum: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexSegmentManifest {
    pub path: String,
    pub bytes_offset: u64,
    pub bytes_length: u64,
    pub checksum: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum VectorCompression {
    #[default]
    None, // Full f32 vectors (1,536 bytes for 384 dims)
    Pq96, // Product quantization with 96 subspaces (96 bytes)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecIndexManifest {
    pub vector_count: u64,
    pub dimension: u32,
    pub bytes_offset: u64,
    pub bytes_length: u64,
    pub checksum: [u8; 32],
    /// Compression mode for vector storage (default: None for backward compatibility)
    #[serde(default)]
    pub compression_mode: VectorCompression,
    /// Model used to generate embeddings (e.g., "openai-text-embedding-3-small").
    /// Added in v2 to prevent model mismatch.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SegmentCompression {
    #[default]
    None,
    Zstd,
    Lz4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub id: u64,
    pub frame_range: (FrameId, FrameId),
    pub primary_checksum: [u8; 32],
    pub compression: SegmentCompression,
    pub bytes_offset: u64,
    pub bytes_length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u16,
    pub footer_offset: u64,
    pub wal_offset: u64,
    pub wal_size: u64,
    pub wal_checkpoint_pos: u64,
    pub wal_sequence: u64,
    pub toc_checksum: [u8; 32],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Toc {
    pub toc_version: u64,
    #[serde(deserialize_with = "deserialize_toc_segments")]
    pub segments: Vec<SegmentMeta>,
    #[serde(deserialize_with = "deserialize_toc_frames")]
    pub frames: Vec<Frame>,
    pub indexes: IndexManifests,
    pub time_index: Option<TimeIndexManifest>,
    /// Always present for backwards compatibility, even if feature is disabled.
    #[serde(default)]
    pub temporal_track: Option<TemporalTrackManifest>,
    /// Structured memory cards track (facts, preferences, events, etc.).
    #[serde(default)]
    pub memories_track: Option<MemoriesTrackManifest>,
    /// Logic-Mesh graph track (entities and relationships for graph traversal).
    #[serde(default)]
    pub logic_mesh: Option<LogicMeshManifest>,
    /// Sketch track for fast candidate generation (`SimHash` + term filters).
    #[serde(default)]
    pub sketch_track: Option<SketchTrackManifest>,
    #[serde(default)]
    pub segment_catalog: SegmentCatalog,
    pub ticket_ref: TicketRef,
    /// Optional memory binding to dashboard
    #[serde(default)]
    pub memory_binding: Option<super::MemoryBinding>,
    /// Optional replay manifest for time-travel debugging sessions
    /// Always present in struct for backward compatibility - feature controls functionality
    #[serde(default)]
    pub replay_manifest: Option<crate::replay::ReplayManifest>,
    /// Enrichment queue for progressive ingestion.
    /// Tracks frames needing background Phase 2 work (full extraction + embeddings).
    #[serde(default)]
    pub enrichment_queue: EnrichmentQueueManifest,
    pub merkle_root: [u8; 32],
    pub toc_checksum: [u8; 32],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimeIndexManifest {
    pub bytes_offset: u64,
    pub bytes_length: u64,
    pub entry_count: u64,
    pub checksum: [u8; 32],
}

/// Always defined for backwards compatibility.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TemporalTrackManifest {
    pub bytes_offset: u64,
    pub bytes_length: u64,
    pub entry_count: u64,
    pub anchor_count: u64,
    pub checksum: [u8; 32],
    #[serde(default)]
    pub flags: u32,
}

/// Manifest for the memories track (structured memory cards).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemoriesTrackManifest {
    /// Offset to the compressed memories track data in the file.
    pub bytes_offset: u64,
    /// Length of the compressed memories track data.
    pub bytes_length: u64,
    /// Number of memory cards in the track.
    pub card_count: u64,
    /// Number of unique entities in the track.
    pub entity_count: u64,
    /// BLAKE3 checksum of the uncompressed data.
    pub checksum: [u8; 32],
}

/// Manifest for the Logic-Mesh graph track (entities and relationships).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogicMeshManifest {
    /// Offset to the compressed Logic-Mesh data in the file.
    pub bytes_offset: u64,
    /// Length of the compressed Logic-Mesh data.
    pub bytes_length: u64,
    /// Number of entity nodes in the mesh.
    pub node_count: u64,
    /// Number of relationship edges in the mesh.
    pub edge_count: u64,
    /// BLAKE3 checksum of the uncompressed data.
    pub checksum: [u8; 32],
}

/// Manifest for the Sketch Track (fast candidate generation).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SketchTrackManifest {
    /// Offset to the sketch track data in the file.
    pub bytes_offset: u64,
    /// Length of the sketch track data.
    pub bytes_length: u64,
    /// Number of sketch entries (one per frame).
    pub entry_count: u64,
    /// Entry size in bytes (32/64/96 for Small/Medium/Large variants).
    pub entry_size: u16,
    /// Feature flags.
    pub flags: u32,
    /// BLAKE3 checksum of the track data.
    pub checksum: [u8; 32],
}

/// Manifest for the enrichment queue (progressive ingestion).
///
/// Tracks frames that need background enrichment (Phase 2: full extraction + embeddings).
/// Persisted in TOC so enrichment can resume after crash/restart.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct EnrichmentQueueManifest {
    /// Frame IDs pending enrichment, with their progress.
    pub tasks: Vec<super::common::EnrichmentTask>,
    /// Timestamp of last queue modification.
    pub updated_at: u64,
}

impl EnrichmentQueueManifest {
    /// Create a new empty enrichment queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            updated_at: 0,
        }
    }

    /// Add a frame to the enrichment queue.
    pub fn push(&mut self, frame_id: super::common::FrameId) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.tasks.push(super::common::EnrichmentTask {
            frame_id,
            created_at: now,
            chunks_done: 0,
            chunks_total: 0,
        });
        self.updated_at = now;
    }

    /// Remove a frame from the enrichment queue (after completion).
    pub fn remove(&mut self, frame_id: super::common::FrameId) {
        self.tasks.retain(|t| t.frame_id != frame_id);
        use std::time::{SystemTime, UNIX_EPOCH};
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Update checkpoint for a frame's embedding progress.
    pub fn update_checkpoint(
        &mut self,
        frame_id: super::common::FrameId,
        chunks_done: u32,
        chunks_total: u32,
    ) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.frame_id == frame_id) {
            task.chunks_done = chunks_done;
            task.chunks_total = chunks_total;
        }
    }

    /// Check if any enrichment work is pending.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Get count of pending tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }
}
