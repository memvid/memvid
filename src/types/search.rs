//! Public search request/response types exposed by the core library.

use serde::{Deserialize, Serialize};

use super::common::FrameId;
#[cfg(feature = "temporal_track")]
use super::frame::AnchorSource;
#[cfg(feature = "temporal_track")]
use super::temporal::{TemporalFilter, TemporalMentionFlags, TemporalMentionKind};

/// Parameters used to page and shape search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    /// Maximum hits to return.
    pub top_k: usize,
    /// Number of characters to capture around matches.
    pub snippet_chars: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Cursor token for pagination.
    pub cursor: Option<String>,
}

/// Engine selected to satisfy a search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchEngineKind {
    Tantivy,
    LexFallback,
    Hybrid,
}

impl Default for SearchEngineKind {
    fn default() -> Self {
        Self::LexFallback
    }
}

/// Search request accepted by the core; supports lexical, hybrid, and temporal filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Query string (lexical or semantic depending on engine).
    pub query: String,
    /// Maximum hits to return.
    pub top_k: usize,
    /// Number of characters to capture around matches.
    pub snippet_chars: usize,
    #[serde(default)]
    /// Restrict search to a specific URI.
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    /// Restrict search to specific frame IDs.
    pub frames: Vec<FrameId>,
    #[serde(default)]
    /// Restrict search to a named scope/collection.
    pub scope: Option<String>,
    #[serde(default)]
    /// Pagination cursor.
    pub cursor: Option<String>,
    #[cfg(feature = "temporal_track")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalFilter>,
    #[serde(default)]
    /// Replay: Filter to frames with id <= as_of_frame (time-travel view).
    pub as_of_frame: Option<FrameId>,
    #[serde(default)]
    /// Replay: Filter to frames with timestamp <= as_of_ts (time-travel view).
    pub as_of_ts: Option<i64>,
    #[serde(default)]
    /// Disable sketch pre-filtering for this query.
    pub no_sketch: bool,
}

/// A single ranked hit with snippet metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub rank: usize,
    pub frame_id: FrameId,
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub range: (usize, usize),
    pub text: String,
    pub matches: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_range: Option<(usize, usize)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<SearchHitMetadata>,
}

/// Entity reference in search hit metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHitEntity {
    /// Entity display name.
    pub name: String,
    /// Entity kind (person, organization, etc.).
    pub kind: String,
    /// Confidence score (0.0-1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Optional per-hit metadata (tags, labels, dates, temporal context).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchHitMetadata {
    #[serde(default)]
    pub matches: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content_dates: Vec<String>,
    /// Entities mentioned in this search hit (from Logic-Mesh).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<SearchHitEntity>,
    #[cfg(feature = "temporal_track")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<SearchHitTemporal>,
}

#[cfg(feature = "temporal_track")]
/// Temporal annotations attached to a hit when temporal tracking is enabled.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchHitTemporal {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<SearchHitTemporalAnchor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mentions: Vec<SearchHitTemporalMention>,
}

#[cfg(feature = "temporal_track")]
/// Anchor timestamp for a temporal hit (absolute and ISO strings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHitTemporalAnchor {
    pub ts_utc: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iso_8601: Option<String>,
    pub source: AnchorSource,
}

#[cfg(feature = "temporal_track")]
/// Temporal mention (range or instant) extracted from the document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHitTemporalMention {
    pub ts_utc: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iso_8601: Option<String>,
    pub kind: TemporalMentionKind,
    pub confidence: u16,
    pub flags: TemporalMentionFlags,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub byte_start: u32,
    pub byte_len: u32,
}

/// Full search response with hits, params, engine, and an optional cursor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Query echoed back for clients.
    pub query: String,
    /// Milliseconds spent satisfying the request.
    pub elapsed_ms: u128,
    /// Total hits found (without pagination applied).
    pub total_hits: usize,
    /// Parameters used for this request, including cursors.
    pub params: SearchParams,
    /// Ranked hits.
    pub hits: Vec<SearchHit>,
    /// Concatenated snippets or context paragraphs.
    pub context: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Cursor for fetching the next page, if any.
    pub next_cursor: Option<String>,
    #[serde(default)]
    /// Engine responsible for the results.
    pub engine: SearchEngineKind,
}
