use serde::{Deserialize, Serialize};

use super::adaptive::AdaptiveConfig;
use super::common::FrameId;
#[cfg(feature = "temporal_track")]
use super::search::SearchHitTemporal;
use super::search::SearchResponse;
#[cfg(feature = "temporal_track")]
use super::temporal::TemporalFilter;
use crate::Result;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AskMode {
    /// Lexical-only retrieval.
    Lex,
    /// Semantic-only retrieval.
    Sem,
    /// Hybrid (lexical + semantic) retrieval.
    Hybrid,
}

impl Default for AskMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AskRetriever {
    /// Lexical-only retriever.
    Lex,
    /// Semantic-only retriever.
    Semantic,
    /// Hybrid retriever with reranking.
    Hybrid,
    /// Lexical fallback when semantic paths are unavailable.
    LexFallback,
    /// Timeline fallback when search returns no results.
    TimelineFallback,
}

/// Request payload for retrieval + synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskRequest {
    pub question: String,
    pub top_k: usize,
    pub snippet_chars: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    /// Restrict search to specific frame IDs.
    pub frames: Vec<FrameId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<i64>,
    #[cfg(feature = "temporal_track")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalFilter>,
    #[serde(default)]
    pub context_only: bool,
    #[serde(default)]
    pub mode: AskMode,
    #[serde(default)]
    /// Replay: Filter to frames with id <= as_of_frame (time-travel view).
    pub as_of_frame: Option<FrameId>,
    #[serde(default)]
    /// Replay: Filter to frames with timestamp <= as_of_ts (time-travel view).
    pub as_of_ts: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Adaptive retrieval configuration. When set, dynamically determines how many
    /// results to retrieve based on relevancy score distribution.
    pub adaptive: Option<AdaptiveConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AskStats {
    /// Time spent retrieving context in milliseconds.
    pub retrieval_ms: u128,
    /// Time spent synthesizing the answer in milliseconds.
    pub synthesis_ms: u128,
    /// End-to-end latency in milliseconds.
    pub latency_ms: u128,
}

/// Structured citation pointing back into the memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskCitation {
    pub index: usize,
    pub frame_id: FrameId,
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_range: Option<(usize, usize)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// Fragment of retrieval context sent to a synthesizer (with ranges and optional temporal info).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskContextFragment {
    pub rank: usize,
    pub frame_id: FrameId,
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(default)]
    pub matches: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<(usize, usize)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_range: Option<(usize, usize)>,
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<AskContextFragmentKind>,
    #[cfg(feature = "temporal_track")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<SearchHitTemporal>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AskContextFragmentKind {
    /// Full span of text used for synthesis.
    Full,
    /// Summarized span of text passed to the synthesizer.
    Summary,
}

/// Response for `ask` containing retrieval context, optional answer, citations, and timings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskResponse {
    pub question: String,
    pub mode: AskMode,
    pub retriever: AskRetriever,
    pub context_only: bool,
    pub retrieval: SearchResponse,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<AskCitation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context_fragments: Vec<AskContextFragment>,
    pub stats: AskStats,
}

pub trait VecEmbedder {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;

    fn embed_chunks(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed_query(text)?);
        }
        Ok(embeddings)
    }

    fn embedding_dimension(&self) -> usize;
}
