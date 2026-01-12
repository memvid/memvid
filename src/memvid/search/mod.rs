//! Search orchestration for `Memvid`.
//!
//! The search entrypoint chooses an engine (Tantivy lexical, optional temporal filters,
//! and lex-only fallback) and returns fully decorated snippets with chunk metadata.
//! Invariants: refuses empty queries, respects cursor/limit bounds, and never mutates
//! the underlying file.

#[cfg(feature = "lex")]
use std::collections::{BTreeSet, HashSet};
#[cfg(feature = "lex")]
use std::time::Instant;

use crate::memvid::lifecycle::Memvid;
use crate::types::{FrameId, SearchEngineKind, SearchParams, SearchRequest, SearchResponse};
use crate::{MemvidError, Result};

mod api;
mod builders;
#[cfg(feature = "lex")]
mod fallback;
pub(crate) mod helpers;
#[cfg(feature = "lex")]
mod tantivy;
#[cfg(any(feature = "lex", feature = "temporal_track"))]
mod time_filter;

// Re-export text indexability helpers for use in validation
pub use api::{
    DEFAULT_MAX_INDEX_PAYLOAD, is_frame_text_indexable, is_text_indexable_mime, max_index_payload,
};

#[cfg(feature = "lex")]
use fallback::{search_with_filters_only, search_with_lex_fallback};
use helpers::empty_search_response;
#[cfg(feature = "lex")]
pub use tantivy::parse_content_date_to_timestamp;
#[cfg(feature = "lex")]
use tantivy::try_tantivy_search;
#[cfg(feature = "temporal_track")]
pub use time_filter::frame_ids_for_temporal_filter;
#[cfg(feature = "lex")]
use time_filter::frame_ids_in_date_range;

#[cfg(feature = "lex")]
impl Memvid {
    pub fn search(&mut self, request: SearchRequest) -> Result<SearchResponse> {
        if !self.lex_enabled {
            return Err(MemvidError::LexNotEnabled);
        }

        let start_time = Instant::now();
        // parse_query can return structured tokens; we only keep non-empty, lower-cased terms.
        let parsed = crate::search::parse_query(&request.query)?;
        let mut query_tokens = parsed.text_tokens();
        query_tokens.retain(|token| !token.trim().is_empty());
        query_tokens = query_tokens
            .into_iter()
            .map(|token| token.to_ascii_lowercase())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let has_text_terms = !query_tokens.is_empty();
        let has_field_terms = parsed.contains_field_terms();

        if !has_text_terms && !has_field_terms {
            return Err(MemvidError::InvalidQuery {
                reason: "query must include at least one search term or field filter".into(),
            });
        }

        let params = SearchParams {
            top_k: request.top_k,
            snippet_chars: request.snippet_chars,
            cursor: request.cursor.clone(),
        };

        let date_range = parsed.required_date_range();
        #[allow(unused_mut)]
        let mut candidate_filter: Option<HashSet<FrameId>> = if let Some(ref range) = date_range {
            if range.is_empty() {
                let elapsed = start_time.elapsed().as_millis();
                return Ok(empty_search_response(
                    request.query.clone(),
                    params.clone(),
                    elapsed,
                    SearchEngineKind::Tantivy,
                ));
            }
            match frame_ids_in_date_range(self, range)? {
                Some(ids) => {
                    if ids.is_empty() {
                        let elapsed = start_time.elapsed().as_millis();
                        return Ok(empty_search_response(
                            request.query.clone(),
                            params.clone(),
                            elapsed,
                            SearchEngineKind::Tantivy,
                        ));
                    }
                    Some(ids.into_iter().collect())
                }
                None => None,
            }
        } else {
            None
        };

        #[cfg(feature = "temporal_track")]
        if let Some(ref temporal_filter) = request.temporal {
            if !temporal_filter.is_empty() {
                match time_filter::frame_ids_for_temporal_filter(self, temporal_filter)? {
                    Some(ids) => {
                        if ids.is_empty() {
                            let elapsed = start_time.elapsed().as_millis();
                            return Ok(empty_search_response(
                                request.query.clone(),
                                params.clone(),
                                elapsed,
                                SearchEngineKind::Tantivy,
                            ));
                        }
                        let new_set: HashSet<FrameId> = ids.into_iter().collect();
                        candidate_filter = match candidate_filter {
                            Some(existing) => {
                                let filtered: HashSet<FrameId> = existing
                                    .into_iter()
                                    .filter(|id| new_set.contains(id))
                                    .collect();
                                if filtered.is_empty() {
                                    let elapsed = start_time.elapsed().as_millis();
                                    return Ok(empty_search_response(
                                        request.query.clone(),
                                        params.clone(),
                                        elapsed,
                                        SearchEngineKind::Tantivy,
                                    ));
                                }
                                Some(filtered)
                            }
                            None => Some(new_set),
                        };
                    }
                    None => {}
                }
            }
        }

        // REPLAY: Filter by as_of_frame or as_of_ts for time-travel views
        if request.as_of_frame.is_some() || request.as_of_ts.is_some() {
            let replay_ids = self.get_replay_frame_ids(&request)?;
            if replay_ids.is_empty() {
                let elapsed = start_time.elapsed().as_millis();
                return Ok(empty_search_response(
                    request.query.clone(),
                    params.clone(),
                    elapsed,
                    SearchEngineKind::Tantivy,
                ));
            }
            let replay_set: HashSet<FrameId> = replay_ids.into_iter().collect();
            candidate_filter = match candidate_filter {
                Some(existing) => {
                    let filtered: HashSet<FrameId> = existing
                        .into_iter()
                        .filter(|id| replay_set.contains(id))
                        .collect();
                    if filtered.is_empty() {
                        let elapsed = start_time.elapsed().as_millis();
                        return Ok(empty_search_response(
                            request.query.clone(),
                            params.clone(),
                            elapsed,
                            SearchEngineKind::Tantivy,
                        ));
                    }
                    Some(filtered)
                }
                None => Some(replay_set),
            };
        }

        // FRAME FILTER: Filter by specific frame IDs if provided
        if !request.frames.is_empty() {
            let frames_set: HashSet<FrameId> = request.frames.iter().copied().collect();
            candidate_filter = match candidate_filter {
                Some(existing) => {
                    let filtered: HashSet<FrameId> = existing
                        .into_iter()
                        .filter(|id| frames_set.contains(id))
                        .collect();
                    if filtered.is_empty() {
                        let elapsed = start_time.elapsed().as_millis();
                        return Ok(empty_search_response(
                            request.query.clone(),
                            params.clone(),
                            elapsed,
                            SearchEngineKind::Tantivy,
                        ));
                    }
                    Some(filtered)
                }
                None => Some(frames_set),
            };
        }

        // SKETCH PRE-FILTER: Use sketch track for fast candidate generation if available
        // This dramatically reduces the number of documents sent to BM25/Tantivy
        if self.has_sketches() && has_text_terms && !request.no_sketch {
            let sketch_start = Instant::now();
            let sketch_options = crate::SketchSearchOptions {
                // Use relaxed threshold for better recall - BM25 will rerank anyway
                hamming_threshold: 32,
                // Get more candidates than needed - BM25 will select the best
                max_candidates: (params.top_k * 10).max(500),
                min_score: 0.0,
            };
            let sketch_candidates =
                self.find_sketch_candidates(&request.query, Some(sketch_options));

            if !sketch_candidates.is_empty() {
                let sketch_set: HashSet<FrameId> =
                    sketch_candidates.iter().map(|c| c.frame_id).collect();

                tracing::debug!(
                    sketch_candidates = sketch_candidates.len(),
                    sketch_time_us = sketch_start.elapsed().as_micros(),
                    "sketch pre-filter applied"
                );

                candidate_filter = match candidate_filter {
                    Some(existing) => {
                        // Intersection: keep only IDs that pass both filters
                        let filtered: HashSet<FrameId> = existing
                            .into_iter()
                            .filter(|id| sketch_set.contains(id))
                            .collect();
                        if filtered.is_empty() {
                            // Fall back to sketch-only if intersection is empty
                            Some(sketch_set)
                        } else {
                            Some(filtered)
                        }
                    }
                    None => Some(sketch_set),
                };
            }
        }

        let mut response = if let Some(response) = try_tantivy_search(
            self,
            &parsed,
            &query_tokens,
            &request,
            &params,
            start_time,
            candidate_filter.as_ref(),
        )? {
            response
        } else {
            self.ensure_lex_index()?;
            if has_text_terms {
                search_with_lex_fallback(
                    self,
                    &parsed,
                    &query_tokens,
                    &request,
                    &params,
                    start_time,
                    candidate_filter.as_ref(),
                )?
            } else {
                search_with_filters_only(
                    self,
                    &parsed,
                    &request,
                    &params,
                    start_time,
                    candidate_filter.as_ref(),
                )?
            }
        };

        // Enrich hits with Logic-Mesh entities if mesh is available
        if self.has_logic_mesh() {
            helpers::enrich_hits_with_entities(&mut response.hits, self);
        }

        // Record the search action if a replay session is active
        #[cfg(feature = "replay")]
        {
            let result_frames: Vec<u64> = response.hits.iter().map(|h| h.frame_id).collect();
            self.record_find_action(
                &request.query,
                &format!("{:?}", response.engine),
                response.hits.len(),
                result_frames,
            );
        }

        Ok(response)
    }
}

#[cfg(not(feature = "lex"))]
impl Memvid {
    pub fn search(&mut self, _request: SearchRequest) -> Result<SearchResponse> {
        Err(MemvidError::LexNotEnabled)
    }
}
