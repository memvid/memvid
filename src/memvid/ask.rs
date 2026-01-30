// Safe unwrap: float comparisons with fallback ordering.
#![allow(clippy::unwrap_used)]
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::num::NonZeroU64;
use std::time::Instant;

use crate::memvid::lifecycle::Memvid;
use crate::memvid::search::helpers::{build_context, reorder_hits_by_token_matches};
#[cfg(feature = "temporal_track")]
use crate::types::TemporalFilter;
use crate::types::{
    AskCitation, AskContextFragment, AskContextFragmentKind, AskMode, AskRequest, AskResponse,
    AskRetriever, AskStats, SearchEngineKind, SearchHit, SearchParams, SearchRequest,
    SearchResponse, TimelineQueryBuilder,
};
use crate::{MemvidError, Result, VecEmbedder};

const RRF_K: f32 = 60.0;

#[cfg(feature = "lex")]
impl Memvid {
    pub fn ask<E>(&mut self, request: AskRequest, embedder: Option<&E>) -> Result<AskResponse>
    where
        E: VecEmbedder + ?Sized,
    {
        if !self.lex_enabled {
            return Err(MemvidError::LexNotEnabled);
        }

        let total_start = Instant::now();
        let lexical_query = sanitize_question_for_lexical(&request.question);
        let primary_tokens: Vec<String> = lexical_query
            .split_whitespace()
            .map(str::to_ascii_lowercase)
            .collect();

        // Detect aggregation questions that need multi-session retrieval
        let is_aggregation = is_aggregation_question(&request.question);
        // Detect recency questions that need the most recent information
        let is_recency = is_recency_question(&request.question);
        // Detect analytical/comparative questions that need comprehensive context
        let is_analytical = is_analytical_question(&request.question);

        let effective_top_k = if is_analytical {
            // Analytical questions require comprehensive context across all time periods
            // to detect changes, reversions, comparisons, etc.
            (request.top_k * 5).max(50)
        } else if is_aggregation {
            // For aggregation questions, retrieve more candidates to ensure diversity
            (request.top_k * 3).max(30)
        } else if is_recency {
            // For recency questions, retrieve more candidates so recency boost can find the newest
            (request.top_k * 2).max(20)
        } else {
            request.top_k
        };

        // For analytical questions, use broad OR query to get comprehensive context
        // These questions require seeing all time periods to compare/analyze
        let search_query = if is_analytical && !primary_tokens.is_empty() {
            let analytical_query = build_analytical_query(&primary_tokens);
            tracing::debug!(
                "analytical question detected, using broad OR query: {}",
                analytical_query
            );
            analytical_query
        } else if is_recency && !primary_tokens.is_empty() {
            // For recency questions, use OR query from the start to maximize recall
            // This ensures we find all relevant documents even with different terminology
            let recency_query = build_recency_query(&primary_tokens);
            tracing::debug!(
                "recency question detected, using OR query: {}",
                recency_query
            );
            recency_query
        } else if lexical_query.is_empty() {
            request.question.clone()
        } else {
            lexical_query.clone()
        };

        let mut search_request = SearchRequest {
            query: search_query,
            top_k: effective_top_k,
            snippet_chars: request.snippet_chars,
            uri: request.uri.clone(),
            scope: request.scope.clone(),
            cursor: request.cursor.clone(),
            #[cfg(feature = "temporal_track")]
            temporal: request.temporal.clone().or_else(|| {
                if request.start.is_some() || request.end.is_some() {
                    Some(TemporalFilter {
                        start_utc: request.start,
                        end_utc: request.end,
                        phrase: None,
                        tz: None,
                    })
                } else {
                    None
                }
            }),
            as_of_frame: request.as_of_frame,
            as_of_ts: request.as_of_ts,
            // Disable sketch pre-filter for ask queries - accuracy is more important than speed
            // SimHash can filter out semantically relevant documents that use different wording
            no_sketch: true,
        };

        // Pre-compute the query embedding once so we can reuse it for vector recall and semantic re-rank
        let mut query_embedding: Option<Vec<f32>> = None;
        if let Some(embedder) = embedder {
            if self.vec_enabled || request.mode != AskMode::Lex {
                query_embedding = Some(embedder.embed_query(&request.question)?);
            }
        }

        tracing::debug!("ask search query: {}", search_request.query);
        let mut retrieval = self.search(search_request.clone())?;
        self.filter_hits_in_time_range(
            &mut retrieval.hits,
            request.start,
            request.end,
            &mut retrieval.total_hits,
        )?;

        let mut lex_fallback_used = false;
        let mut timeline_fallback_used = false;
        if retrieval.hits.is_empty() {
            if !primary_tokens.is_empty() {
                if let Some(or_query) = build_disjunctive_query(&primary_tokens) {
                    if or_query != search_request.query {
                        let mut or_request = search_request.clone();
                        or_request.query = or_query.clone();
                        let mut or_response = self.search(or_request)?;
                        self.filter_hits_in_time_range(
                            &mut or_response.hits,
                            request.start,
                            request.end,
                            &mut or_response.total_hits,
                        )?;
                        if !or_response.hits.is_empty() {
                            retrieval = or_response;
                            search_request.query = or_query;
                            lex_fallback_used = true;
                        }
                    }
                }
            }
            if retrieval.hits.is_empty() {
                if let Some(fallback_query) = lexical_fallback_query(&request.question) {
                    if fallback_query != search_request.query {
                        let mut fallback_request = search_request.clone();
                        fallback_request.query = fallback_query.clone();
                        let mut fallback_response = self.search(fallback_request)?;
                        self.filter_hits_in_time_range(
                            &mut fallback_response.hits,
                            request.start,
                            request.end,
                            &mut fallback_response.total_hits,
                        )?;
                        if !fallback_response.hits.is_empty() {
                            retrieval = fallback_response;
                            search_request.query = fallback_query;
                            lex_fallback_used = true;
                        }
                    }
                }
            }
            // Expanded query fallback: try singular/plural variants for better recall
            if retrieval.hits.is_empty() {
                let expanded_queries = build_expanded_queries(&primary_tokens);
                for expanded_query in expanded_queries {
                    if expanded_query != search_request.query {
                        let mut expanded_request = search_request.clone();
                        expanded_request.query = expanded_query.clone();
                        let mut expanded_response = self.search(expanded_request)?;
                        self.filter_hits_in_time_range(
                            &mut expanded_response.hits,
                            request.start,
                            request.end,
                            &mut expanded_response.total_hits,
                        )?;
                        if !expanded_response.hits.is_empty() {
                            retrieval = expanded_response;
                            search_request.query = expanded_query;
                            lex_fallback_used = true;
                            break;
                        }
                    }
                }
            }
            // Timeline fallback: if still no hits, sample from timeline to provide context
            if retrieval.hits.is_empty() {
                tracing::debug!("ask: no search hits, falling back to timeline sampling");
                if let Ok(timeline_response) = self.build_timeline_fallback_response(
                    &request,
                    &search_request,
                    retrieval.elapsed_ms,
                ) {
                    if !timeline_response.hits.is_empty() {
                        retrieval = timeline_response;
                        timeline_fallback_used = true;
                    }
                }
            }
        }

        // Build multiple candidate lists (lexical variants + vector) and fuse with RRF.
        let mut candidate_lists: Vec<Vec<SearchHit>> = Vec::new();
        let mut vector_used = false;

        // For analytical questions, use timeline directly (skip RRF mixing with search)
        // This ensures full-text hits are preserved - search hits are truncated and would
        // replace timeline hits in RRF due to rank comparison
        if is_analytical {
            // Create a modified request with a large top_k to retrieve the full timeline
            // Analytical questions need to see ALL time periods to detect changes/reversions
            let mut analytical_request = request.clone();
            analytical_request.top_k = 100; // Get up to 100 timeline entries for comprehensive context
            if let Ok(timeline_response) = self.build_timeline_fallback_response(
                &analytical_request,
                &search_request,
                retrieval.elapsed_ms,
            ) {
                if !timeline_response.hits.is_empty() {
                    tracing::debug!(
                        "analytical question: using {} timeline documents (full text) as primary context",
                        timeline_response.hits.len()
                    );
                    // For analytical questions, use ONLY timeline hits (with full text)
                    // Skip mixing with search hits which have truncated text
                    candidate_lists.push(timeline_response.hits);
                }
            }
        } else {
            // For non-analytical questions, add search hits to candidate list
            candidate_lists.push(retrieval.hits.clone());
        }

        // OR-expanded lexical query even when base has hits to widen recall.
        // Skip for analytical questions - they use timeline directly with full text
        if !is_analytical {
            if let Some(or_query) = build_disjunctive_query(&primary_tokens) {
                if or_query != search_request.query {
                    let mut or_request = search_request.clone();
                    or_request.query = or_query.clone();
                    let mut or_response = self.search(or_request)?;
                    self.filter_hits_in_time_range(
                        &mut or_response.hits,
                        request.start,
                        request.end,
                        &mut or_response.total_hits,
                    )?;
                    if !or_response.hits.is_empty() {
                        candidate_lists.push(or_response.hits);
                    }
                }
            }

            if self.vec_enabled && query_embedding.is_some() {
                let vec_hits = vector_hits(
                    self,
                    query_embedding.as_deref().unwrap_or(&[]),
                    &request,
                    effective_top_k.max(24).min(64),
                    embedder.as_ref().and_then(|e| e.model_name()),
                )?;
                if !vec_hits.is_empty() {
                    candidate_lists.push(vec_hits);
                    vector_used = true;
                }
            }
        }

        // Search for corrections that might be relevant to the question
        // Corrections have high priority and should override older information
        if !primary_tokens.is_empty() {
            let correction_query = format!(
                "uri:mv2://correction/* AND ({})",
                primary_tokens.join(" OR ")
            );
            let mut correction_request = search_request.clone();
            correction_request.query = correction_query;
            correction_request.top_k = 10; // Limit to 10 corrections
            if let Ok(correction_response) = self.search(correction_request) {
                if !correction_response.hits.is_empty() {
                    tracing::debug!(
                        "found {} potential corrections for question",
                        correction_response.hits.len()
                    );
                    candidate_lists.push(correction_response.hits);
                }
            }
        }

        // Fuse all candidates with RRF and rebuild retrieval.
        if let Some(fused) = fuse_hits_rrf(candidate_lists, effective_top_k.max(24)) {
            retrieval.hits = fused;
            retrieval.total_hits = retrieval.hits.len();
            if vector_used {
                retrieval.engine = SearchEngineKind::Hybrid;
            }
        }

        if lex_fallback_used && !primary_tokens.is_empty() {
            tracing::debug!(
                "lex fallback reordering with {} primary tokens",
                primary_tokens.len()
            );
            reorder_hits_by_token_matches(&mut retrieval.hits, &primary_tokens);
            if let Some(best_idx) = retrieval
                .hits
                .iter()
                .position(|hit| tokens_present_in_hit(hit, &primary_tokens))
            {
                if best_idx != 0 {
                    retrieval.hits.swap(0, best_idx);
                }
            }
            retrieval.context = build_context(&retrieval.hits);
        }

        if is_update_question(&request.question) || is_recency {
            promote_temporal_extremes(
                self,
                &mut retrieval.hits,
                is_update_question(&request.question),
            )?;
        }

        // For aggregation questions, diversify hits to ensure unique sessions are represented
        if is_aggregation && retrieval.hits.len() > request.top_k {
            tracing::debug!(
                "aggregation question detected: diversifying {} hits to {} unique sessions",
                retrieval.hits.len(),
                request.top_k
            );
            diversify_hits_for_aggregation(&mut retrieval.hits, request.top_k);
            retrieval.total_hits = retrieval.hits.len();
        }

        let retrieval_ms = retrieval.elapsed_ms;
        let mut semantic_scores: HashMap<u64, f32> = HashMap::new();
        let semantics_applied = if request.mode == AskMode::Lex {
            false
        } else {
            self.apply_semantic_ranking(
                embedder,
                &request,
                &mut retrieval.hits,
                &mut semantic_scores,
                query_embedding.as_deref(),
            )?
        };

        if semantics_applied && (is_update_question(&request.question) || is_recency) {
            promote_temporal_extremes(
                self,
                &mut retrieval.hits,
                is_update_question(&request.question),
            )?;
        }

        // Apply correction boost AFTER all other reranking - corrections should have final priority
        // This ensures user corrections override all other ranking signals
        promote_corrections(self, &mut retrieval.hits)?;

        retrieval.context = build_context(&retrieval.hits);

        let (answer, citations, synthesis_ms) = if request.context_only {
            (None, Vec::new(), 0)
        } else {
            let synth_start = Instant::now();
            let citations = build_citations(&retrieval.hits, &semantic_scores);
            let answer = synthesize_answer(&request.question, &retrieval.hits, &citations);
            let synth_ms = synth_start.elapsed().as_millis();
            (answer, citations, synth_ms)
        };

        let retriever = determine_retriever(
            request.mode,
            semantics_applied,
            lex_fallback_used,
            timeline_fallback_used,
        );
        let stats = AskStats {
            retrieval_ms,
            synthesis_ms,
            latency_ms: total_start.elapsed().as_millis(),
        };

        let context_fragments: Vec<AskContextFragment> = retrieval
            .hits
            .iter()
            .map(|hit| AskContextFragment {
                rank: hit.rank,
                frame_id: hit.frame_id,
                uri: hit.uri.clone(),
                title: hit.title.clone(),
                score: hit.score,
                matches: hit.matches,
                range: Some(hit.range),
                chunk_range: hit.chunk_range,
                text: hit.chunk_text.clone().unwrap_or_else(|| hit.text.clone()),
                kind: Some(AskContextFragmentKind::Full),
                #[cfg(feature = "temporal_track")]
                temporal: hit
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.temporal.clone()),
            })
            .collect();

        Ok(AskResponse {
            question: request.question,
            mode: request.mode,
            retriever,
            context_only: request.context_only,
            retrieval,
            answer,
            citations,
            context_fragments,
            stats,
        })
    }

    fn filter_hits_in_time_range(
        &mut self,
        hits: &mut Vec<SearchHit>,
        start: Option<i64>,
        end: Option<i64>,
        total_hits: &mut usize,
    ) -> Result<()> {
        if start.is_none() && end.is_none() {
            return Ok(());
        }

        hits.retain(|hit| match self.frame_by_id(hit.frame_id) {
            Ok(frame) => {
                let effective_ts = self
                    .effective_temporal_timestamp(frame.id, frame.timestamp)
                    .unwrap_or(frame.timestamp);
                if let Some(start_ts) = start {
                    if effective_ts < start_ts {
                        return false;
                    }
                }
                if let Some(end_ts) = end {
                    if effective_ts > end_ts {
                        return false;
                    }
                }
                true
            }
            Err(_) => false,
        });
        for (idx, hit) in hits.iter_mut().enumerate() {
            hit.rank = idx + 1;
        }
        *total_hits = hits.len();
        Ok(())
    }

    fn apply_semantic_ranking<E>(
        &mut self,
        embedder: Option<&E>,
        request: &AskRequest,
        hits: &mut Vec<SearchHit>,
        scores_out: &mut HashMap<u64, f32>,
        query_embedding_hint: Option<&[f32]>,
    ) -> Result<bool>
    where
        E: VecEmbedder + ?Sized,
    {
        let Some(embedder) = embedder else {
            return Ok(false);
        };
        if !self.vec_enabled {
            return Ok(false);
        }

        let query_embedding_cow: Cow<'_, [f32]> = if let Some(existing) = query_embedding_hint {
            Cow::Borrowed(existing)
        } else {
            let embedding = embedder.embed_query(&request.question)?;
            if embedding.is_empty() {
                return Ok(false);
            }
            Cow::Owned(embedding)
        };
        if query_embedding_cow.is_empty() {
            return Ok(false);
        }
        let query_embedding = query_embedding_cow.as_ref();
        let expected_dimension = embedder.embedding_dimension();
        let stored_dimension = self
            .toc
            .indexes
            .vec
            .as_ref()
            .map(|manifest| manifest.dimension)
            .filter(|dim| *dim > 0)
            .or_else(|| {
                self.vec_index.as_ref().and_then(|index| {
                    index
                        .entries()
                        .next()
                        .map(|(_, emb)| u32::try_from(emb.len()).unwrap_or(0))
                })
            })
            .unwrap_or(0);
        if stored_dimension > 0
            && u32::try_from(query_embedding.len()).unwrap_or(u32::MAX) != stored_dimension
        {
            return Err(MemvidError::VecDimensionMismatch {
                expected: stored_dimension,
                actual: query_embedding.len(),
            });
        }

        let mut semantic_scores: HashMap<u64, f32> = HashMap::new();
        for hit in hits.iter() {
            if let Some(embedding) = self.frame_embedding(hit.frame_id)? {
                if expected_dimension == 0 || embedding.len() == expected_dimension {
                    let score = cosine_similarity(query_embedding, &embedding);
                    semantic_scores.insert(hit.frame_id, score);
                }
            }
        }
        if semantic_scores.is_empty() {
            return Ok(false);
        }

        reorder_hits_with_semantic_scores(hits, &semantic_scores, request.mode);
        scores_out.extend(semantic_scores);
        Ok(true)
    }

    /// Build a fallback `SearchResponse` from timeline entries when search returns no hits.
    /// This gives the LLM some context to work with for general questions about the document.
    /// For comprehensive coverage, includes child frames (e.g., document pages) as well.
    fn build_timeline_fallback_response(
        &mut self,
        request: &AskRequest,
        search_request: &SearchRequest,
        elapsed_ms: u128,
    ) -> Result<SearchResponse> {
        // Get timeline entries (up to top_k frames)
        let limit = NonZeroU64::new(request.top_k as u64).unwrap_or(NonZeroU64::new(8).unwrap());
        let timeline_query = TimelineQueryBuilder::default().limit(limit).build();
        let entries = self.timeline(timeline_query)?;

        if entries.is_empty() {
            return Ok(SearchResponse {
                query: search_request.query.clone(),
                hits: Vec::new(),
                total_hits: 0,
                context: String::new(),
                next_cursor: None,
                engine: SearchEngineKind::LexFallback,
                elapsed_ms,
                params: SearchParams {
                    top_k: request.top_k,
                    snippet_chars: request.snippet_chars,
                    cursor: search_request.cursor.clone(),
                },
            });
        }

        // Collect all frame IDs including child frames for comprehensive coverage
        // This is critical for analytical questions that need full document context
        let mut all_frame_ids: Vec<(u64, Option<String>)> = Vec::new();
        for entry in &entries {
            // Add parent frame
            all_frame_ids.push((entry.frame_id, entry.uri.clone()));
            // Add all child frames (e.g., document pages)
            for child_id in &entry.child_frames {
                all_frame_ids.push((*child_id, None));
            }
        }

        tracing::debug!(
            "timeline fallback: expanding {} parent entries to {} total frames (including children)",
            entries.len(),
            all_frame_ids.len()
        );

        // Convert all frames to SearchHits
        let mut hits = Vec::with_capacity(all_frame_ids.len());
        let mut context_parts = Vec::new();

        for (rank, (frame_id, parent_uri)) in all_frame_ids.iter().enumerate() {
            // Get full frame content for the context
            let (frame_text, frame_uri) = match self.frame_by_id(*frame_id) {
                Ok(frame) => {
                    let content = self.frame_content(&frame).unwrap_or_else(|_| String::new());
                    let uri = frame
                        .uri
                        .clone()
                        .or_else(|| parent_uri.clone())
                        .unwrap_or_else(|| format!("mv2://frame/{frame_id}"));
                    (content, uri)
                }
                Err(_) => continue, // Skip frames we can't read
            };

            if frame_text.is_empty() {
                continue;
            }

            // For timeline fallback, keep the FULL text in BOTH text and chunk_text
            // This is critical for analytical questions that need complete context
            // build_context uses hit.text, so we must put full content there

            // Build context from full frame text
            context_parts.push(format!("[{}] {}", rank + 1, frame_text));

            hits.push(SearchHit {
                rank: rank + 1,
                score: None,
                frame_id: *frame_id,
                uri: frame_uri,
                title: None,
                matches: 0, // No keyword matches for timeline fallback
                range: (0, frame_text.len()),
                chunk_range: Some((0, frame_text.len())),
                // Full text in both fields for complete context
                text: frame_text.clone(),
                chunk_text: Some(frame_text.clone()),
                metadata: None,
            });
        }

        let context = context_parts.join("\n\n");
        let total_hits = hits.len();

        Ok(SearchResponse {
            query: search_request.query.clone(),
            hits,
            total_hits,
            context,
            next_cursor: None,
            engine: SearchEngineKind::LexFallback, // Mark as fallback
            elapsed_ms,
            params: SearchParams {
                top_k: request.top_k,
                snippet_chars: request.snippet_chars,
                cursor: search_request.cursor.clone(),
            },
        })
    }
}

#[cfg(not(feature = "lex"))]
impl Memvid {
    pub fn ask<E>(&mut self, _request: AskRequest, _embedder: Option<&E>) -> Result<AskResponse>
    where
        E: VecEmbedder + ?Sized,
    {
        Err(MemvidError::LexNotEnabled)
    }
}

fn determine_retriever(
    mode: AskMode,
    semantics_applied: bool,
    lex_fallback_used: bool,
    timeline_fallback_used: bool,
) -> AskRetriever {
    // Timeline fallback takes precedence if used
    if timeline_fallback_used {
        return AskRetriever::TimelineFallback;
    }

    match mode {
        AskMode::Lex => AskRetriever::Lex,
        AskMode::Sem => {
            if semantics_applied {
                AskRetriever::Semantic
            } else if lex_fallback_used {
                AskRetriever::LexFallback
            } else {
                AskRetriever::LexFallback
            }
        }
        AskMode::Hybrid => {
            if semantics_applied {
                AskRetriever::Hybrid
            } else if lex_fallback_used {
                AskRetriever::LexFallback
            } else {
                AskRetriever::Lex
            }
        }
    }
}

fn reorder_hits_with_semantic_scores(
    hits: &mut Vec<SearchHit>,
    semantic_scores: &HashMap<u64, f32>,
    mode: AskMode,
) {
    let mut semantic_rank: HashMap<u64, usize> = HashMap::new();
    let mut sorted_semantic: Vec<(u64, f32)> = semantic_scores
        .iter()
        .map(|(frame_id, score)| (*frame_id, *score))
        .collect();
    sorted_semantic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (idx, (frame_id, _)) in sorted_semantic.iter().enumerate() {
        semantic_rank.insert(*frame_id, idx + 1);
    }

    let mut ordering: Vec<(usize, f32, usize)> = hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| {
            let lexical_rank = hit.rank;
            let semantic_score = semantic_scores.get(&hit.frame_id).copied().unwrap_or(0.0);
            let combined = match mode {
                AskMode::Sem => semantic_score,
                AskMode::Hybrid => {
                    let lexical_rrf = 1.0 / (RRF_K + lexical_rank as f32);
                    let semantic_rrf = semantic_rank
                        .get(&hit.frame_id)
                        .map_or(0.0, |rank| 1.0 / (RRF_K + *rank as f32));
                    semantic_score + lexical_rrf + semantic_rrf
                }
                AskMode::Lex => 1.0 / (RRF_K + lexical_rank as f32),
            };
            (idx, combined, lexical_rank)
        })
        .collect();

    ordering.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });

    let mut reordered = Vec::with_capacity(hits.len());
    for (new_rank, (idx, _, _)) in ordering.into_iter().enumerate() {
        let mut hit = hits[idx].clone();
        hit.rank = new_rank + 1;
        if let Some(score) = semantic_scores.get(&hit.frame_id) {
            hit.score = Some(*score);
        }
        reordered.push(hit);
    }
    *hits = reordered;
}

fn build_citations(hits: &[SearchHit], semantic_scores: &HashMap<u64, f32>) -> Vec<AskCitation> {
    hits.iter()
        .enumerate()
        .map(|(idx, hit)| AskCitation {
            index: idx + 1,
            frame_id: hit.frame_id,
            uri: hit.uri.clone(),
            chunk_range: hit.chunk_range.or(Some(hit.range)),
            score: semantic_scores.get(&hit.frame_id).copied().or(hit.score),
        })
        .collect()
}

fn synthesize_answer(
    question: &str,
    hits: &[SearchHit],
    citations: &[AskCitation],
) -> Option<String> {
    if hits.is_empty() {
        return None;
    }

    let mut segments = Vec::new();
    for citation in citations.iter().take(3) {
        if let Some(hit) = hits
            .iter()
            .find(|candidate| candidate.frame_id == citation.frame_id)
        {
            let snippet = hit.text.trim();
            if snippet.is_empty() {
                continue;
            }
            let sanitized = snippet.split_whitespace().collect::<Vec<_>>().join(" ");
            if sanitized.is_empty() {
                continue;
            }
            segments.push(format!("{} [{}]", sanitized, citation.index));
        }
    }

    if segments.is_empty() {
        return Some(format!(
            "No direct synthesis available for '{question}'. Review the top contexts manually.",
        ));
    }

    Some(segments.join(" "))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        sum_a += x * x;
        sum_b += y * y;
    }

    if sum_a <= f32::EPSILON || sum_b <= f32::EPSILON {
        0.0
    } else {
        dot / (sum_a.sqrt() * sum_b.sqrt())
    }
}

fn lexical_fallback_query(question: &str) -> Option<String> {
    let sanitized_full = sanitize_question_for_lexical(question);
    if sanitized_full.is_empty() {
        return None;
    }

    let sanitized_tokens: Vec<String> = sanitized_full
        .split_whitespace()
        .map(std::string::ToString::to_string)
        .collect();

    let mut candidates: Vec<String> = question
        .split_whitespace()
        .filter_map(|raw| {
            let candidate = sanitize_question_for_lexical(raw);
            if candidate.is_empty() {
                return None;
            }
            if raw.chars().any(|c| c.is_ascii_uppercase()) {
                let lower = candidate.to_ascii_lowercase();
                if !is_stopword(&lower) {
                    return Some(candidate);
                }
            }
            None
        })
        .collect();

    if candidates.is_empty() {
        for token in &sanitized_tokens {
            let lower = token.to_ascii_lowercase();
            if token.len() > 3 && !is_stopword(&lower) {
                candidates.push(token.clone());
            }
        }
    }

    if candidates.is_empty() {
        candidates.extend(sanitized_tokens);
    }

    candidates
        .into_iter()
        .map(|candidate| candidate.trim().to_string())
        .find(|candidate| !candidate.is_empty())
}

fn is_stopword(token: &str) -> bool {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "does", "do",
        "did", "else", "for", "from", "had", "have", "has", "he", "her", "here", "hers", "him",
        "his", "how", "i", "if", "in", "is", "it", "its", "it's", "many", "me", "mine", "more",
        "most", "much", "my", "no", "not", "of", "on", "or", "our", "ours", "she", "so", "that",
        "the", "their", "them", "there", "these", "they", "this", "those", "through", "to", "us",
        "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "with", "you",
        "your", "yours",
    ];
    STOPWORDS.contains(&token)
}

fn sanitize_question_for_lexical(question: &str) -> String {
    let mut sanitized = String::with_capacity(question.len());
    let mut prev_was_space = false;

    for ch in question.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, ':' | '/' | '_' | '-' | '.' | '@') {
            sanitized.push(ch);
            prev_was_space = false;
        } else if ch.is_whitespace() {
            if !prev_was_space && !sanitized.is_empty() {
                sanitized.push(' ');
                prev_was_space = true;
            }
        } else if !sanitized.is_empty() && !prev_was_space {
            sanitized.push(' ');
            prev_was_space = true;
        }
    }

    let trimmed = sanitized.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.is_empty() {
        return String::new();
    }

    let mut filtered: Vec<&str> = Vec::new();
    for token in tokens.iter().copied() {
        if token.contains(':') {
            filtered.push(token);
            continue;
        }
        let lower = token.to_ascii_lowercase();
        if !is_stopword(&lower) {
            filtered.push(token);
        }
    }

    let final_tokens = if filtered.is_empty() {
        tokens
    } else {
        filtered
    };
    final_tokens.join(" ")
}

fn build_disjunctive_query(tokens: &[String]) -> Option<String> {
    let mut unique: BTreeSet<String> = BTreeSet::new();
    for token in tokens {
        if token.trim().is_empty() {
            continue;
        }
        unique.insert(token.to_ascii_lowercase());
    }
    if unique.is_empty() {
        None
    } else {
        Some(unique.into_iter().collect::<Vec<_>>().join(" OR "))
    }
}

/// Build expanded query variants for better recall on aggregation questions.
/// For example, "weddings attended" expands to include possessive forms like
/// "cousin's wedding", "friend's wedding", etc.
fn build_expanded_queries(tokens: &[String]) -> Vec<String> {
    let mut variants = Vec::new();

    // Find key nouns that might have possessive relationships
    let key_nouns: Vec<&str> = tokens
        .iter()
        .filter(|t| !is_stopword(t) && t.len() > 3)
        .map(std::string::String::as_str)
        .collect();

    if key_nouns.is_empty() {
        return variants;
    }

    // For each key noun, try singular/plural and possessive forms
    for noun in &key_nouns {
        // Try the base form
        variants.push((*noun).to_string());

        // Try singular/plural variants
        if noun.ends_with('s') && noun.len() > 4 {
            // "weddings" -> "wedding"
            let singular = &noun[..noun.len() - 1];
            variants.push(singular.to_string());
        } else if !noun.ends_with('s') {
            // "wedding" -> "weddings"
            variants.push(format!("{noun}s"));
        }
    }

    // Create OR queries from the variants
    if variants.is_empty() {
        Vec::new()
    } else {
        let or_query = variants.join(" OR ");
        vec![or_query]
    }
}

fn tokens_present_in_hit(hit: &SearchHit, tokens: &[String]) -> bool {
    if tokens.is_empty() {
        return false;
    }
    let haystack = hit
        .chunk_text
        .as_ref()
        .unwrap_or(&hit.text)
        .to_ascii_lowercase();
    tokens.iter().all(|token| haystack.contains(token.as_str()))
}

/// Detect questions that imply a before/after update (e.g., "initially", "now", "before", "currently").
fn is_update_question(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();

    let change_markers = [
        "before",
        "initial",
        "initially",
        "originally",
        "used to",
        "earlier",
        "previous",
        "first",
        "when i started",
        "start",
    ];
    let now_markers = [
        "now",
        "currently",
        "these days",
        "as of",
        "latest",
        "today",
        "currently",
        "present",
    ];

    let has_change = change_markers.iter().any(|marker| lower.contains(marker));
    let has_now = now_markers.iter().any(|marker| lower.contains(marker));

    has_change && has_now
        || lower.contains("update")
        || lower.contains("changed")
        || lower.contains("still")
}

/// Detect if a question requires aggregation across multiple sessions.
/// These are questions like "how many X have I done", "list all X", "what is the total".
fn is_aggregation_question(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();

    // Counting patterns
    let counting_patterns = [
        "how many",
        "how much",
        "what is the total",
        "what's the total",
        "count of",
        "number of",
        "total number",
    ];

    // Listing patterns
    let listing_patterns = [
        "list all",
        "list the",
        "what are all",
        "what were all",
        "name all",
        "tell me all",
        "all the times",
        "every time",
    ];

    // Aggregation verbs with "have I" or "did I" patterns
    let aggregation_verbs = [
        "have i attended",
        "have i been to",
        "have i visited",
        "have i done",
        "have i completed",
        "have i watched",
        "have i read",
        "did i attend",
        "did i go to",
        "did i visit",
    ];

    for pattern in counting_patterns
        .iter()
        .chain(listing_patterns.iter())
        .chain(aggregation_verbs.iter())
    {
        if lower.contains(pattern) {
            return true;
        }
    }

    false
}

/// Detect if a question is asking for the most recent/current information.
/// These questions need recency-weighted search to find the latest updates.
/// Examples: "What is my current X?", "What's my latest Y?", "What is my X now?"
fn is_recency_question(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();

    // Multi-word patterns (safe to use contains)
    let multi_word_patterns = [
        "most recent",
        "right now",
        "these days",
        "at the moment",
        "up to date",
    ];

    for pattern in &multi_word_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }

    // Single-word patterns need word boundary checking to avoid false matches
    // e.g., "now" should not match "know"
    let single_word_patterns = [
        "current",
        "currently",
        "latest",
        "nowadays",
        "presently",
        "today",
    ];

    // Split into words and check for exact matches
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).collect();
    for pattern in &single_word_patterns {
        if words.contains(pattern) {
            return true;
        }
    }

    // Special case: "now" at end of sentence (common in "What is X now?")
    if words.last() == Some(&"now") || lower.ends_with(" now?") || lower.ends_with(" now") {
        return true;
    }

    false
}

/// Build a broad OR query for analytical questions.
/// For questions that require comparing states across time, we use a very permissive
/// query to retrieve as many relevant documents as possible.
fn build_analytical_query(tokens: &[String]) -> String {
    // Remove abstract analytical words that won't match document content
    let analytical_stopwords: HashSet<&str> = [
        "any",
        "are",
        "there",
        "that",
        "reverted",
        "revert",
        "previous",
        "value",
        "values",
        "changed",
        "change",
        "changes",
        "compare",
        "comparison",
        "different",
        "difference",
        "between",
        "vs",
        "versus",
        "if",
        "so",
        "which",
        "what",
        "did",
        "does",
        "how",
        "when",
        "over",
        "time",
        "throughout",
        "evolution",
        "history",
        "timeline",
        "progression",
        "back",
        "went",
        "go",
        "returned",
    ]
    .into_iter()
    .collect();

    // Keep only content-bearing terms
    let content_terms: Vec<&str> = tokens
        .iter()
        .map(std::string::String::as_str)
        .filter(|t| !analytical_stopwords.contains(*t) && t.len() > 2)
        .collect();

    if content_terms.is_empty() {
        // If no content terms, use a wildcard-style query
        // Return empty to trigger timeline fallback
        String::new()
    } else {
        // Create OR query with all content terms
        content_terms.join(" OR ")
    }
}

/// Detect if a question requires analytical/comparative reasoning across time periods.
/// These questions need comprehensive context to compare states, find reversions, or
/// track changes over time. Examples:
/// - "Are there any attributes that reverted?"
/// - "What changed between X and Y?"
/// - "Compare the state in 2024 vs 2025"
/// - "Did anything go back to a previous value?"
fn is_analytical_question(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();

    // Patterns indicating comparative/analytical reasoning
    let analytical_patterns = [
        "reverted",
        "revert",
        "went back",
        "go back",
        "changed back",
        "returned to",
        "compare",
        "comparison",
        "difference between",
        "changed over time",
        "over time",
        "across all",
        "throughout",
        "evolution of",
        "history of",
        "timeline of",
        "progression of",
        "changed from",
        "differ from",
        "vs ",
        "versus",
        "before and after",
        "any changes",
        "any attributes",
        "any differences",
    ];

    for pattern in &analytical_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }

    false
}

/// Build an OR query for recency questions to maximize recall.
/// This helps find all relevant documents so recency boosting can pick the newest.
fn build_recency_query(tokens: &[String]) -> String {
    // Filter out temporal modifier words that don't help with content matching
    let temporal_modifiers: HashSet<&str> = [
        "current",
        "currently",
        "latest",
        "recent",
        "recently",
        "now",
        "today",
        "presently",
        "moment",
        "nowadays",
    ]
    .into_iter()
    .collect();

    let content_tokens: Vec<&String> = tokens
        .iter()
        .filter(|t| !temporal_modifiers.contains(t.as_str()))
        .collect();

    if content_tokens.is_empty() {
        return tokens.join(" OR ");
    }

    // Build OR query for better recall
    content_tokens
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Diversify hits for aggregation questions by ensuring unique URIs/sessions.
/// This helps find information scattered across multiple conversations.
fn diversify_hits_for_aggregation(hits: &mut Vec<SearchHit>, target_unique: usize) {
    if hits.len() <= target_unique {
        return;
    }

    let mut seen_uris: HashSet<String> = HashSet::new();
    let mut diversified: Vec<SearchHit> = Vec::new();
    let mut remaining: Vec<SearchHit> = Vec::new();

    // First pass: collect one hit per unique URI (base URI without fragment)
    for hit in hits.drain(..) {
        let base_uri = hit.uri.split('#').next().unwrap_or(&hit.uri).to_string();
        if !seen_uris.contains(&base_uri) && diversified.len() < target_unique {
            seen_uris.insert(base_uri);
            diversified.push(hit);
        } else {
            remaining.push(hit);
        }
    }

    // Second pass: fill remaining slots with best scoring remaining hits
    let slots_left = target_unique.saturating_sub(diversified.len());
    for hit in remaining.into_iter().take(slots_left) {
        diversified.push(hit);
    }

    // Re-rank
    for (idx, hit) in diversified.iter_mut().enumerate() {
        hit.rank = idx + 1;
    }

    *hits = diversified;
}

/// Retrieve pure vector hits for fusion.
fn vector_hits(
    memvid: &mut Memvid,
    query_embedding: &[f32],
    request: &AskRequest,
    limit: usize,
    model_name: Option<&str>,
) -> Result<Vec<SearchHit>> {
    if !memvid.vec_enabled || query_embedding.is_empty() {
        return Ok(Vec::new());
    }

    // Use adaptive retrieval if configured
    if let Some(ref adaptive_config) = request.adaptive {
        if adaptive_config.enabled {
            let result = memvid.search_adaptive_verified(
                &request.question,
                query_embedding,
                adaptive_config.clone(),
                request.snippet_chars,
                request.scope.as_deref(),
                model_name,
            )?;
            tracing::debug!(
                "adaptive retrieval: {} -> {} results ({})",
                result.stats.total_considered,
                result.stats.returned,
                result.stats.triggered_by
            );
            return Ok(result.results);
        }
    }

    let vec_response = memvid.vec_search_with_embedding_verified(
        &request.question,
        query_embedding,
        limit,
        request.snippet_chars,
        request.scope.as_deref(),
        model_name,
    )?;

    Ok(vec_response.hits)
}

/// Fuse multiple hit lists using Reciprocal Rank Fusion.
fn fuse_hits_rrf(mut lists: Vec<Vec<SearchHit>>, target: usize) -> Option<Vec<SearchHit>> {
    if lists.is_empty() {
        return None;
    }
    lists.retain(|list| !list.is_empty());
    if lists.is_empty() {
        return None;
    }

    let mut fused: HashMap<u64, (f32, SearchHit)> = HashMap::new();

    for list in &lists {
        for (idx, hit) in list.iter().enumerate() {
            let rank = idx + 1;
            let contribution = 1.0 / (RRF_K + rank as f32);
            let entry = fused
                .entry(hit.frame_id)
                .or_insert_with(|| (0.0, hit.clone()));

            // Keep the hit with more matches or earlier rank as the representative.
            if hit.matches > entry.1.matches
                || (hit.matches == entry.1.matches && rank < entry.1.rank)
            {
                entry.1 = hit.clone();
            }
            entry.0 += contribution;
        }
    }

    let mut combined: Vec<(u64, f32, SearchHit)> = fused
        .into_iter()
        .map(|(id, (score, hit))| (id, score, hit))
        .collect();

    combined.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.rank.cmp(&b.2.rank))
    });

    let mut result = Vec::new();
    for (_id, score, mut hit) in combined.into_iter().take(target.max(1)) {
        hit.score = Some(score);
        result.push(hit);
    }

    for (idx, hit) in result.iter_mut().enumerate() {
        hit.rank = idx + 1;
    }

    Some(result)
}

/// Promote corrections to the top of the hit list.
/// Corrections are user-provided facts that should override older information.
/// The MOST RECENT correction takes priority (sorted by timestamp, newest first).
fn promote_corrections(memvid: &mut Memvid, hits: &mut Vec<SearchHit>) -> Result<()> {
    if hits.is_empty() {
        return Ok(());
    }

    // Find corrections with their timestamps and boost factors
    // (idx, timestamp, boost)
    let mut corrections: Vec<(usize, i64, f32)> = Vec::new();
    for (idx, hit) in hits.iter().enumerate() {
        // Check if this is a correction by looking at the URI
        if hit.uri.contains("mv2://correction/") {
            if let Ok(frame) = memvid.frame_by_id(hit.frame_id) {
                let boost = frame
                    .extra_metadata
                    .get("memvid.correction.boost")
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(2.0);
                corrections.push((idx, frame.timestamp, boost));
            }
        }
    }

    if corrections.is_empty() {
        return Ok(());
    }

    // Sort corrections by timestamp DESC (newest first), then by boost DESC
    corrections.sort_by(|a, b| {
        b.1.cmp(&a.1) // timestamp descending (newest first)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)) // boost descending
    });

    tracing::debug!(
        "promoting {} corrections to top of hit list (newest first)",
        corrections.len()
    );

    // Reorder: corrections first (newest first), then other hits
    let mut reordered: Vec<SearchHit> = Vec::with_capacity(hits.len());
    let mut seen: HashSet<usize> = HashSet::new();

    // Add corrections first (newest first)
    for (idx, _ts, _boost) in &corrections {
        reordered.push(hits[*idx].clone());
        seen.insert(*idx);
    }

    // Add remaining hits
    for (idx, hit) in hits.iter().enumerate() {
        if !seen.contains(&idx) {
            reordered.push(hit.clone());
        }
    }

    *hits = reordered;
    Ok(())
}

/// Promote earliest/latest hits into the visible context so update/recency questions see both ends.
/// Uses `content_dates` for temporal ordering (dates extracted from document content),
/// falling back to frame.timestamp (ingestion time) if no content dates are available.
#[cfg(feature = "lex")]
fn promote_temporal_extremes(
    memvid: &mut Memvid,
    hits: &mut Vec<SearchHit>,
    include_earliest: bool,
) -> Result<()> {
    use crate::memvid::search::parse_content_date_to_timestamp;

    if hits.len() < 2 {
        return Ok(());
    }

    let mut with_ts: Vec<(usize, i64, u64)> = Vec::new();
    for (idx, hit) in hits.iter().enumerate() {
        if let Ok(frame) = memvid.frame_by_id(hit.frame_id) {
            // Prefer content_dates (dates from document content) over frame.timestamp (ingestion time)
            // This ensures documents about recent events rank higher even if ingested at the same time
            let effective_ts =
                parse_content_date_to_timestamp(&frame.content_dates).unwrap_or(frame.timestamp);
            with_ts.push((idx, effective_ts, hit.frame_id));
        }
    }

    if with_ts.len() < 2 {
        return Ok(());
    }

    with_ts.sort_by_key(|(_, ts, _)| *ts);
    let earliest_id = with_ts.first().map(|(_, _, id)| *id);
    let latest_id = with_ts.last().map(|(_, _, id)| *id);

    let mut priority: Vec<u64> = Vec::new();
    if include_earliest {
        if let Some(id) = earliest_id {
            priority.push(id);
        }
    }
    if let Some(id) = latest_id {
        if !priority.contains(&id) {
            priority.push(id);
        }
    }

    if priority.is_empty() {
        return Ok(());
    }

    let mut reordered: Vec<SearchHit> = Vec::with_capacity(hits.len());
    let mut seen: HashSet<u64> = HashSet::new();

    for id in priority {
        if let Some(pos) = hits.iter().position(|hit| hit.frame_id == id) {
            if seen.insert(id) {
                reordered.push(hits[pos].clone());
            }
        }
    }

    for hit in hits.iter() {
        if seen.insert(hit.frame_id) {
            reordered.push(hit.clone());
        }
    }

    for (idx, hit) in reordered.iter_mut().enumerate() {
        hit.rank = idx + 1;
    }

    *hits = reordered;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{build_disjunctive_query, lexical_fallback_query, sanitize_question_for_lexical};

    #[test]
    fn sanitize_question_strips_trailing_punctuation() {
        let sanitized = sanitize_question_for_lexical("Safari appears?");
        assert_eq!(sanitized, "Safari appears");
    }

    #[test]
    fn sanitize_preserves_field_queries() {
        let sanitized = sanitize_question_for_lexical("tag:security Safari updates!");
        assert_eq!(sanitized, "tag:security Safari updates");
    }

    #[test]
    fn sanitize_removes_stopwords_when_possible() {
        let sanitized = sanitize_question_for_lexical("How much is the Header Checksum?");
        assert_eq!(sanitized, "Header Checksum");
    }

    #[test]
    fn fallback_prefers_proper_noun() {
        let fallback = lexical_fallback_query("How many times does Safari appears?");
        assert_eq!(fallback.as_deref(), Some("Safari"));
    }

    #[test]
    fn fallback_skips_stopwords() {
        let fallback = lexical_fallback_query("what is the index size");
        assert_eq!(fallback.as_deref(), Some("index"));
    }

    #[test]
    fn disjunctive_query_deduplicates_tokens() {
        let tokens = vec![
            "header".to_string(),
            "checksum".to_string(),
            "header".to_string(),
        ];
        let query = build_disjunctive_query(&tokens).expect("query");
        assert_eq!(query, "checksum OR header");
    }
}
