// Safe unwrap: guaranteed non-empty vector operations.
#![allow(clippy::unwrap_used)]
use crate::MemvidError;
use crate::Result;
use crate::memvid::lifecycle::Memvid;
#[cfg(not(feature = "temporal_track"))]
#[allow(unused_imports)]
use crate::types::FrameId;
#[cfg(feature = "temporal_track")]
use crate::types::{
    FrameId, SearchHitTemporal, SearchHitTemporalAnchor, SearchHitTemporalMention, TemporalMention,
};
use crate::types::{SearchEngineKind, SearchHit, SearchHitMetadata, SearchParams, SearchResponse};
#[cfg(feature = "temporal_track")]
use std::collections::HashMap;
#[cfg(feature = "temporal_track")]
use std::collections::HashSet;
use std::collections::{BTreeMap, HashSet as StdHashSet};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

pub(super) fn empty_search_response(
    query: String,
    params: SearchParams,
    elapsed_ms: u128,
    engine: SearchEngineKind,
) -> SearchResponse {
    SearchResponse {
        query,
        elapsed_ms,
        total_hits: 0,
        params,
        hits: Vec::new(),
        context: String::new(),
        next_cursor: None,
        engine,
    }
}

pub(super) fn timestamp_to_rfc3339(timestamp: i64) -> Option<String> {
    OffsetDateTime::from_unix_timestamp(timestamp)
        .ok()
        .map(|dt| {
            dt.format(&Rfc3339)
                .unwrap_or_else(|_| timestamp.to_string())
        })
}

pub(super) fn parse_cursor(cursor: Option<&str>, total_hits: usize) -> Result<usize> {
    let Some(token) = cursor else {
        return Ok(0);
    };
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    let value = trimmed
        .parse::<usize>()
        .map_err(|_| MemvidError::InvalidCursor {
            reason: "cursor not an integer",
        })?;
    if value > total_hits {
        return Err(MemvidError::InvalidCursor {
            reason: "cursor beyond total hits",
        });
    }
    Ok(value)
}

/// Build context for LLM from search hits using a multi-document strategy.
///
/// Key design decisions for deterministic, comprehensive context:
/// 1. Uses BTreeMap for deterministic iteration order (sorted by URI)
/// 2. Includes top hits from MULTIPLE documents for diverse context
/// 3. Prioritizes by rank while ensuring document diversity
/// 4. Maximum 24 hits for balanced context (not too much noise, not too little coverage)
pub(crate) fn build_context(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return String::new();
    }

    // Maximum hits to include in context
    // Balanced at 24 to provide good coverage without overwhelming the LLM with noise
    const MAX_CONTEXT_HITS: usize = 24;

    // Group hits by base URI using BTreeMap for deterministic iteration
    let mut groups: BTreeMap<String, GroupSummary> = BTreeMap::new();
    for (idx, hit) in hits.iter().enumerate() {
        let base = hit
            .uri
            .split('#')
            .next()
            .unwrap_or(&hit.uri)
            .to_ascii_lowercase();
        let entry = groups.entry(base).or_insert_with(GroupSummary::default);
        entry.indices.push(idx);
        entry.total_matches += hit.matches.max(1);
        entry.best_rank = entry.best_rank.min(hit.rank);
    }

    // Multi-document strategy: select diverse hits from different URIs
    // First pass: take best hit from each unique URI for diversity
    let mut selected_indices: Vec<usize> = Vec::with_capacity(MAX_CONTEXT_HITS);
    let mut seen_uris: StdHashSet<String> = StdHashSet::new();

    // Collect groups sorted by best_rank (lower is better)
    let mut sorted_groups: Vec<(String, GroupSummary)> = groups.into_iter().collect();
    sorted_groups.sort_by(|a, b| {
        a.1.best_rank
            .cmp(&b.1.best_rank)
            .then(b.1.total_matches.cmp(&a.1.total_matches))
    });

    // First pass: one hit per unique document (for diversity)
    for (uri, group) in &sorted_groups {
        if selected_indices.len() >= MAX_CONTEXT_HITS {
            break;
        }
        if !seen_uris.contains(uri) {
            // Take the best-ranked hit from this group (first index after sorting)
            if let Some(&best_idx) = group.indices.first() {
                selected_indices.push(best_idx);
                seen_uris.insert(uri.clone());
            }
        }
    }

    // Second pass: fill remaining slots with additional hits by rank order
    if selected_indices.len() < MAX_CONTEXT_HITS {
        // Collect all remaining hits not yet selected, sorted by rank
        let mut remaining: Vec<(usize, usize)> = hits
            .iter()
            .enumerate()
            .filter(|(idx, _)| !selected_indices.contains(idx))
            .map(|(idx, hit)| (idx, hit.rank))
            .collect();
        remaining.sort_by_key(|(_, rank)| *rank);

        for (idx, _) in remaining {
            if selected_indices.len() >= MAX_CONTEXT_HITS {
                break;
            }
            selected_indices.push(idx);
        }
    }

    // Sort by original index for stable output order
    selected_indices.sort_unstable();

    // Render selected hits
    selected_indices
        .into_iter()
        .filter_map(|idx| hits.get(idx))
        .map(render_hit)
        .collect::<Vec<_>>()
        .join("\n\n")
}

struct GroupSummary {
    indices: Vec<usize>,
    total_matches: usize,
    best_rank: usize,
}

impl Default for GroupSummary {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            total_matches: 0,
            best_rank: usize::MAX,
        }
    }
}

fn render_hit(hit: &SearchHit) -> String {
    let display_uri = hit.uri.strip_prefix("mv2://").unwrap_or(&hit.uri);
    let heading = hit.title.as_deref().unwrap_or(display_uri);
    format!(
        "### [{}] {} — {}\n{}\n(matches: {})",
        hit.rank, display_uri, heading, hit.text, hit.matches
    )
}

pub(super) fn collect_token_occurrences(
    content_lower: &str,
    tokens: &[String],
) -> Vec<(usize, usize)> {
    let mut occurrences = Vec::new();
    for token in tokens {
        let needle = token.trim();
        if needle.is_empty() {
            continue;
        }
        let mut start = 0usize;
        while let Some(pos) = content_lower[start..].find(needle) {
            let absolute = start + pos;
            let end = absolute + needle.len();
            occurrences.push((absolute, end));
            start = end;
        }
    }
    occurrences.sort_unstable();
    occurrences.dedup();
    occurrences
}

pub(crate) fn reorder_hits_by_token_matches(hits: &mut Vec<SearchHit>, tokens: &[String]) {
    if hits.is_empty() || tokens.is_empty() {
        return;
    }

    hits.sort_by(|a, b| {
        let metrics_a = token_match_metrics(a, tokens);
        let metrics_b = token_match_metrics(b, tokens);
        tracing::debug!(
            "reorder metrics for hit {}: unique={} total={} span={}",
            a.frame_id,
            metrics_a.unique_tokens,
            metrics_a.total_occurrences,
            metrics_a.tightest_span
        );
        tracing::debug!(
            "reorder metrics for hit {}: unique={} total={} span={}",
            b.frame_id,
            metrics_b.unique_tokens,
            metrics_b.total_occurrences,
            metrics_b.tightest_span
        );
        metrics_b
            .unique_tokens
            .cmp(&metrics_a.unique_tokens)
            .then(
                metrics_b
                    .total_occurrences
                    .cmp(&metrics_a.total_occurrences),
            )
            .then(metrics_a.tightest_span.cmp(&metrics_b.tightest_span))
            .then(a.rank.cmp(&b.rank))
    });

    for (idx, hit) in hits.iter_mut().enumerate() {
        hit.rank = idx + 1;
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
struct TokenMetrics {
    unique_tokens: usize,
    total_occurrences: usize,
    tightest_span: usize,
}

fn token_match_metrics(hit: &SearchHit, tokens: &[String]) -> TokenMetrics {
    let haystack = hit
        .chunk_text
        .as_ref()
        .unwrap_or(&hit.text)
        .to_ascii_lowercase();

    let mut unique = 0usize;
    let mut total = 0usize;
    let mut positions: Vec<usize> = Vec::new();
    for token in tokens {
        let mut search_start = 0usize;
        let mut found = false;
        while let Some(pos) = haystack[search_start..].find(token) {
            let absolute = search_start + pos;
            positions.push(absolute);
            total += 1;
            found = true;
            search_start = absolute + token.len();
        }
        if found {
            unique += 1;
        }
    }

    positions.sort_unstable();
    let span = if positions.len() >= 2 {
        positions.last().copied().unwrap() - positions[0]
    } else {
        usize::MAX
    };

    TokenMetrics {
        unique_tokens: unique,
        total_occurrences: total,
        tightest_span: span,
    }
}

#[cfg(feature = "temporal_track")]
pub(crate) fn attach_temporal_metadata(memvid: &mut Memvid, hits: &mut [SearchHit]) -> Result<()> {
    if hits.is_empty() {
        return Ok(());
    }

    let Some(track) = memvid.temporal_track_ref()?.cloned() else {
        return Ok(());
    };

    let frame_ids: HashSet<FrameId> = hits.iter().map(|hit| hit.frame_id).collect();
    if frame_ids.is_empty() {
        return Ok(());
    }

    let mut mentions_by_frame: HashMap<FrameId, Vec<&TemporalMention>> = HashMap::new();
    for mention in &track.mentions {
        if frame_ids.contains(&mention.frame_id) {
            mentions_by_frame
                .entry(mention.frame_id)
                .or_default()
                .push(mention);
        }
    }

    let mut canonical_cache: HashMap<FrameId, String> = HashMap::new();

    for hit in hits.iter_mut() {
        let frame_id = hit.frame_id;
        let metadata = hit.metadata.get_or_insert_with(SearchHitMetadata::default);

        let mut temporal = SearchHitTemporal::default();

        if let Some(anchor) = track.anchor_for_frame(frame_id) {
            temporal.anchor = Some(SearchHitTemporalAnchor {
                ts_utc: anchor.anchor_ts,
                iso_8601: timestamp_to_rfc3339(anchor.anchor_ts),
                source: anchor.source,
            });
        }

        if let Some(mentions) = mentions_by_frame.get(&frame_id) {
            let mut collected = Vec::new();
            for mention in mentions {
                let mention_start = mention.byte_start as usize;
                let mention_end = mention_start.saturating_add(mention.byte_len as usize);
                if mention_start == mention_end {
                    continue;
                }
                let (hit_start, hit_end) = hit.range;
                if mention_end <= hit_start || mention_start >= hit_end {
                    continue;
                }

                let text = if mention_end > mention_start {
                    if !canonical_cache.contains_key(&frame_id) {
                        let frame = memvid.toc.frames.get(frame_id as usize).cloned().ok_or(
                            MemvidError::InvalidTimeIndex {
                                reason: "frame id out of range".into(),
                            },
                        )?;
                        let content = memvid.frame_content(&frame)?;
                        canonical_cache.insert(frame_id, content);
                    }
                    canonical_cache.get(&frame_id).and_then(|content| {
                        if mention_end <= content.len() {
                            let slice = &content.as_bytes()[mention_start..mention_end];
                            let raw = String::from_utf8_lossy(slice).to_string();
                            let trimmed = raw.trim();
                            if trimmed.is_empty() {
                                None
                            } else {
                                Some(trimmed.to_owned())
                            }
                        } else {
                            None
                        }
                    })
                } else {
                    None
                };

                collected.push(SearchHitTemporalMention {
                    ts_utc: mention.ts_utc,
                    iso_8601: timestamp_to_rfc3339(mention.ts_utc),
                    kind: mention.kind,
                    confidence: mention.confidence,
                    flags: mention.flags,
                    text,
                    byte_start: mention.byte_start,
                    byte_len: mention.byte_len,
                });
            }

            if !collected.is_empty() {
                temporal.mentions = collected;
            }
        }

        if temporal.anchor.is_some() || !temporal.mentions.is_empty() {
            metadata.temporal = Some(temporal);
        }
    }

    Ok(())
}

/// Enrich search hits with entities from the Logic-Mesh.
///
/// For each hit, looks up entities that are associated with the hit's frame.
/// If the frame is a DocumentChunk (page), also checks the parent document frame
/// for entities since NER extraction happens on the full document.
pub(super) fn enrich_hits_with_entities(hits: &mut [SearchHit], memvid: &Memvid) {
    for hit in hits.iter_mut() {
        let mut entities = memvid.frame_entities_for_search(hit.frame_id);

        // If no entities found and this is a chunk, check the parent frame
        if entities.is_empty() {
            if let Some(frame) = memvid.toc.frames.get(hit.frame_id as usize) {
                if let Some(parent_id) = frame.parent_id {
                    entities = memvid.frame_entities_for_search(parent_id);
                }
            }
        }

        if !entities.is_empty() {
            let metadata = hit.metadata.get_or_insert_with(SearchHitMetadata::default);
            metadata.entities = entities;
        }
    }
}
