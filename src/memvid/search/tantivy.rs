// Safe unwrap: regex from validated patterns.
#![allow(clippy::unwrap_used)]
#![cfg(feature = "lex")]
#[cfg(feature = "temporal_track")]
use super::helpers::attach_temporal_metadata;
use super::helpers::{
    build_context, collect_token_occurrences, parse_cursor, timestamp_to_rfc3339,
};
use crate::lex::compute_snippet_slices;
use crate::memvid::frame::ChunkInfo;
use crate::memvid::lifecycle::Memvid;
use crate::search::{EvaluationContext, ParsedQuery};
use crate::types::{
    FrameId, SearchEngineKind, SearchHit, SearchHitMetadata, SearchParams, SearchRequest,
    SearchResponse,
};
use crate::{MemvidError, Result};
use log::warn;
use std::collections::HashSet;
use std::time::Instant;

pub(super) fn try_tantivy_search(
    memvid: &mut Memvid,
    parsed: &ParsedQuery,
    query_tokens: &[String],
    request: &SearchRequest,
    params: &SearchParams,
    start_time: Instant,
    candidate_filter: Option<&HashSet<FrameId>>,
) -> Result<Option<SearchResponse>> {
    let engine = match memvid.tantivy.as_ref() {
        Some(engine) => engine,
        None => {
            return Ok(None);
        }
    };

    // Use stemmed tokens for evaluation to match what Tantivy indexed.
    // Tantivy stems during indexing (e.g., "technology" → "technolog"), so we need
    // to search for stemmed forms in the content as well.
    let mut stemmed_tokens = Vec::new();
    for token in query_tokens {
        let analyzed = engine.analyse_text(token);
        stemmed_tokens.extend(analyzed);
    }
    let stemmed_tokens = stemmed_tokens;

    let offset_hint = request
        .cursor
        .as_deref()
        .and_then(|cursor| cursor.parse::<usize>().ok())
        .unwrap_or(0);
    let base_docs = request.top_k.max(1) + offset_hint;
    let mut doc_limit = base_docs.saturating_mul(4).max(20);
    if let Some(filter) = candidate_filter {
        doc_limit = doc_limit.min(filter.len().max(1));
    }
    let uri_filter = request.uri.as_deref();
    let scope_filter = if uri_filter.is_some() {
        None
    } else {
        request.scope.as_deref()
    };

    let frame_filter_vec: Option<Vec<u64>> =
        candidate_filter.map(|set| set.iter().copied().collect());
    let frame_filter_slice = frame_filter_vec.as_deref();

    let search_hits = match engine.search_documents(
        parsed,
        uri_filter,
        scope_filter,
        frame_filter_slice,
        doc_limit,
    ) {
        Ok(hits) => hits,
        Err(err) => {
            warn!("tantivy search failed: {}", err);
            return Ok(None);
        }
    };
    tracing::debug!(
        "tantivy hits for query '{}': {}",
        request.query,
        search_hits.len()
    );
    if search_hits.is_empty() {
        // Fall back to legacy lex search when Tantivy yields no hits. This avoids silent
        // zero-hit responses when the analyzer drops tokens (e.g., qtoken_123).
        // BUT only fall back if lex_index actually exists and has data.
        let has_lex_data = memvid
            .toc
            .indexes
            .lex
            .as_ref()
            .map(|manifest| manifest.bytes_length > 0)
            .unwrap_or(false);
        if has_lex_data {
            memvid.ensure_lex_index()?;
            return Ok(Some(super::fallback::search_with_lex_fallback(
                memvid,
                parsed,
                query_tokens,
                request,
                params,
                start_time,
                candidate_filter,
            )?));
        }
        // No lex fallback available, return empty Tantivy results
        let elapsed = start_time.elapsed().as_millis();
        return Ok(Some(super::helpers::empty_search_response(
            request.query.clone(),
            params.clone(),
            elapsed,
            crate::types::SearchEngineKind::Tantivy,
        )));
    }

    let snippet_window = request.snippet_chars.max(80);
    let max_snippets_per_doc = request.top_k.max(1);
    let mut evaluated = Vec::new();
    for hit in search_hits {
        let frame_meta = memvid
            .toc
            .frames
            .get(hit.frame_id as usize)
            .cloned()
            .ok_or(MemvidError::InvalidTimeIndex {
                reason: "frame id out of range".into(),
            })?;
        let _ = &hit.content;
        if let Some(uri_expected) = uri_filter {
            if !uri_matches(frame_meta.uri.as_deref(), uri_expected) {
                continue;
            }
        } else if let Some(scope) = scope_filter {
            match frame_meta.uri.as_deref() {
                Some(uri) if uri.starts_with(scope) => {}
                _ => continue,
            }
        }

        let chunk_info = match memvid.resolve_chunk_context(&frame_meta) {
            Ok(info) => info,
            Err(err) => {
                warn!(
                    "unable to resolve chunk context for frame {}: {}",
                    frame_meta.id, err
                );
                continue;
            }
        };

        let content_lower = chunk_info.text.to_ascii_lowercase();

        // Skip redundant evaluation for text terms - Tantivy already matched them using stemming.
        // We still need to evaluate field terms (uri, track, tags, etc.) to filter results.
        let ctx = EvaluationContext {
            frame: &frame_meta,
            content_lower: &content_lower,
        };
        // Only evaluate if there are field terms; skip text-only queries since Tantivy already matched
        if parsed.contains_field_terms() && !parsed.evaluate(&ctx) {
            tracing::debug!(
                "tantivy hit {} culled: failed query evaluation",
                frame_meta.id
            );
            continue;
        }
        let occurrences = collect_token_occurrences(&content_lower, &stemmed_tokens);
        let slices = compute_snippet_slices(
            &chunk_info.text,
            &occurrences,
            snippet_window,
            max_snippets_per_doc,
        );
        if slices.is_empty() {
            tracing::debug!("tantivy hit {} culled: no snippet slices", frame_meta.id);
            continue;
        }
        // Use content_dates if available, otherwise fall back to frame timestamp
        let effective_ts = parse_content_date_to_timestamp(&frame_meta.content_dates)
            .unwrap_or(frame_meta.timestamp);
        evaluated.push((hit, occurrences, slices, chunk_info, effective_ts));
    }

    // Apply recency boosting: re-sort by combined score (BM25 + recency)
    // This helps knowledge-update questions find the most recent information
    if evaluated.len() > 1 {
        // Use RELATIVE recency within the result set, not absolute time from "now"
        // This ensures documents from different time periods are fairly compared
        let max_ts = evaluated
            .iter()
            .map(|(_, _, _, _, ts)| *ts)
            .max()
            .unwrap_or(0);

        // Calculate recency-boosted scores and attach to items
        let mut with_scores: Vec<(f32, _)> = evaluated
            .into_iter()
            .map(|(hit, occurrences, slices, chunk_info, timestamp)| {
                let bm25_score = hit.score;
                // Age relative to the most recent document in results
                let age_seconds = (max_ts - timestamp).max(0) as f32;
                // Decay factor: half-life of ~1 day for aggressive recency preference
                // This ensures even a few days difference has significant impact
                let decay_factor = 0.00000802; // ln(2) / 86400 (1 day)
                let recency_boost = (-decay_factor * age_seconds).exp();
                // Combine: 40% BM25 + 60% recency boost - strongly prefer recent
                let combined_score = bm25_score * 0.4 + (bm25_score * recency_boost * 0.6);
                (
                    combined_score,
                    (hit, occurrences, slices, chunk_info, timestamp),
                )
            })
            .collect();

        // Sort by combined score (descending)
        with_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Extract back to evaluated
        evaluated = with_scores.into_iter().map(|(_, item)| item).collect();
    }

    if evaluated.is_empty() {
        tracing::debug!("tantivy evaluation produced zero hits; falling back to legacy lex",);
        memvid.ensure_lex_index()?;
        return Ok(Some(super::fallback::search_with_lex_fallback(
            memvid,
            parsed,
            query_tokens,
            request,
            params,
            start_time,
            candidate_filter,
        )?));
    }

    let total_slices: usize = evaluated
        .iter()
        .map(|(_, _, slices, _, _)| slices.len())
        .sum();
    if total_slices == 0 {
        tracing::debug!(
            "tantivy evaluation produced zero total slices; falling back to legacy lex",
        );
        memvid.ensure_lex_index()?;
        return Ok(Some(super::fallback::search_with_lex_fallback(
            memvid,
            parsed,
            query_tokens,
            request,
            params,
            start_time,
            candidate_filter,
        )?));
    }

    let offset = parse_cursor(request.cursor.as_deref(), total_slices)?;
    let effective_top_k = request.top_k.max(1);

    let mut hits = Vec::new();
    let mut produced = 0usize;
    for (hit, occurrences, slices, chunk_info, _timestamp) in evaluated {
        if hits.len() == effective_top_k && produced >= offset {
            break;
        }
        let frame_meta = memvid
            .toc
            .frames
            .get(hit.frame_id as usize)
            .cloned()
            .ok_or(MemvidError::InvalidTimeIndex {
                reason: "frame id out of range".into(),
            })?;
        let uri = frame_meta
            .uri
            .clone()
            .unwrap_or_else(|| crate::default_uri(hit.frame_id));
        let title = frame_meta
            .title
            .clone()
            .or_else(|| crate::infer_title_from_uri(&uri));

        let ChunkInfo {
            start: chunk_start,
            end: chunk_end,
            text: chunk_text,
        } = chunk_info;
        let chunk_bytes = chunk_text.as_bytes();
        let chunk_range = (chunk_start, chunk_end);

        for (start, end) in slices {
            if produced < offset {
                produced += 1;
                continue;
            }
            if hits.len() == effective_top_k {
                break;
            }
            let local_start = start.min(chunk_bytes.len());
            let local_end = end.min(chunk_bytes.len());
            if local_end <= local_start {
                produced += 1;
                continue;
            }
            let matches_in_slice = occurrences
                .iter()
                .filter(|(s, e)| *s >= local_start && *e <= local_end)
                .count()
                .max(1);
            let metadata = SearchHitMetadata {
                matches: matches_in_slice,
                tags: frame_meta.tags.clone(),
                labels: frame_meta.labels.clone(),
                track: frame_meta.track.clone(),
                created_at: timestamp_to_rfc3339(frame_meta.timestamp),
                content_dates: frame_meta.content_dates.clone(),
                entities: Vec::new(),
                extra_metadata: frame_meta.extra_metadata.clone(),
                #[cfg(feature = "temporal_track")]
                temporal: None,
            };
            let global_start = chunk_start + local_start;
            let global_end = chunk_start + local_end;
            if global_end <= global_start {
                produced += 1;
                continue;
            }
            let snippet_text = chunk_text[local_start..local_end].to_string();
            hits.push(SearchHit {
                rank: hits.len() + 1,
                frame_id: hit.frame_id,
                uri: uri.clone(),
                title: title.clone(),
                range: (global_start, global_end),
                text: snippet_text,
                matches: matches_in_slice,
                chunk_range: Some(chunk_range),
                chunk_text: Some(chunk_text.clone()),
                score: Some(hit.score),
                metadata: Some(metadata),
            });
            produced += 1;
        }
    }

    let next_cursor = if produced < total_slices {
        Some(produced.to_string())
    } else {
        None
    };
    #[cfg(feature = "temporal_track")]
    attach_temporal_metadata(memvid, &mut hits)?;
    let elapsed_ms = start_time.elapsed().as_millis().max(1);
    let context = build_context(&hits);

    Ok(Some(SearchResponse {
        query: request.query.clone(),
        elapsed_ms,
        total_hits: total_slices,
        params: params.clone(),
        hits,
        context,
        next_cursor,
        engine: SearchEngineKind::Tantivy,
    }))
}

fn uri_matches(candidate: Option<&str>, expected: &str) -> bool {
    let Some(uri) = candidate else {
        return false;
    };
    if expected.contains('#') {
        uri.eq_ignore_ascii_case(expected)
    } else {
        let expected_lower = expected.to_ascii_lowercase();
        let candidate_lower = uri.to_ascii_lowercase();
        candidate_lower.starts_with(&expected_lower)
    }
}

/// Parse content dates (from frame metadata) to find the most relevant timestamp.
/// Content dates are strings like "2023/06/30 (Fri) 14:20", ISO dates, or spelled-out dates.
/// Returns the most recent timestamp found, or None if parsing fails.
pub fn parse_content_date_to_timestamp(content_dates: &[String]) -> Option<i64> {
    if content_dates.is_empty() {
        return None;
    }

    let mut best_ts: Option<i64> = None;

    for date_str in content_dates {
        // Try to parse format "YYYY/MM/DD (Day) HH:MM" -> "2023/06/30 (Fri) 14:20"
        if let Some(ts) = parse_custom_date_format(date_str) {
            best_ts = Some(best_ts.map_or(ts, |prev| prev.max(ts)));
            continue;
        }

        // Try ISO format
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(date_str) {
            let ts = dt.timestamp();
            best_ts = Some(best_ts.map_or(ts, |prev| prev.max(ts)));
            continue;
        }

        // Try simple YYYY-MM-DD
        if let Ok(date) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            let ts = date
                .and_hms_opt(0, 0, 0)
                .map(|dt| dt.and_utc().timestamp())
                .unwrap_or(0);
            if ts > 0 {
                best_ts = Some(best_ts.map_or(ts, |prev| prev.max(ts)));
                continue;
            }
        }

        // Try spelled-out dates: "September 1, 2024" or "Sept 10, 2024"
        if let Some(ts) = parse_spelled_date(date_str) {
            best_ts = Some(best_ts.map_or(ts, |prev| prev.max(ts)));
            continue;
        }

        // Try European format: "1 September 2024"
        if let Some(ts) = parse_euro_date(date_str) {
            best_ts = Some(best_ts.map_or(ts, |prev| prev.max(ts)));
            continue;
        }

        // Try year-only: "2024" -> January 1, 2024
        // Only accept reasonable years (1900-2100) to avoid matching employee IDs like "2044"
        if let Ok(year) = date_str.trim().parse::<i32>() {
            if (1900..=2100).contains(&year) {
                if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, 1, 1) {
                    let ts = date
                        .and_hms_opt(0, 0, 0)
                        .map(|dt| dt.and_utc().timestamp())
                        .unwrap_or(0);
                    if ts > 0 {
                        best_ts = Some(best_ts.map_or(ts, |prev| prev.max(ts)));
                    }
                }
            }
        }
    }

    best_ts
}

/// Parse spelled-out date like "September 1, 2024" or "Sept 10, 2024" or "September 1st, 2024"
fn parse_spelled_date(s: &str) -> Option<i64> {
    // Normalize whitespace: replace newlines and multiple spaces with single space
    // PDF extraction can produce "September\n1,\n2024" instead of "September 1, 2024"
    let normalized: String = s.split_whitespace().collect::<Vec<_>>().join(" ");

    // Strip ordinal suffixes (1st, 2nd, 3rd, 4th, etc.)
    let without_ordinals = strip_ordinal_suffixes(&normalized);

    // Common formats: "September 1, 2024", "Sept 10, 2024", "March 15 2024"
    let formats = [
        "%B %d, %Y", // September 1, 2024
        "%B %d %Y",  // September 1 2024
        "%b %d, %Y", // Sep 1, 2024
        "%b %d %Y",  // Sep 1 2024
    ];

    for fmt in &formats {
        if let Ok(date) = chrono::NaiveDate::parse_from_str(&without_ordinals, fmt) {
            return date.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc().timestamp());
        }
    }
    None
}

/// Strip ordinal suffixes from day numbers: "1st" -> "1", "2nd" -> "2", etc.
fn strip_ordinal_suffixes(s: &str) -> String {
    static ORDINAL_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"(\d+)(?:st|nd|rd|th)\b").unwrap());
    ORDINAL_RE.replace_all(s, "$1").to_string()
}

/// Parse European date like "1 September 2024" or "1st September 2024"
fn parse_euro_date(s: &str) -> Option<i64> {
    // Normalize whitespace for PDF-extracted dates
    let normalized: String = s.split_whitespace().collect::<Vec<_>>().join(" ");

    // Strip ordinal suffixes (1st, 2nd, 3rd, 4th, etc.)
    let without_ordinals = strip_ordinal_suffixes(&normalized);

    let formats = [
        "%d %B %Y", // 1 September 2024
        "%d %b %Y", // 1 Sep 2024
    ];

    for fmt in &formats {
        if let Ok(date) = chrono::NaiveDate::parse_from_str(&without_ordinals, fmt) {
            return date.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc().timestamp());
        }
    }
    None
}

/// Parse custom date format "YYYY/MM/DD (Day) HH:MM" to timestamp
fn parse_custom_date_format(s: &str) -> Option<i64> {
    // Format: "2023/06/30 (Fri) 14:20"
    // We'll extract: 2023, 06, 30, 14, 20

    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }

    // First part should be date: "2023/06/30"
    let date_parts: Vec<&str> = parts[0].split('/').collect();
    if date_parts.len() != 3 {
        return None;
    }

    let year: i32 = date_parts[0].parse().ok()?;
    let month: u32 = date_parts[1].parse().ok()?;
    let day: u32 = date_parts[2].parse().ok()?;

    // Look for time part (HH:MM)
    let (hour, minute) = if parts.len() >= 3 {
        // Skip "(Day)" part, get time
        let time_str = parts.iter().find(|p| p.contains(':'))?;
        let time_parts: Vec<&str> = time_str.split(':').collect();
        if time_parts.len() >= 2 {
            (
                time_parts[0].parse::<u32>().ok()?,
                time_parts[1].parse::<u32>().ok()?,
            )
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    };

    let date = chrono::NaiveDate::from_ymd_opt(year, month, day)?;
    let datetime = date.and_hms_opt(hour, minute, 0)?;
    Some(datetime.and_utc().timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_spelled_date_with_newlines() {
        // PDF extraction produces dates with newlines
        let date_with_newlines = "September\n1,\n2024";
        let ts = parse_spelled_date(date_with_newlines);
        assert!(ts.is_some(), "Should parse date with newlines");

        // Verify it's September 1, 2024 (UTC timestamp)
        let expected = chrono::NaiveDate::from_ymd_opt(2024, 9, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        assert_eq!(ts.unwrap(), expected);
    }

    #[test]
    fn test_parse_content_date_picks_most_recent() {
        // Test that we pick the most recent date from multiple options
        let dates = vec![
            "2024".to_string(),                // year only
            "September\n1,\n2024".to_string(), // spelled out with newlines
        ];
        let ts = parse_content_date_to_timestamp(&dates);
        assert!(ts.is_some(), "Should parse at least one date");

        // September 1, 2024 should be more recent than just "2024" (Jan 1, 2024)
        let sept_ts = chrono::NaiveDate::from_ymd_opt(2024, 9, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        assert_eq!(ts.unwrap(), sept_ts);
    }

    #[test]
    fn test_parse_ordinal_dates() {
        // Test ordinal suffixes: 1st, 2nd, 3rd, 4th, etc.
        let ts1 = parse_spelled_date("September 1st, 2024");
        assert!(ts1.is_some(), "Should parse 'September 1st, 2024'");

        let ts2 = parse_spelled_date("March 22nd, 2024");
        assert!(ts2.is_some(), "Should parse 'March 22nd, 2024'");

        let ts3 = parse_euro_date("3rd October 2024");
        assert!(ts3.is_some(), "Should parse '3rd October 2024'");

        // Verify correct day is parsed
        let expected = chrono::NaiveDate::from_ymd_opt(2024, 9, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        assert_eq!(ts1.unwrap(), expected);
    }
}
