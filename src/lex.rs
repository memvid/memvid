use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
};

use blake3::hash;
use serde::{Deserialize, Serialize};

use crate::{MemvidError, Result, types::FrameId};

// Bincode configuration reused for deterministic layout.
fn lex_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian()
}

#[allow(clippy::cast_possible_truncation)]
const LEX_DECODE_LIMIT: usize = crate::MAX_INDEX_BYTES as usize;
const LEX_SECTION_SOFT_CHARS: usize = 900;
const LEX_SECTION_HARD_CHARS: usize = 1400;
const LEX_SECTION_MAX_COUNT: usize = 2048;

/// Intermediate builder that collects documents prior to serialisation.
#[derive(Default)]
pub struct LexIndexBuilder {
    documents: Vec<LexDocument>,
}

impl LexIndexBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_document(
        &mut self,
        frame_id: FrameId,
        uri: &str,
        title: Option<&str>,
        content: &str,
        tags: &HashMap<String, String>,
    ) {
        let tokens = tokenize(content);
        // Convert HashMap to BTreeMap for deterministic serialization
        let tags: BTreeMap<_, _> = tags.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let mut sections = chunk_sections(content);

        let (content_owned, content_lower) = if content.is_empty() {
            (String::new(), String::new())
        } else if sections.is_empty() {
            let owned = content.to_string();
            let lower = owned.to_ascii_lowercase();
            sections.push(LexSection {
                offset: 0,
                content: owned.clone(),
                content_lower: lower.clone(),
            });
            (owned, lower)
        } else {
            (String::new(), String::new())
        };
        self.documents.push(LexDocument {
            frame_id,
            tokens,
            tags,
            content: content_owned,
            content_lower,
            uri: Some(uri.to_string()),
            title: title.map(ToString::to_string),
            sections,
        });
    }

    pub fn finish(mut self) -> Result<LexIndexArtifact> {
        for document in &mut self.documents {
            document.ensure_sections();
        }
        let bytes = bincode::serde::encode_to_vec(&self.documents, lex_config())?;
        let checksum = *hash(&bytes).as_bytes();
        Ok(LexIndexArtifact {
            bytes,
            doc_count: self.documents.len() as u64,
            checksum,
        })
    }
}

/// Serialized lexical index artifact ready to be embedded in the `.mv2` file.
#[derive(Debug, Clone)]
pub struct LexIndexArtifact {
    pub bytes: Vec<u8>,
    pub doc_count: u64,
    pub checksum: [u8; 32],
}

/// Read-only lexical index decoded from persisted bytes.
#[derive(Debug, Clone)]
pub struct LexIndex {
    documents: Vec<LexDocument>,
}

impl LexIndex {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let new_config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian()
            .with_limit::<LEX_DECODE_LIMIT>();
        if let Ok((documents, read)) =
            bincode::serde::decode_from_slice::<Vec<LexDocument>, _>(bytes, new_config)
        {
            if read == bytes.len() {
                return Ok(Self::from_documents(documents));
            }
        }

        let legacy_fixed = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian()
            .with_limit::<LEX_DECODE_LIMIT>();
        if let Ok((legacy_docs, read)) =
            bincode::serde::decode_from_slice::<Vec<LegacyLexDocument>, _>(bytes, legacy_fixed)
        {
            if read == bytes.len() {
                let documents = legacy_docs.into_iter().map(legacy_to_current).collect();
                return Ok(Self::from_documents(documents));
            }
        }

        let legacy_config = bincode::config::standard()
            .with_little_endian()
            .with_limit::<LEX_DECODE_LIMIT>();
        if let Ok((legacy_docs, read)) =
            bincode::serde::decode_from_slice::<Vec<LegacyLexDocument>, _>(bytes, legacy_config)
        {
            if read == bytes.len() {
                let documents = legacy_docs.into_iter().map(legacy_to_current).collect();
                return Ok(Self::from_documents(documents));
            }
        }

        Err(MemvidError::InvalidToc {
            reason: "unsupported lex index encoding".into(),
        })
    }

    fn from_documents(mut documents: Vec<LexDocument>) -> Self {
        for document in &mut documents {
            document.ensure_sections();
        }
        Self { documents }
    }

    #[must_use]
    pub fn search(&self, query: &str, limit: usize) -> Vec<LexSearchHit> {
        let mut query_tokens = parse_query_terms(query);
        query_tokens.retain(|token| !token.is_empty());
        if query_tokens.is_empty() {
            return Vec::new();
        }
        let mut matches = self.compute_matches(&query_tokens, None, None);
        matches.truncate(limit);
        matches
            .into_iter()
            .map(|m| {
                let snippets = build_snippets(&m.content, &m.occurrences, 160, 3);
                LexSearchHit {
                    frame_id: m.frame_id,
                    score: m.score,
                    match_count: m.occurrences.len(),
                    snippets,
                }
            })
            .collect()
    }

    pub(crate) fn documents_mut(&mut self) -> &mut [LexDocument] {
        &mut self.documents
    }

    pub(crate) fn remove_document(&mut self, frame_id: FrameId) {
        self.documents.retain(|doc| doc.frame_id != frame_id);
    }

    pub(crate) fn compute_matches(
        &self,
        query_tokens: &[String],
        uri_filter: Option<&str>,
        scope_filter: Option<&str>,
    ) -> Vec<LexMatch> {
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut hits = Vec::new();
        let phrase = query_tokens.join(" ");
        for document in &self.documents {
            if let Some(uri) = uri_filter {
                if !uri_matches(document.uri.as_deref(), uri) {
                    continue;
                }
            } else if let Some(scope) = scope_filter {
                match document.uri.as_deref() {
                    Some(candidate) if candidate.starts_with(scope) => {}
                    _ => continue,
                }
            }

            if document.sections.is_empty() {
                continue;
            }

            for section in &document.sections {
                let haystack = section.content_lower.as_str();
                if haystack.is_empty() {
                    continue;
                }

                let mut occurrences: Vec<(usize, usize)> = Vec::new();

                if query_tokens.len() == 1 {
                    let needle = &query_tokens[0];
                    if needle.is_empty() {
                        continue;
                    }
                    let mut start = 0usize;
                    while let Some(idx) = haystack[start..].find(needle) {
                        let local_start = start + idx;
                        let local_end = local_start + needle.len();
                        occurrences.push((local_start, local_end));
                        start = local_end;
                    }
                } else {
                    let mut all_occurrences = Vec::new();
                    let mut all_present = true;
                    for needle in query_tokens {
                        if needle.is_empty() {
                            all_present = false;
                            break;
                        }
                        let mut start = 0usize;
                        let mut found_for_token = false;
                        while let Some(idx) = haystack[start..].find(needle) {
                            found_for_token = true;
                            let local_start = start + idx;
                            let local_end = local_start + needle.len();
                            all_occurrences.push((local_start, local_end));
                            start = local_end;
                        }
                        if !found_for_token {
                            all_present = false;
                            break;
                        }
                    }
                    if !all_present {
                        continue;
                    }
                    occurrences = all_occurrences;
                }

                if occurrences.is_empty() {
                    continue;
                }

                occurrences.sort_by_key(|(start, _)| *start);
                #[allow(clippy::cast_precision_loss)]
                let mut score = occurrences.len() as f32;
                if !phrase.is_empty() && section.content_lower.contains(&phrase) {
                    score += 1000.0;
                }
                hits.push(LexMatch {
                    frame_id: document.frame_id,
                    score,
                    occurrences,
                    content: section.content.clone(),
                    uri: document.uri.clone(),
                    title: document.title.clone(),
                    chunk_offset: section.offset,
                });
            }
        }

        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        // Deduplicate by frame_id, keeping the highest-scoring match for each frame.
        // This prevents the same document from appearing multiple times when it has
        // multiple sections that match the query.
        let mut seen_frames: std::collections::HashSet<FrameId> = std::collections::HashSet::new();
        let mut deduped = Vec::with_capacity(hits.len());
        for hit in hits {
            if seen_frames.insert(hit.frame_id) {
                deduped.push(hit);
            }
        }
        deduped
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LexDocument {
    pub(crate) frame_id: FrameId,
    tokens: Vec<String>,
    tags: BTreeMap<String, String>,
    #[serde(default)]
    content: String,
    #[serde(default)]
    pub(crate) content_lower: String,
    #[serde(default)]
    pub(crate) uri: Option<String>,
    #[serde(default)]
    pub(crate) title: Option<String>,
    #[serde(default)]
    sections: Vec<LexSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LexSection {
    pub(crate) offset: usize,
    #[serde(default)]
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) content_lower: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyLexDocument {
    frame_id: FrameId,
    tokens: Vec<String>,
    tags: BTreeMap<String, String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    uri: Option<String>,
    #[serde(default)]
    title: Option<String>,
}

impl LexDocument {
    fn ensure_sections(&mut self) {
        if !self.sections.is_empty() {
            return;
        }

        if self.content.is_empty() {
            return;
        }

        if self.content_lower.is_empty() {
            self.content_lower = self.content.to_ascii_lowercase();
        }

        self.sections.push(LexSection {
            offset: 0,
            content: self.content.clone(),
            content_lower: self.content_lower.clone(),
        });
    }
}

fn legacy_to_current(legacy: LegacyLexDocument) -> LexDocument {
    let content = legacy.content.unwrap_or_default();
    let content_lower = content.to_ascii_lowercase();
    let sections = if content.is_empty() {
        Vec::new()
    } else {
        vec![LexSection {
            offset: 0,
            content: content.clone(),
            content_lower: content_lower.clone(),
        }]
    };
    LexDocument {
        frame_id: legacy.frame_id,
        tokens: legacy.tokens,
        tags: legacy.tags,
        content,
        content_lower,
        uri: legacy.uri,
        title: legacy.title,
        sections,
    }
}

#[derive(Debug, Clone)]
pub struct LexSearchHit {
    pub frame_id: FrameId,
    pub score: f32,
    pub match_count: usize,
    pub snippets: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct LexMatch {
    pub frame_id: FrameId,
    pub score: f32,
    pub occurrences: Vec<(usize, usize)>,
    pub content: String,
    pub uri: Option<String>,
    pub title: Option<String>,
    pub chunk_offset: usize,
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|c: char| !is_token_char(c))
        .filter_map(|token| {
            if token.chars().any(char::is_alphanumeric) {
                Some(token.to_lowercase())
            } else {
                None
            }
        })
        .collect()
}

fn parse_query_terms(input: &str) -> Vec<String> {
    let trimmed = input.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        let phrase = trimmed.trim_matches('"').trim().to_ascii_lowercase();
        if !phrase.is_empty() {
            return vec![phrase];
        }
    }

    tokenize(input)
}

fn is_token_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '&' | '@' | '+' | '/' | '_')
}

fn build_snippets(
    content: &str,
    occurrences: &[(usize, usize)],
    window: usize,
    max_snippets: usize,
) -> Vec<String> {
    compute_snippet_slices(content, occurrences, window, max_snippets)
        .into_iter()
        .map(|(start, end)| content[start..end].replace('\n', " "))
        .collect()
}

fn chunk_sections(content: &str) -> Vec<LexSection> {
    if content.is_empty() {
        return Vec::new();
    }

    if content.len() <= LEX_SECTION_HARD_CHARS {
        return vec![LexSection {
            offset: 0,
            content: content.to_string(),
            content_lower: content.to_ascii_lowercase(),
        }];
    }

    let mut sections: Vec<LexSection> = Vec::new();
    let mut chunk_start = 0usize;
    let mut last_soft_break = None;
    let mut iter = content.char_indices().peekable();

    while let Some((idx, ch)) = iter.next() {
        let char_end = idx + ch.len_utf8();
        let current_len = char_end.saturating_sub(chunk_start);
        let next_char = iter.peek().map(|(_, next)| *next);

        if is_soft_boundary(ch, next_char) {
            last_soft_break = Some(char_end);
            if current_len < LEX_SECTION_SOFT_CHARS {
                continue;
            }
        }

        if current_len < LEX_SECTION_HARD_CHARS {
            continue;
        }

        let mut split_at = last_soft_break.unwrap_or(char_end);
        if split_at <= chunk_start {
            split_at = char_end;
        }

        push_section(&mut sections, content, chunk_start, split_at);
        chunk_start = split_at;
        last_soft_break = None;

        if sections.len() >= LEX_SECTION_MAX_COUNT {
            break;
        }
    }

    if chunk_start < content.len() {
        if sections.len() >= LEX_SECTION_MAX_COUNT {
            if let Some(last) = sections.last_mut() {
                let slice = &content[last.offset..];
                last.content = slice.to_string();
                last.content_lower = slice.to_ascii_lowercase();
            }
        } else {
            push_section(&mut sections, content, chunk_start, content.len());
        }
    }

    if sections.is_empty() {
        sections.push(LexSection {
            offset: 0,
            content: content.to_string(),
            content_lower: content.to_ascii_lowercase(),
        });
    }

    sections
}

fn push_section(sections: &mut Vec<LexSection>, content: &str, start: usize, end: usize) {
    if end <= start {
        return;
    }

    let slice = &content[start..end];
    sections.push(LexSection {
        offset: start,
        content: slice.to_string(),
        content_lower: slice.to_ascii_lowercase(),
    });
}

fn is_soft_boundary(ch: char, next: Option<char>) -> bool {
    match ch {
        '.' | '!' | '?' => next.is_none_or(char::is_whitespace),
        '\n' => true,
        _ => false,
    }
}

pub(crate) fn compute_snippet_slices(
    content: &str,
    occurrences: &[(usize, usize)],
    window: usize,
    max_snippets: usize,
) -> Vec<(usize, usize)> {
    if content.is_empty() {
        return Vec::new();
    }

    if occurrences.is_empty() {
        let end = advance_boundary(content, 0, window);
        return vec![(0, end)];
    }

    let mut merged: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in occurrences {
        let mut snippet_start = start.saturating_sub(window / 2);
        let mut snippet_end = (end + window / 2).min(content.len());

        if let Some(adj) = sentence_start_before(content, snippet_start) {
            snippet_start = adj;
        }
        if let Some(adj) = sentence_end_after(content, snippet_end) {
            snippet_end = adj;
        }

        snippet_start = prev_char_boundary(content, snippet_start);
        snippet_end = next_char_boundary(content, snippet_end);

        if snippet_end <= snippet_start {
            continue;
        }

        if let Some(last) = merged.last_mut() {
            if snippet_start <= last.1 + 20 {
                last.1 = last.1.max(snippet_end);
                continue;
            }
        }

        merged.push((
            snippet_start.min(content.len()),
            snippet_end.min(content.len()),
        ));
        if merged.len() >= max_snippets {
            break;
        }
    }

    if merged.is_empty() {
        let end = advance_boundary(content, 0, window);
        merged.push((0, end));
    }

    merged
}

fn sentence_start_before(content: &str, idx: usize) -> Option<usize> {
    if idx == 0 {
        return Some(0);
    }
    let mut idx = idx.min(content.len());
    idx = prev_char_boundary(content, idx);
    let mut candidate = None;
    for (pos, ch) in content[..idx].char_indices() {
        if matches!(ch, '.' | '!' | '?' | '\n') {
            candidate = Some(pos + ch.len_utf8());
        }
    }
    candidate.map(|pos| {
        let mut pos = next_char_boundary(content, pos);
        while pos < content.len() && content.as_bytes()[pos].is_ascii_whitespace() {
            pos += 1;
        }
        prev_char_boundary(content, pos)
    })
}

fn sentence_end_after(content: &str, idx: usize) -> Option<usize> {
    if idx >= content.len() {
        return Some(content.len());
    }
    let mut idx = idx;
    idx = prev_char_boundary(content, idx);
    for (offset, ch) in content[idx..].char_indices() {
        let global = idx + offset;
        if matches!(ch, '.' | '!' | '?') {
            return Some(next_char_boundary(content, global + ch.len_utf8()));
        }
        if ch == '\n' {
            return Some(global);
        }
    }
    None
}

fn prev_char_boundary(content: &str, mut idx: usize) -> usize {
    if idx > content.len() {
        idx = content.len();
    }
    while idx > 0 && !content.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn next_char_boundary(content: &str, mut idx: usize) -> usize {
    if idx > content.len() {
        idx = content.len();
    }
    while idx < content.len() && !content.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

fn advance_boundary(content: &str, start: usize, mut window: usize) -> usize {
    if start >= content.len() {
        return content.len();
    }
    let mut last = content.len();
    for (offset, _) in content[start..].char_indices() {
        if window == 0 {
            return start + offset;
        }
        last = start + offset;
        window -= 1;
    }
    content.len().max(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_produces_artifact() {
        let mut builder = LexIndexBuilder::new();
        let mut tags = HashMap::new();
        tags.insert("source".into(), "test".into());
        builder.add_document(0, "mv2://docs/one", Some("Doc One"), "hello world", &tags);
        builder.add_document(
            1,
            "mv2://docs/two",
            Some("Doc Two"),
            "rust systems",
            &HashMap::new(),
        );

        let artifact = builder.finish().expect("finish");
        assert_eq!(artifact.doc_count, 2);
        assert!(!artifact.bytes.is_empty());

        let index = LexIndex::decode(&artifact.bytes).expect("decode");
        let hits = index.search("rust", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].frame_id, 1);
        assert!(hits[0].match_count >= 1);
        assert!(!hits[0].snippets.is_empty());
    }

    #[test]
    fn tokenizer_lowercases_and_filters() {
        let tokens = tokenize("Hello, Rust-lang!");
        assert_eq!(tokens, vec!["hello", "rust", "lang"]);
    }

    #[test]
    fn tokenizer_retains_connector_characters() {
        let tokens = tokenize("N&M EXPRESS LLC @ 2024");
        assert_eq!(tokens, vec!["n&m", "express", "llc", "2024"]);
    }

    #[test]
    fn compute_matches_deduplicates_by_frame_id() {
        // Create a document with content long enough to be split into multiple sections.
        // The section soft limit is 900 chars, hard limit is 1400 chars.
        // We'll create content > 2000 chars with the search term appearing in each section.
        let mut builder = LexIndexBuilder::new();

        // Build content with "quantum" appearing in multiple sections
        let section1 = "Quantum computing represents a revolutionary approach to computation. \
            The fundamental principles of quantum mechanics enable quantum computers to process \
            information in ways classical computers cannot. Quantum bits or qubits can exist in \
            superposition states, allowing quantum algorithms to explore multiple solutions \
            simultaneously. This quantum parallelism offers exponential speedups for certain \
            computational problems. Researchers continue to advance quantum hardware and software. \
            The field of quantum computing is rapidly evolving with new breakthroughs. \
            Major tech companies invest heavily in quantum research and development. \
            Quantum error correction remains a significant challenge for practical quantum computers.";

        let section2 = "Applications of quantum computing span many domains including cryptography, \
            drug discovery, and optimization problems. Quantum cryptography promises unbreakable \
            encryption through quantum key distribution protocols. In the pharmaceutical industry, \
            quantum simulations could revolutionize how we discover new medicines. Quantum \
            algorithms like Shor's algorithm threaten current encryption standards. Financial \
            institutions explore quantum computing for portfolio optimization. The quantum \
            advantage may soon be demonstrated for practical real-world applications. Quantum \
            machine learning combines quantum computing with artificial intelligence techniques. \
            The future of quantum computing holds immense promise for scientific discovery.";

        let full_content = format!("{} {}", section1, section2);
        assert!(
            full_content.len() > 1400,
            "Content should be long enough to create multiple sections"
        );

        builder.add_document(
            42, // frame_id
            "mv2://docs/quantum",
            Some("Quantum Computing Overview"),
            &full_content,
            &HashMap::new(),
        );

        let artifact = builder.finish().expect("finish should succeed");
        let index = LexIndex::decode(&artifact.bytes).expect("decode should succeed");

        // Search for "quantum" which appears many times across both sections
        let query_tokens = tokenize("quantum");
        let matches = index.compute_matches(&query_tokens, None, None);

        // Verify: no duplicate frame_ids in results
        let frame_ids: Vec<_> = matches.iter().map(|m| m.frame_id).collect();
        let unique_frame_ids: std::collections::HashSet<_> = frame_ids.iter().copied().collect();

        assert_eq!(
            frame_ids.len(),
            unique_frame_ids.len(),
            "Results should not contain duplicate frame_ids. Found: {:?}",
            frame_ids
        );

        // Should have exactly one result for frame_id 42
        assert_eq!(matches.len(), 1, "Should have exactly one match");
        assert_eq!(matches[0].frame_id, 42, "Match should be for frame_id 42");
        assert!(matches[0].score > 0.0, "Match should have a positive score");
    }

    #[test]
    fn compute_matches_keeps_highest_score_per_frame() {
        // Test that when multiple sections match, we keep the highest-scoring one
        let mut builder = LexIndexBuilder::new();

        // Create content where "target" appears more times in the second section
        let section1 = "This is the first section with one target mention. \
            It contains various other words to pad the content and make it long enough \
            to be split into multiple sections by the chunking algorithm. We need quite \
            a bit of text here to ensure the sections are created properly. The content \
            continues with more filler text about various topics. Keep writing to reach \
            the section boundary. More text follows to ensure we cross the soft limit. \
            This should be enough to trigger section creation at the boundary point.";

        let section2 = "The second section has target target target multiple times. \
            Target appears here repeatedly: target target target target. This section \
            should score higher because it has more occurrences of the search term target. \
            We mention target again to boost the score further. Target target target. \
            The abundance of target keywords makes this section rank higher in relevance.";

        let full_content = format!("{} {}", section1, section2);

        builder.add_document(
            99,
            "mv2://docs/multi-section",
            Some("Multi-Section Document"),
            &full_content,
            &HashMap::new(),
        );

        let artifact = builder.finish().expect("finish");
        let index = LexIndex::decode(&artifact.bytes).expect("decode");

        let query_tokens = tokenize("target");
        let matches = index.compute_matches(&query_tokens, None, None);

        // Should have exactly one result (deduplicated)
        assert_eq!(
            matches.len(),
            1,
            "Should have exactly one deduplicated match"
        );

        // The match should have the higher score (from section2 with more "target" occurrences)
        // Section1 has 1 occurrence, Section2 has ~10+ occurrences
        assert!(
            matches[0].score >= 5.0,
            "Should keep the highest-scoring match, score was: {}",
            matches[0].score
        );
    }
}
