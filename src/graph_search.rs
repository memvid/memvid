//! Graph-aware search combining MemoryCards/Logic-Mesh with vector search.
//!
//! This module provides hybrid retrieval that can:
//! 1. Parse natural language queries for relational patterns
//! 2. Match patterns against entity state (MemoryCards) or graph (Logic-Mesh)
//! 3. Combine graph-filtered candidates with vector ranking

use std::collections::{HashMap, HashSet};

use crate::types::{
    GraphMatchResult, GraphPattern, HybridSearchHit, PatternTerm, QueryPlan, SearchRequest,
    TriplePattern,
};
use crate::{FrameId, Memvid, Result};

/// Query planner that analyzes queries and creates execution plans.
#[derive(Debug, Default)]
pub struct QueryPlanner {
    /// Patterns for detecting relational queries
    entity_patterns: Vec<EntityPattern>,
}

/// Pattern for detecting entity-related queries.
#[derive(Debug, Clone)]
struct EntityPattern {
    /// Keywords that trigger this pattern
    keywords: Vec<&'static str>,
    /// Slot to query
    slot: &'static str,
    /// Whether the pattern looks for a specific value
    needs_value: bool,
}

impl QueryPlanner {
    /// Create a new query planner.
    #[must_use]
    pub fn new() -> Self {
        let mut planner = Self::default();
        planner.init_patterns();
        planner
    }

    fn init_patterns(&mut self) {
        // Location patterns
        self.entity_patterns.push(EntityPattern {
            keywords: vec![
                "who lives in",
                "people in",
                "users in",
                "from",
                "located in",
                "based in",
            ],
            slot: "location",
            needs_value: true,
        });

        // Employer/workplace patterns
        self.entity_patterns.push(EntityPattern {
            keywords: vec![
                "who works at",
                "employees of",
                "people at",
                "works for",
                "employed by",
            ],
            slot: "workplace", // OpenAI enrichment uses "workplace" not "employer"
            needs_value: true,
        });

        // Preference patterns
        self.entity_patterns.push(EntityPattern {
            keywords: vec![
                "who likes",
                "who loves",
                "fans of",
                "people who like",
                "people who love",
            ],
            slot: "preference",
            needs_value: true,
        });

        // Entity state patterns
        self.entity_patterns.push(EntityPattern {
            keywords: vec!["what is", "where does", "who is", "what does"],
            slot: "",
            needs_value: false,
        });
    }

    /// Analyze a query and produce an execution plan.
    #[must_use]
    pub fn plan(&self, query: &str, top_k: usize) -> QueryPlan {
        let query_lower = query.to_lowercase();

        // Try to detect relational patterns
        if let Some(pattern) = self.detect_pattern(&query_lower, query) {
            if pattern.triples.is_empty() {
                // No specific pattern found, use vector search
                QueryPlan::vector_only(Some(query.to_string()), None, top_k)
            } else {
                // Found relational pattern - use hybrid search
                QueryPlan::hybrid(pattern, Some(query.to_string()), None, top_k)
            }
        } else {
            // Default to vector-only search
            QueryPlan::vector_only(Some(query.to_string()), None, top_k)
        }
    }

    fn detect_pattern(&self, query_lower: &str, _original: &str) -> Option<GraphPattern> {
        let mut pattern = GraphPattern::new();

        for ep in &self.entity_patterns {
            for keyword in &ep.keywords {
                if query_lower.contains(keyword) {
                    // Extract the value after the keyword
                    if let Some(pos) = query_lower.find(keyword) {
                        let after = &query_lower[pos + keyword.len()..];
                        let value = extract_value(after);

                        if !value.is_empty() && ep.needs_value {
                            // Create pattern: ?entity :slot "value"
                            pattern.add(TriplePattern::any_slot_value("entity", ep.slot, &value));
                            return Some(pattern);
                        }
                    }
                }
            }
        }

        // Check for entity-specific queries like "alice's employer" or "what is alice's job"
        if let Some((entity, slot)) = extract_possessive_query(query_lower) {
            pattern.add(TriplePattern::entity_slot_any(&entity, &slot, "value"));
            return Some(pattern);
        }

        Some(pattern)
    }
}

/// Extract a value from text after a keyword.
fn extract_value(text: &str) -> String {
    let trimmed = text.trim();
    // Take words until we hit a common query continuation
    let stop_words = ["and", "or", "who", "what", "that", "?"];
    let mut words = Vec::new();

    for word in trimmed.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
        if stop_words.contains(&clean.to_lowercase().as_str()) {
            break;
        }
        if !clean.is_empty() {
            words.push(clean);
        }
        // Stop after a few words
        if words.len() >= 3 {
            break;
        }
    }

    words.join(" ")
}

/// Extract entity and slot from possessive queries like "alice's employer".
fn extract_possessive_query(query: &str) -> Option<(String, String)> {
    // Pattern: "X's Y" or "X's Y is"
    if let Some(pos) = query.find("'s ") {
        let entity = query[..pos].split_whitespace().last()?;
        let after = &query[pos + 3..];
        let slot = after.split_whitespace().next()?;

        // Map common slot aliases
        let slot = match slot {
            "job" | "work" | "employer" | "role" | "company" => "workplace",
            "home" | "city" | "address" => "location",
            "favorite" => "preference",
            "wife" | "husband" | "spouse" | "partner" => "spouse",
            other => other,
        };

        return Some((entity.to_string(), slot.to_string()));
    }
    None
}

/// Graph matcher that executes patterns against MemoryCards.
pub struct GraphMatcher<'a> {
    memvid: &'a Memvid,
}

impl<'a> GraphMatcher<'a> {
    /// Create a new graph matcher.
    pub fn new(memvid: &'a Memvid) -> Self {
        Self { memvid }
    }

    /// Execute a graph pattern and return matching results.
    pub fn execute(&self, pattern: &GraphPattern) -> Vec<GraphMatchResult> {
        let mut results = Vec::new();

        for triple in &pattern.triples {
            let matches = self.match_triple(triple);
            results.extend(matches);
        }

        // Deduplicate by entity
        let mut seen = HashSet::new();
        results.retain(|r| seen.insert(r.entity.clone()));

        results
    }

    fn match_triple(&self, triple: &TriplePattern) -> Vec<GraphMatchResult> {
        let mut results = Vec::new();

        match (&triple.subject, &triple.predicate, &triple.object) {
            // Pattern: ?entity :slot "value" - find entities with this slot value
            (
                PatternTerm::Variable(var),
                PatternTerm::Literal(slot),
                PatternTerm::Literal(value),
            ) => {
                // Iterate all entities and check for matching slot value
                for entity in self.memvid.memory_entities() {
                    let cards = self.memvid.get_entity_memories(&entity);
                    for card in cards {
                        if card.slot.to_lowercase() == *slot
                            && card.value.to_lowercase().contains(&value.to_lowercase())
                        {
                            let mut result = GraphMatchResult::new(
                                entity.clone(),
                                vec![card.source_frame_id],
                                1.0,
                            );
                            result.bind(var, entity.clone());
                            results.push(result);
                            break; // One match per entity
                        }
                    }
                }
            }

            // Pattern: "entity" :slot ?value - get entity's slot value
            (
                PatternTerm::Literal(entity),
                PatternTerm::Literal(slot),
                PatternTerm::Variable(var),
            ) => {
                if let Some(card) = self.memvid.get_current_memory(entity, slot) {
                    let mut result =
                        GraphMatchResult::new(entity.clone(), vec![card.source_frame_id], 1.0);
                    result.bind(var, card.value.clone());
                    results.push(result);
                }
            }

            // Pattern: "entity" :slot "value" - check if entity has this exact value
            (
                PatternTerm::Literal(entity),
                PatternTerm::Literal(slot),
                PatternTerm::Literal(value),
            ) => {
                if let Some(card) = self.memvid.get_current_memory(entity, slot) {
                    if card.value.to_lowercase().contains(&value.to_lowercase()) {
                        let result =
                            GraphMatchResult::new(entity.clone(), vec![card.source_frame_id], 1.0);
                        results.push(result);
                    }
                }
            }

            _ => {
                // Other patterns not yet implemented
            }
        }

        results
    }

    /// Get frame IDs from graph matches for use in vector search filtering.
    #[must_use]
    pub fn get_candidate_frames(&self, matches: &[GraphMatchResult]) -> Vec<FrameId> {
        let mut frame_ids: Vec<FrameId> = matches
            .iter()
            .flat_map(|m| m.frame_ids.iter().copied())
            .collect();
        frame_ids.sort_unstable();
        frame_ids.dedup();
        frame_ids
    }

    /// Get matched entities for context.
    #[must_use]
    pub fn get_matched_entities(&self, matches: &[GraphMatchResult]) -> HashMap<FrameId, String> {
        let mut map = HashMap::new();
        for m in matches {
            for &fid in &m.frame_ids {
                map.insert(fid, m.entity.clone());
            }
        }
        map
    }
}

/// Execute a hybrid search: graph filter + vector ranking.
pub fn hybrid_search(memvid: &mut Memvid, plan: &QueryPlan) -> Result<Vec<HybridSearchHit>> {
    match plan {
        QueryPlan::VectorOnly {
            query_text, top_k, ..
        } => {
            // Fall back to regular lexical search
            let query = query_text.as_deref().unwrap_or("");
            let request = SearchRequest {
                query: query.to_string(),
                top_k: *top_k,
                snippet_chars: 200,
                uri: None,
                frames: Vec::new(),
                scope: None,
                cursor: None,
                #[cfg(feature = "temporal_track")]
                temporal: None,
                as_of_frame: None,
                as_of_ts: None,
                no_sketch: false,
            };
            let response = memvid.search(request)?;
            Ok(response
                .hits
                .iter()
                .map(|h| {
                    let score = h.score.unwrap_or(0.0);
                    HybridSearchHit {
                        frame_id: h.frame_id,
                        score,
                        graph_score: 0.0,
                        vector_score: score,
                        matched_entity: None,
                        preview: Some(h.text.clone()),
                    }
                })
                .collect())
        }

        QueryPlan::GraphOnly { pattern, limit } => {
            let matcher = GraphMatcher::new(memvid);
            let matches = matcher.execute(pattern);

            Ok(matches
                .into_iter()
                .take(*limit)
                .map(|m| HybridSearchHit {
                    frame_id: m.frame_ids.first().copied().unwrap_or(0),
                    score: m.confidence,
                    graph_score: m.confidence,
                    vector_score: 0.0,
                    matched_entity: Some(m.entity),
                    preview: None,
                })
                .collect())
        }

        QueryPlan::Hybrid {
            graph_filter,
            query_text,
            top_k,
            ..
        } => {
            // Step 1: Execute graph pattern to get candidate frames
            let matcher = GraphMatcher::new(memvid);
            let matches = matcher.execute(graph_filter);
            let entity_map = matcher.get_matched_entities(&matches);
            let candidate_frames = matcher.get_candidate_frames(&matches);

            if candidate_frames.is_empty() {
                // No graph matches - fall back to lexical search
                let query = query_text.as_deref().unwrap_or("");
                let request = SearchRequest {
                    query: query.to_string(),
                    top_k: *top_k,
                    snippet_chars: 200,
                    uri: None,
                    frames: Vec::new(),
                    scope: None,
                    cursor: None,
                    #[cfg(feature = "temporal_track")]
                    temporal: None,
                    as_of_frame: None,
                    as_of_ts: None,
                    no_sketch: false,
                };
                let response = memvid.search(request)?;
                return Ok(response
                    .hits
                    .iter()
                    .map(|h| {
                        let score = h.score.unwrap_or(0.0);
                        HybridSearchHit {
                            frame_id: h.frame_id,
                            score,
                            graph_score: 0.0,
                            vector_score: score,
                            matched_entity: None,
                            preview: Some(h.text.clone()),
                        }
                    })
                    .collect());
            }

            // Step 2: Return graph matches directly with frame previews
            // Graph matching already found the relevant frames - return them
            let mut hybrid_hits: Vec<HybridSearchHit> = Vec::new();

            for &frame_id in &candidate_frames {
                let matched_entity = entity_map.get(&frame_id).cloned();

                // Get frame preview if possible
                let preview = memvid.frame_preview_by_id(frame_id).ok();

                hybrid_hits.push(HybridSearchHit {
                    frame_id,
                    score: 1.0, // Graph match score
                    graph_score: 1.0,
                    vector_score: 0.0,
                    matched_entity,
                    preview,
                });
            }

            Ok(hybrid_hits.into_iter().take(*top_k).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_planner_detects_location() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("who lives in San Francisco", 10);

        match plan {
            QueryPlan::Hybrid { graph_filter, .. } => {
                assert!(!graph_filter.is_empty());
                let triple = &graph_filter.triples[0];
                assert!(matches!(&triple.predicate, PatternTerm::Literal(s) if s == "location"));
            }
            _ => panic!("Expected hybrid plan for location query"),
        }
    }

    #[test]
    fn test_query_planner_detects_workplace() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("who works at Google", 10);

        match plan {
            QueryPlan::Hybrid { graph_filter, .. } => {
                assert!(!graph_filter.is_empty());
                let triple = &graph_filter.triples[0];
                assert!(matches!(&triple.predicate, PatternTerm::Literal(s) if s == "workplace"));
            }
            _ => panic!("Expected hybrid plan for workplace query"),
        }
    }

    #[test]
    fn test_query_planner_possessive() {
        let planner = QueryPlanner::new();
        let plan = planner.plan("what is alice's employer", 10);

        match plan {
            QueryPlan::Hybrid { graph_filter, .. } => {
                assert!(!graph_filter.is_empty());
                let triple = &graph_filter.triples[0];
                assert!(matches!(&triple.subject, PatternTerm::Literal(s) if s == "alice"));
            }
            _ => panic!("Expected hybrid plan for possessive query"),
        }
    }

    #[test]
    fn test_extract_value() {
        assert_eq!(extract_value("San Francisco and"), "San Francisco");
        assert_eq!(extract_value("Google who"), "Google");
        assert_eq!(extract_value("New York City"), "New York City");
    }

    #[test]
    fn test_extract_possessive() {
        assert_eq!(
            extract_possessive_query("what is alice's job"),
            Some(("alice".to_string(), "workplace".to_string()))
        );
        assert_eq!(
            extract_possessive_query("bob's location"),
            Some(("bob".to_string(), "location".to_string()))
        );
    }
}
