//! Tag suggestion and exploration utilities.

use std::cmp::Ordering;
use std::collections::HashMap;

use super::lifecycle::Memvid;
use crate::Result;
use crate::types::{FrameStatus, VecEmbedder};

/// Result of a tag suggestion query.
#[derive(Debug, Clone, PartialEq)]
pub struct TagSuggestion {
    /// The tag text
    pub tag: String,
    /// How many documents contain this tag
    pub count: usize,
    /// Match score (higher is better)
    pub score: f32,
}

impl Eq for TagSuggestion {}

impl PartialOrd for TagSuggestion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TagSuggestion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.count.cmp(&other.count))
            .then_with(|| self.tag.cmp(&other.tag))
    }
}

impl Memvid {
    /// Search for tags similar to the query string.
    ///
    /// This is useful for auto-completion or exploring the tag space.
    ///
    /// # Ranking Logic
    /// Results are sorted by:
    /// 1. Match type (Exact > StartsWith > Contains)
    /// 2. Frequency (High count > Low count)
    /// 3. Alphabetical
    ///
    /// # Example
    /// ```ignore
    /// let tags = memvid.suggest_tags("pro", 10);
    /// // Returns ["project", "production", "profile"]
    /// ```
    pub fn suggest_tags(&self, query: &str, limit: usize) -> Vec<TagSuggestion> {
        let query_lower = query.trim().to_lowercase();
        let mut tag_counts: HashMap<String, usize> = HashMap::new();

        // 1. Collect and count all tags from active frames
        // Since TOC is in memory, this is extremely fast even for 100k+ frames
        for frame in &self.toc.frames {
            if frame.status != FrameStatus::Active {
                continue;
            }
            for tag in &frame.tags {
                // Normalize to lowercase for counting, but we might want to preserve case later
                // For now, we count unique lowercase versions
                *tag_counts.entry(tag.to_string()).or_insert(0) += 1;
            }
        }

        // 2. Filter and Score
        let mut candidates: Vec<TagSuggestion> = tag_counts
            .into_iter()
            .filter_map(|(tag, count)| {
                let tag_lower = tag.to_lowercase();

                // Scoring logic
                let score = if tag_lower == query_lower {
                    3.0 // Exact match
                } else if tag_lower.starts_with(&query_lower) {
                    2.0 // Prefix match (Auto-complete)
                } else if tag_lower.contains(&query_lower) {
                    1.0 // Substring match
                } else {
                    0.0 // No match
                };

                if score > 0.0 {
                    Some(TagSuggestion { tag, count, score })
                } else {
                    None
                }
            })
            .collect();

        // 3. Sort
        candidates.sort_by(|a, b| b.cmp(a));

        // 4. Truncate
        candidates.truncate(limit);
        candidates
    }

    /// Search for tags semantically similar to the query using embeddings.
    ///
    /// This method:
    /// 1. Embeds the query string.
    /// 2. Embeds all unique tags (caching them in memory).
    /// 3. Computes cosine similarity between query and tags.
    /// 4. Returns top-k tags sorted by similarity.
    ///
    /// # Arguments
    /// * `query` - The search query (e.g., "coding").
    /// * `embedder` - The embedding model to use.
    /// * `limit` - Maximum number of suggestions to return.
    pub fn suggest_tags_semantic<E>(
        &mut self,
        query: &str,
        embedder: &E,
        limit: usize,
    ) -> Result<Vec<TagSuggestion>>
    where
        E: VecEmbedder + ?Sized,
    {
        // 1. Collect all unique tags and their counts
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        for frame in &self.toc.frames {
            if frame.status != FrameStatus::Active {
                continue;
            }
            for tag in &frame.tags {
                *tag_counts.entry(tag.to_string()).or_insert(0) += 1;
            }
        }

        if tag_counts.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Identify missing embeddings
        let missing_tags: Vec<String> = {
            let cache = self.tag_embeddings_cache.read().unwrap();
            tag_counts
                .keys()
                .filter(|tag| !cache.contains_key(*tag))
                .cloned()
                .collect()
        };

        // 3. Embed missing tags (batch processing)
        if !missing_tags.is_empty() {
            // Embed in batches of 32
            const BATCH_SIZE: usize = 32;
            for chunk in missing_tags.chunks(BATCH_SIZE) {
                let chunk_refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                let embeddings = embedder.embed_chunks(&chunk_refs)?;
                
                let mut cache = self.tag_embeddings_cache.write().unwrap();
                for (tag, embedding) in chunk.iter().zip(embeddings) {
                    cache.insert(tag.clone(), embedding);
                }
            }
        }

        // 4. Embed query
        let query_embedding = embedder.embed_query(query)?;

        // 5. Compute similarities
        let cache = self.tag_embeddings_cache.read().unwrap();
        let mut candidates: Vec<TagSuggestion> = tag_counts
            .into_iter()
            .filter_map(|(tag, count)| {
                if let Some(embedding) = cache.get(&tag) {
                    let score = cosine_similarity(&query_embedding, embedding);
                    Some(TagSuggestion { tag, count, score })
                } else {
                    None
                }
            })
            .collect();

        // 6. Sort by score descending
        candidates.sort_by(|a, b| b.cmp(a));

        // 7. Truncate
        candidates.truncate(limit);
        Ok(candidates)
    }

    /// Get all unique tags in the memory with their counts.
    pub fn all_tags(&self) -> Vec<(String, usize)> {
        let mut tag_counts: HashMap<String, usize> = HashMap::new();

        for frame in &self.toc.frames {
            if frame.status != FrameStatus::Active {
                continue;
            }
            for tag in &frame.tags {
                *tag_counts.entry(tag.to_string()).or_insert(0) += 1;
            }
        }

        let mut result: Vec<_> = tag_counts.into_iter().collect();
        // Sort by count descending
        result.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        result
    }
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
