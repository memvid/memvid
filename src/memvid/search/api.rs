#[cfg(feature = "lex")]
use crate::search::{EmbeddedLexSegment, TantivyEngine};
#[cfg(feature = "lex")]
use std::fs::{self, File};
#[cfg(feature = "lex")]
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(feature = "lex")]
use tempfile::TempDir;

use crate::memvid::lifecycle::Memvid;
use crate::types::{
    AdaptiveConfig, AdaptiveResult, AdaptiveStats, EmbeddingQualityStats, Frame, FrameId,
    FrameStatus, SearchHit, TimelineEntry, TimelineQuery, VecSegmentDescriptor,
    compute_embedding_quality, find_adaptive_cutoff,
};
use crate::{LexSearchHit, MemvidError, Result, VecSearchHit};

impl Memvid {
    pub fn enable_lex(&mut self) -> Result<()> {
        self.ensure_writable()?;
        if self.lex_enabled {
            // If index exists on disk but not in memory, load it
            #[cfg(feature = "lex")]
            if self.lex_index.is_none() && crate::memvid::lifecycle::has_lex_index(&self.toc) {
                self.load_lex_index_from_manifest()?;
            }
            return Ok(());
        }
        self.lex_enabled = true;
        self.toc.segment_catalog.lex_enabled = true;
        self.dirty = true;
        #[cfg(feature = "lex")]
        self.init_tantivy()?;

        // Create empty lex manifest so the flag persists across open/close
        if self.toc.indexes.lex.is_none() {
            // Use data_end as offset for empty manifest to avoid conflicts with header
            let empty_offset = self.data_end;
            // SHA256 hash of empty data
            let empty_checksum = *b"\xe3\xb0\xc4\x42\x98\xfc\x1c\x14\x9a\xfb\xf4\xc8\x99\x6f\xb9\x24\
                                    \x27\xae\x41\xe4\x64\x9b\x93\x4c\xa4\x95\x99\x1b\x78\x52\xb8\x55";
            self.toc.indexes.lex = Some(crate::types::LexIndexManifest {
                doc_count: 0,
                generation: 0,
                bytes_offset: empty_offset,
                bytes_length: 0,
                checksum: empty_checksum,
            });
        }

        self.commit()
    }

    pub fn search_lex(&mut self, query: &str, limit: usize) -> Result<Vec<LexSearchHit>> {
        if !self.lex_enabled {
            return Err(MemvidError::LexNotEnabled);
        }
        self.ensure_lex_index()?;
        let index = self.lex_index.as_ref().ok_or(MemvidError::LexNotEnabled)?;
        Ok(index.search(query, limit))
    }

    /// Human-friendly alias for [`Self::search_lex`].
    pub fn find(&mut self, query: &str, limit: usize) -> Result<Vec<LexSearchHit>> {
        self.search_lex(query, limit)
    }

    pub fn enable_vec(&mut self) -> Result<()> {
        self.ensure_writable()?;

        // Always set vec_enabled to true when explicitly requested,
        // regardless of compile-time feature flags
        self.vec_enabled = true;
        self.dirty = true;

        // Create empty vec manifest so the flag persists across open/close
        if self.toc.indexes.vec.is_none() {
            // Use data_end as offset for empty manifest to avoid conflicts with header
            let empty_offset = self.data_end;
            // SHA256 hash of empty data
            let empty_checksum = *b"\xe3\xb0\xc4\x42\x98\xfc\x1c\x14\x9a\xfb\xf4\xc8\x99\x6f\xb9\x24\
                                    \x27\xae\x41\xe4\x64\x9b\x93\x4c\xa4\x95\x99\x1b\x78\x52\xb8\x55";
            self.toc.indexes.vec = Some(crate::types::VecIndexManifest {
                vector_count: 0,
                dimension: 0,
                bytes_offset: empty_offset,
                bytes_length: 0,
                checksum: empty_checksum,
                compression_mode: self.vec_compression.clone(),
                model: self.vec_model.clone(),
            });
        }

        // No need to commit here - the manifest will be written during next commit/seal
        Ok(())
    }

    /// Set the expected embedding model for the vector index.
    ///
    /// If the index is already bound to a model (from a previous session or call),
    /// this validates that the requested model matches the existing one.
    /// If unbound, it binds the index to the new model.
    pub fn set_vec_model(&mut self, model: &str) -> Result<()> {
        if let Some(existing) = &self.vec_model {
            if existing != model {
                return Err(MemvidError::ModelMismatch {
                    expected: existing.clone(),
                    actual: model.to_string(),
                });
            }
        } else {
            self.vec_model = Some(model.to_string());
            // If manifest exists, update it to persist the binding
            if let Some(manifest) = self.toc.indexes.vec.as_mut() {
                manifest.model = Some(model.to_string());
                self.dirty = true;
            }
        }
        Ok(())
    }

    pub fn search_vec(&mut self, query: &[f32], limit: usize) -> Result<Vec<VecSearchHit>> {
        if !self.vec_enabled {
            return Err(MemvidError::VecNotEnabled);
        }
        let mut ensured_vec_index = false;
        let expected_dim = if let Some(dim) = self.effective_vec_index_dimension()? {
            dim
        } else {
            self.ensure_vec_index()?;
            ensured_vec_index = true;
            self.vec_index
                .as_ref()
                .and_then(|index| {
                    index
                        .entries()
                        .next()
                        .map(|(_, emb)| u32::try_from(emb.len()).unwrap_or(0))
                })
                .unwrap_or(0)
        };
        // Safe: embedding dimensions are small (< few thousands)
        #[allow(clippy::cast_possible_truncation)]
        if expected_dim > 0 && (query.len() as u32) != expected_dim {
            return Err(MemvidError::VecDimensionMismatch {
                expected: expected_dim,
                actual: query.len(),
            });
        }

        if !ensured_vec_index {
            self.ensure_vec_index()?;
        }
        let index = self.vec_index.as_ref().ok_or(MemvidError::VecNotEnabled)?;
        Ok(index.search(query, limit))
    }

    /// Enable CLIP visual embeddings index.
    ///
    /// CLIP allows semantic search across images using natural language queries.
    /// Unlike text vec embeddings (384/768/1536 dims), CLIP embeddings have
    /// fixed 512 dimensions (MobileCLIP-S2) and are stored in a separate index.
    pub fn enable_clip(&mut self) -> Result<()> {
        self.ensure_writable()?;

        self.clip_enabled = true;
        self.dirty = true;

        // Create empty clip manifest so the flag persists across open/close
        if self.toc.indexes.clip.is_none() {
            let empty_offset = self.data_end;
            let empty_checksum = *b"\xe3\xb0\xc4\x42\x98\xfc\x1c\x14\x9a\xfb\xf4\xc8\x99\x6f\xb9\x24\
                                    \x27\xae\x41\xe4\x64\x9b\x93\x4c\xa4\x95\x99\x1b\x78\x52\xb8\x55";
            self.toc.indexes.clip = Some(crate::clip::ClipIndexManifest {
                bytes_offset: empty_offset,
                bytes_length: 0,
                vector_count: 0,
                dimension: crate::clip::MOBILECLIP_DIMS,
                checksum: empty_checksum,
                model_name: "mobileclip-s2".to_string(),
            });
        }

        Ok(())
    }

    /// Add a CLIP embedding for a frame (legacy, no page info).
    ///
    /// This adds the visual embedding to the CLIP index for later semantic search.
    /// The frame must already exist. Use `ClipModel::encode_image()` to generate embeddings.
    pub fn add_clip_embedding(&mut self, frame_id: u64, embedding: Vec<f32>) -> Result<()> {
        self.add_clip_embedding_with_page(frame_id, None, embedding)
    }

    /// Add a CLIP embedding for a frame and optional page number.
    ///
    /// Page is 1-indexed when provided (PDF pages).
    pub fn add_clip_embedding_with_page(
        &mut self,
        frame_id: u64,
        page: Option<u32>,
        embedding: Vec<f32>,
    ) -> Result<()> {
        self.ensure_writable()?;
        if !self.clip_enabled {
            return Err(MemvidError::ClipNotEnabled);
        }

        // Initialize clip index if needed
        if self.clip_index.is_none() {
            self.clip_index = Some(crate::clip::ClipIndex::new());
        }

        // Add the document to the index
        if let Some(ref mut index) = self.clip_index {
            index.add_document(frame_id, page, embedding);
        }

        self.dirty = true;
        Ok(())
    }

    /// Search CLIP index with a pre-computed query embedding.
    ///
    /// Use `ClipModel::encode_text(query)` to generate the query embedding.
    pub fn search_clip(
        &mut self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<crate::clip::ClipSearchHit>> {
        tracing::debug!(
            "search_clip: clip_enabled={} query_len={} limit={}",
            self.clip_enabled,
            query.len(),
            limit
        );
        if !self.clip_enabled {
            tracing::debug!("search_clip: CLIP not enabled, returning error");
            return Err(MemvidError::ClipNotEnabled);
        }
        self.ensure_clip_index()?;
        let index = self
            .clip_index
            .as_ref()
            .ok_or(MemvidError::ClipNotEnabled)?;
        tracing::debug!("search_clip: clip_index has {} documents", index.len());
        let hits = index.search(query, limit);
        tracing::debug!("search_clip: returning {} hits", hits.len());
        Ok(hits)
    }

    /// Check if CLIP index is loaded, loading it if needed
    pub(crate) fn ensure_clip_index(&mut self) -> Result<()> {
        if self.clip_index.is_none() && self.toc.indexes.clip.is_some() {
            self.load_clip_index_from_manifest()?;
        }
        Ok(())
    }

    /// Perform pure vector search using a pre-computed query embedding.
    /// This searches the entire vector index directly, like Chroma does.
    pub fn vec_search_with_embedding(
        &mut self,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
        snippet_chars: usize,
        scope: Option<&str>,
    ) -> Result<crate::types::SearchResponse> {
        use super::helpers::{build_context, timestamp_to_rfc3339};
        use crate::types::{
            SearchEngineKind, SearchHit, SearchHitMetadata, SearchParams, SearchResponse,
        };
        use std::time::Instant;

        if !self.vec_enabled {
            return Err(MemvidError::VecNotEnabled);
        }

        // Validate embedding dimension BEFORE searching to prevent silent wrong results.
        // For segment-only memories, dimension may only be discoverable after loading segments.
        let mut ensured_vec_index = false;
        let expected_dim = if let Some(dim) = self.effective_vec_index_dimension()? {
            dim
        } else {
            self.ensure_vec_index()?;
            ensured_vec_index = true;
            self.vec_index
                .as_ref()
                .and_then(|index| {
                    index
                        .entries()
                        .next()
                        .map(|(_, emb)| u32::try_from(emb.len()).unwrap_or(0))
                })
                .unwrap_or(0)
        };
        // Safe: embedding dimensions are small
        #[allow(clippy::cast_possible_truncation)]
        if expected_dim > 0 && (query_embedding.len() as u32) != expected_dim {
            return Err(MemvidError::VecDimensionMismatch {
                expected: expected_dim,
                actual: query_embedding.len(),
            });
        }

        let start_time = Instant::now();

        // Ensure vector index is loaded
        if !ensured_vec_index {
            self.ensure_vec_index()?;
        }

        let vec_index = self.vec_index.as_ref().ok_or(MemvidError::VecNotEnabled)?;

        // Do pure vector search over entire index
        let vec_hits = vec_index.search(query_embedding, top_k * 2);

        if vec_hits.is_empty() {
            let elapsed_ms = start_time.elapsed().as_millis();
            return Ok(SearchResponse {
                query: query.to_string(),
                elapsed_ms,
                total_hits: 0,
                params: SearchParams {
                    top_k,
                    snippet_chars,
                    cursor: None,
                },
                hits: Vec::new(),
                context: build_context(&[]),
                next_cursor: None,
                engine: SearchEngineKind::Hybrid,
            });
        }

        // Convert VecSearchHit to SearchHit with full metadata
        let mut hits = Vec::new();
        let snippet_limit = snippet_chars.max(80);

        for vec_hit in vec_hits {
            // Apply scope filter if provided
            // Apply scope filter if provided
            let frame_idx = if let Ok(idx) = usize::try_from(vec_hit.frame_id) {
                idx
            } else {
                continue;
            };

            let frame = match self.toc.frames.get(frame_idx) {
                Some(f) => f.clone(),
                None => continue,
            };

            if let Some(scope_prefix) = scope {
                let default_uri = crate::default_uri(frame.id);
                let uri = frame.uri.as_ref().unwrap_or(&default_uri);
                if !uri.starts_with(scope_prefix) {
                    continue;
                }
            }

            // Get frame content for snippet
            let content = match self.frame_content(&frame) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let snippet: String = content.chars().take(snippet_limit).collect();
            let snippet_bytes = snippet.len();

            let uri = frame
                .uri
                .clone()
                .unwrap_or_else(|| crate::default_uri(frame.id));
            let title = frame
                .title
                .clone()
                .or_else(|| crate::infer_title_from_uri(&uri));

            // VecIndex returns distance (lower is better), convert back to similarity (higher is better)
            // distance = 1.0 - similarity, so similarity = 1.0 - distance
            let similarity_score = 1.0 - vec_hit.distance;

            let metadata = SearchHitMetadata {
                matches: 1,
                tags: frame.tags.clone(),
                labels: frame.labels.clone(),
                track: frame.track.clone(),
                created_at: timestamp_to_rfc3339(frame.timestamp),
                content_dates: frame.content_dates.clone(),
                entities: Vec::new(),
                extra_metadata: frame.extra_metadata.clone(),
                #[cfg(feature = "temporal_track")]
                temporal: None,
            };

            hits.push(SearchHit {
                rank: hits.len() + 1,
                frame_id: vec_hit.frame_id,
                uri,
                title,
                range: (0, snippet_bytes),
                text: snippet.clone(),
                matches: 1,
                chunk_range: Some((0, snippet_bytes)),
                chunk_text: Some(snippet),
                score: Some(similarity_score),
                metadata: Some(metadata),
            });

            if hits.len() >= top_k {
                break;
            }
        }

        let elapsed_ms = start_time.elapsed().as_millis();

        #[cfg(feature = "temporal_track")]
        super::helpers::attach_temporal_metadata(self, &mut hits)?;

        let context = build_context(&hits);

        Ok(SearchResponse {
            query: query.to_string(),
            elapsed_ms,
            total_hits: hits.len(),
            params: SearchParams {
                top_k,
                snippet_chars,
                cursor: None,
            },
            hits,
            context,
            next_cursor: None,
            engine: SearchEngineKind::Hybrid,
        })
    }

    /// Perform adaptive vector search that dynamically determines how many results to return.
    ///
    /// Unlike fixed `top_k` retrieval, adaptive search examines relevancy score distribution
    /// to include all relevant results while excluding noise. This is crucial when:
    /// - Answers span multiple chunks (missing relevant context)
    /// - Score distribution varies by query (some queries have many relevant matches)
    ///
    /// # Arguments
    /// * `query` - The search query string
    /// * `query_embedding` - Pre-computed embedding vector for the query
    /// * `config` - Adaptive retrieval configuration
    /// * `snippet_chars` - Maximum characters for result snippets
    /// * `scope` - Optional URI prefix filter
    ///
    /// # Example
    /// ```ignore
    /// let config = AdaptiveConfig::with_relative_threshold(0.6);
    /// let result = memvid.search_adaptive("query", &embedding, config, 200, None)?;
    /// println!("Returned {} of {} results", result.stats.returned, result.stats.total_considered);
    /// ```
    pub fn search_adaptive(
        &mut self,
        query: &str,
        query_embedding: &[f32],
        config: AdaptiveConfig,
        snippet_chars: usize,
        scope: Option<&str>,
    ) -> Result<AdaptiveResult<SearchHit>> {
        use std::time::Instant;

        if !config.enabled {
            // Fall back to standard search with max_results as top_k
            let response = self.vec_search_with_embedding(
                query,
                query_embedding,
                config.max_results,
                snippet_chars,
                scope,
            )?;
            return Ok(AdaptiveResult {
                results: response.hits,
                stats: AdaptiveStats {
                    total_considered: response.total_hits,
                    returned: response.total_hits,
                    cutoff_index: response.total_hits,
                    cutoff_score: None,
                    top_score: None,
                    cutoff_ratio: None,
                    triggered_by: "adaptive_disabled".to_string(),
                },
            });
        }

        let start_time = Instant::now();

        // Over-retrieve: get max_results to have enough candidates
        let response = self.vec_search_with_embedding(
            query,
            query_embedding,
            config.max_results,
            snippet_chars,
            scope,
        )?;

        if response.hits.is_empty() {
            return Ok(AdaptiveResult::empty());
        }

        // Extract scores for cutoff analysis
        let scores: Vec<f32> = response.hits.iter().filter_map(|hit| hit.score).collect();

        if scores.is_empty() {
            // No scores available, return all results
            return Ok(AdaptiveResult {
                results: response.hits,
                stats: AdaptiveStats {
                    total_considered: response.total_hits,
                    returned: response.total_hits,
                    cutoff_index: response.total_hits,
                    cutoff_score: None,
                    top_score: None,
                    cutoff_ratio: None,
                    triggered_by: "no_scores".to_string(),
                },
            });
        }

        // Find adaptive cutoff
        let (cutoff_index, triggered_by) = find_adaptive_cutoff(&scores, &config);

        // Apply cutoff
        let mut results: Vec<SearchHit> = response.hits.into_iter().take(cutoff_index).collect();

        // Update ranks after cutoff
        for (i, hit) in results.iter_mut().enumerate() {
            hit.rank = i + 1;
        }

        let top_score = scores.first().copied();
        let cutoff_score = if cutoff_index > 0 && cutoff_index <= scores.len() {
            Some(scores[cutoff_index.saturating_sub(1)])
        } else {
            None
        };
        let cutoff_ratio = match (top_score, cutoff_score) {
            (Some(top), Some(cut)) if top > f32::EPSILON => Some(cut / top),
            _ => None,
        };

        let elapsed = start_time.elapsed();
        tracing::debug!(
            "adaptive search: {} -> {} results in {:?} ({})",
            scores.len(),
            results.len(),
            elapsed,
            triggered_by
        );

        Ok(AdaptiveResult {
            results,
            stats: AdaptiveStats {
                total_considered: scores.len(),
                returned: cutoff_index,
                cutoff_index,
                cutoff_score,
                top_score,
                cutoff_ratio,
                triggered_by,
            },
        })
    }

    /// Compute embedding quality statistics for the vector index.
    ///
    /// This analyzes the distribution of embeddings to provide insights about:
    /// - How similar/diverse the embeddings are
    /// - Recommended adaptive retrieval thresholds
    /// - Overall embedding quality rating
    ///
    /// Returns `None` if vector index is not enabled or empty.
    pub fn embedding_quality(&mut self) -> Result<Option<EmbeddingQualityStats>> {
        if !self.vec_enabled {
            return Ok(None);
        }

        self.ensure_vec_index()?;

        let vec_index = match &self.vec_index {
            Some(index) => index,
            None => return Ok(None),
        };

        // Collect embeddings from the index
        let embeddings: Vec<(u64, Vec<f32>)> = vec_index
            .entries()
            .map(|(frame_id, embedding)| (frame_id, embedding.to_vec()))
            .collect();

        if embeddings.is_empty() {
            return Ok(None);
        }

        Ok(Some(compute_embedding_quality(&embeddings)))
    }

    /// Get frame IDs filtered by Replay parameters (`as_of_frame` or `as_of_ts`).
    /// Used for time-travel memory views.
    pub(crate) fn get_replay_frame_ids(
        &self,
        request: &crate::types::SearchRequest,
    ) -> Result<Vec<FrameId>> {
        let frames = &self.toc.frames;
        let mut matching_ids: Vec<FrameId> = Vec::new();

        for frame in frames {
            if frame.status != FrameStatus::Active {
                continue;
            }

            // Check as_of_frame filter
            if let Some(cutoff_frame) = request.as_of_frame {
                if frame.id > cutoff_frame {
                    continue;
                }
            }

            // Check as_of_ts filter
            if let Some(cutoff_ts) = request.as_of_ts {
                if frame.timestamp > cutoff_ts {
                    continue;
                }
            }

            matching_ids.push(frame.id);
        }

        Ok(matching_ids)
    }

    pub fn timeline(&mut self, query: TimelineQuery) -> Result<Vec<TimelineEntry>> {
        let TimelineQuery {
            limit,
            since,
            until,
            reverse,
            #[cfg(feature = "temporal_track")]
            temporal,
        } = query;

        #[cfg(feature = "temporal_track")]
        {
            crate::memvid::timeline::build_timeline(
                self,
                limit,
                since,
                until,
                reverse,
                temporal.as_ref(),
            )
        }
        #[cfg(not(feature = "temporal_track"))]
        {
            crate::memvid::timeline::build_timeline(self, limit, since, until, reverse)
        }
    }
}

#[cfg(feature = "lex")]
impl Memvid {
    #[allow(dead_code)]
    fn materialize_tantivy_segments(&mut self, segments: &[EmbeddedLexSegment]) -> Result<TempDir> {
        let dir = TempDir::new().map_err(|err| MemvidError::Tantivy {
            reason: format!("failed to allocate Tantivy work directory: {err}"),
        })?;
        if segments.is_empty() {
            return Ok(dir);
        }

        let mut file_len =
            self.file
                .metadata()
                .map(|meta| meta.len())
                .map_err(|err| MemvidError::Tantivy {
                    reason: format!("failed to inspect memvid file metadata: {err}"),
                })?;
        let mut data_limit = self.header.footer_offset;
        let mut buffer = vec![0u8; 64 * 1024];
        let cursor = self.file.stream_position()?;
        for segment in segments {
            let dest = dir.path().join(&segment.path);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(|err| MemvidError::Tantivy {
                    reason: format!(
                        "failed to prepare Tantivy directory {}: {}",
                        parent.display(),
                        err
                    ),
                })?;
            }
            let mut writer = File::create(&dest).map_err(|err| MemvidError::Tantivy {
                reason: format!(
                    "failed to materialize Tantivy segment {}: {}",
                    dest.display(),
                    err
                ),
            })?;
            if segment.bytes_length == 0 {
                continue;
            }
            let end = segment
                .bytes_offset
                .checked_add(segment.bytes_length)
                .ok_or_else(|| MemvidError::Tantivy {
                    reason: format!(
                        "embedded segment {} length overflow (offset {}, length {})",
                        segment.path, segment.bytes_offset, segment.bytes_length
                    ),
                })?;
            if end > file_len || end > data_limit {
                if self.align_footer_with_catalog()? {
                    file_len = self.file.metadata().map(|meta| meta.len()).map_err(|err| {
                        MemvidError::Tantivy {
                            reason: format!("failed to refresh memvid file metadata: {err}"),
                        }
                    })?;
                    data_limit = self.header.footer_offset;
                }
                if end > file_len || end > data_limit {
                    return Err(MemvidError::Tantivy {
                        reason: format!(
                            "embedded segment {} out of bounds (offset {} length {} data_limit {} file_len {})",
                            segment.path,
                            segment.bytes_offset,
                            segment.bytes_length,
                            data_limit,
                            file_len
                        ),
                    });
                }
            }
            self.file.seek(SeekFrom::Start(segment.bytes_offset))?;
            let mut remaining = segment.bytes_length;
            while remaining > 0 {
                // Safe: chunk is at most buffer.len() which is usize
                #[allow(clippy::cast_possible_truncation)]
                let chunk = remaining.min(buffer.len() as u64) as usize;
                if let Err(err) = self.file.read_exact(&mut buffer[..chunk]) {
                    return Err(MemvidError::Tantivy {
                        reason: format!(
                            "failed to read embedded segment {} (offset {}, remaining {}, chunk {}): {}",
                            segment.path, segment.bytes_offset, remaining, chunk, err
                        ),
                    });
                }
                writer.write_all(&buffer[..chunk])?;
                remaining -= chunk as u64;
            }
        }
        self.file.seek(SeekFrom::Start(cursor))?;
        Ok(dir)
    }

    pub(crate) fn init_tantivy(&mut self) -> Result<()> {
        if !self.lex_enabled {
            self.tantivy = None;
            self.tantivy_dirty = false;
            return Ok(());
        }

        let segments = if self.toc.segment_catalog.tantivy_segments.is_empty() {
            match self.lex_storage.read() {
                Ok(storage) => {
                    if storage.is_empty() {
                        None
                    } else {
                        Some(storage.segments().cloned().collect::<Vec<_>>())
                    }
                }
                Err(_) => None,
            }
        } else {
            Some(
                self.toc
                    .segment_catalog
                    .tantivy_segments
                    .iter()
                    .map(|descriptor| EmbeddedLexSegment {
                        path: descriptor.path.clone(),
                        bytes_offset: descriptor.common.bytes_offset,
                        bytes_length: descriptor.common.bytes_length,
                        checksum: descriptor.common.checksum,
                    })
                    .collect::<Vec<_>>(),
            )
        };

        let mut engine = match segments {
            Some(segments) => {
                match self
                    .materialize_tantivy_segments(&segments)
                    .and_then(TantivyEngine::open_from_dir)
                {
                    Ok(engine) => engine,
                    Err(err) => {
                        tracing::debug!(
                            "failed to open embedded Tantivy index: {}, rebuilding",
                            err
                        );
                        TantivyEngine::create()?
                    }
                }
            }
            None => TantivyEngine::create()?,
        };

        // Use consolidated helper for expected doc count
        let expected_docs = self
            .lex_storage
            .read()
            .ok()
            .and_then(|storage| crate::memvid::lifecycle::lex_doc_count(&self.toc, &storage));

        let mut rebuilt = false;
        let actual_docs = engine.num_docs();

        let has_tantivy_segments = !self.toc.segment_catalog.tantivy_segments.is_empty();
        let needs_rebuild = if has_tantivy_segments {
            // Trust existing Tantivy segments, don't rebuild
            false
        } else {
            expected_docs != Some(actual_docs)
        };

        if needs_rebuild {
            if let Some(expected) = expected_docs {
                if actual_docs != 0 || expected != 0 {
                    tracing::debug!(
                        "rebuilding Tantivy index: expected {} docs, found {}",
                        expected,
                        actual_docs
                    );
                }
            }
            rebuilt = self.rebuild_tantivy_engine(&mut engine)?;
        }

        self.tantivy_dirty = rebuilt;
        self.tantivy = Some(engine);

        // This handles files created before the segment-based lex_enabled check was added
        self.lex_enabled = true;

        Ok(())
    }

    #[must_use]
    pub fn vec_segment_descriptor(&self, segment_id: u64) -> Option<VecSegmentDescriptor> {
        self.toc
            .segment_catalog
            .vec_segments
            .iter()
            .find(|descriptor| descriptor.common.segment_id == segment_id)
            .cloned()
    }

    pub fn read_vec_segment(
        &mut self,
        segment_id: u64,
    ) -> Result<Option<(VecSegmentDescriptor, Vec<u8>)>> {
        let Some(descriptor) = self.vec_segment_descriptor(segment_id) else {
            return Ok(None);
        };
        let bytes = self.read_range(
            descriptor.common.bytes_offset,
            descriptor.common.bytes_length,
        )?;
        Ok(Some((descriptor, bytes)))
    }
}

/// Default maximum payload size for text indexing (256 MiB)
/// Can be overridden via `MEMVID_MAX_INDEX_PAYLOAD` environment variable
pub const DEFAULT_MAX_INDEX_PAYLOAD: u64 = 256 * 1024 * 1024;

/// Get the maximum indexable payload size from environment or use default
#[must_use]
pub fn max_index_payload() -> u64 {
    std::env::var("MEMVID_MAX_INDEX_PAYLOAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_INDEX_PAYLOAD)
}

/// Check if a MIME type represents text-based content that should be indexed
#[must_use]
pub fn is_text_indexable_mime(mime: &str) -> bool {
    let mime_lower = mime.to_lowercase();

    // Text types
    if mime_lower.starts_with("text/") {
        return true;
    }

    // Application types that contain text
    let text_application_types = [
        "application/json",
        "application/xml",
        "application/xhtml+xml",
        "application/javascript",
        "application/typescript",
        "application/x-javascript",
        "application/ecmascript",
        "application/pdf",
        "application/rtf",
        "application/x-yaml",
        "application/yaml",
        "application/toml",
        "application/x-sh",
        "application/x-python",
        "application/sql",
        "application/graphql",
    ];

    if text_application_types.iter().any(|&t| mime_lower == t) {
        return true;
    }

    // Document types (Office, etc.)
    if mime_lower.contains("document")
        || mime_lower.contains("spreadsheet")
        || mime_lower.contains("presentation")
        || mime_lower.contains("wordprocessing")
        || mime_lower.contains("opendocument")
    {
        return true;
    }

    // Common text file extensions in MIME
    if mime_lower.contains("+xml") || mime_lower.contains("+json") {
        return true;
    }

    false
}

/// Check if a frame should be indexed for text search
#[must_use]
pub fn is_frame_text_indexable(frame: &crate::types::Frame) -> bool {
    // Must be active
    if frame.status != crate::types::FrameStatus::Active {
        return false;
    }

    // Get MIME type from metadata
    let mime = frame
        .metadata
        .as_ref()
        .and_then(|m| m.mime.as_deref())
        .unwrap_or("application/octet-stream");

    // Skip binary content types entirely (videos, images, audio)
    if !is_text_indexable_mime(mime) {
        return false;
    }

    // Check payload size limit for text content
    let max_payload = max_index_payload();
    if frame.payload_length > max_payload {
        return false;
    }

    // Must have non-empty search text
    frame
        .search_text
        .as_ref()
        .is_some_and(|t| !t.trim().is_empty())
}

#[cfg(feature = "lex")]
impl Memvid {
    pub(crate) fn rebuild_tantivy_engine(&mut self, engine: &mut TantivyEngine) -> Result<bool> {
        let mut prepared_docs: Vec<(Frame, String)> = Vec::new();
        let frames = self.toc.frames.clone();
        let active_frames: Vec<_> = frames
            .into_iter()
            .filter(|frame| frame.status == FrameStatus::Active)
            .collect();

        let max_payload = max_index_payload();

        for frame in active_frames {
            // Check if frame has explicit search_text first - if so, use it directly
            // This handles frames created via put_bytes() or other APIs that set search_text
            // but don't have text-indexable MIME types
            if let Some(search_text) = frame.search_text.clone() {
                if !search_text.trim().is_empty() {
                    prepared_docs.push((frame, search_text));
                    continue;
                }
            }

            // Get MIME type from metadata
            let mime = frame
                .metadata
                .as_ref()
                .and_then(|m| m.mime.as_deref())
                .unwrap_or("application/octet-stream");

            // Skip binary content types (videos, images, audio, etc.)
            if !is_text_indexable_mime(mime) {
                tracing::debug!(
                    "skipping frame {} - binary content type: {} (not text-indexable)",
                    frame.id,
                    mime
                );
                continue;
            }

            // Check payload size limit for text content
            if frame.payload_length > max_payload {
                tracing::debug!(
                    "skipping frame {} - payload {} exceeds max indexable size {} (MEMVID_MAX_INDEX_PAYLOAD)",
                    frame.id,
                    frame.payload_length,
                    max_payload
                );
                continue;
            }

            let text = self.frame_search_text(&frame)?;
            if text.trim().is_empty() {
                continue;
            }
            prepared_docs.push((frame, text));
        }

        if prepared_docs.is_empty() {
            engine.reset()?;
            engine.commit()?;
            return Ok(true);
        }

        engine.reset()?;
        for (frame, text) in &prepared_docs {
            engine.add_frame(frame, text)?;
        }
        engine.commit()?;
        Ok(true)
    }
}
