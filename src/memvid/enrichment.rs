//! Enrichment methods for Memvid.
//!
//! Provides background enrichment worker integration for progressive ingestion:
//! - Start/stop background worker
//! - Process enrichment queue
//! - Full text re-extraction for skim frames
//! - Embedding generation

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::enrichment_worker::{
    EmbeddingBatcher, EnrichmentWorkerConfig, EnrichmentWorkerHandle, EnrichmentWorkerStats,
    TaskResult,
};
use crate::error::Result;
use crate::extract_budgeted::ExtractionBudget;
use crate::types::{EnrichmentState, EnrichmentTask, FrameId, FrameStatus, VecEmbedder};
use crate::vec::VecIndexBuilder;

use super::Memvid;

/// Handle for the background enrichment worker thread.
pub struct EnrichmentHandle {
    /// Control handle for the worker.
    pub handle: EnrichmentWorkerHandle,
    /// Thread join handle.
    thread: Option<JoinHandle<()>>,
}

impl EnrichmentHandle {
    /// Stop the worker and wait for it to finish.
    #[must_use]
    pub fn stop_and_wait(mut self) -> EnrichmentWorkerStats {
        self.handle.stop();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        self.handle.stats()
    }

    /// Check if worker is still running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.handle.is_running()
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> EnrichmentWorkerStats {
        self.handle.stats()
    }

    /// Signal the worker to stop (non-blocking).
    pub fn stop(&self) {
        self.handle.stop();
    }
}

/// Start a background enrichment worker thread.
///
/// The worker processes frames from the enrichment queue asynchronously:
/// - Re-extracts full text for skim extractions
/// - Updates Tantivy index with enriched content
/// - Marks frames as Enriched when complete
///
/// # Arguments
/// * `memvid` - Arc-wrapped Memvid instance for thread-safe access
/// * `config` - Optional worker configuration (uses defaults if None)
///
/// # Returns
/// An `EnrichmentHandle` to control and monitor the worker.
///
/// # Example
/// ```ignore
/// use std::sync::{Arc, Mutex};
/// let memvid = Arc::new(Mutex::new(Memvid::open("memory.mv2")?));
/// let handle = start_enrichment_worker(Arc::clone(&memvid), None);
///
/// // Do other work while enrichment runs in background...
///
/// // When done, stop and wait
/// let stats = handle.stop_and_wait();
/// println!("Processed {} frames", stats.frames_processed);
/// ```
pub fn start_enrichment_worker(
    memvid: Arc<Mutex<Memvid>>,
    config: Option<EnrichmentWorkerConfig>,
) -> EnrichmentHandle {
    let config = config.unwrap_or_default();
    let handle = EnrichmentWorkerHandle::new();
    let worker_handle = handle.clone_handle();

    let memvid_clone = Arc::clone(&memvid);
    let config_clone = config.clone();

    let thread = std::thread::spawn(move || {
        crate::enrichment_worker::run_worker_loop(
            &worker_handle,
            &config_clone,
            // get_next_task
            || {
                let mv = memvid_clone.lock().ok()?;
                mv.next_enrichment_task()
            },
            // process_task
            |task| {
                let mut mv = match memvid_clone.lock() {
                    Ok(mv) => mv,
                    Err(_) => {
                        return TaskResult {
                            frame_id: task.frame_id,
                            re_extracted: false,
                            embeddings_generated: 0,
                            elapsed_ms: 0,
                            error: Some("Failed to acquire lock".to_string()),
                        };
                    }
                };
                mv.process_enrichment_task(task)
            },
            // mark_complete
            |frame_id| {
                if let Ok(mut mv) = memvid_clone.lock() {
                    mv.complete_enrichment_task(frame_id);
                }
            },
            // checkpoint
            || {
                if let Ok(mut mv) = memvid_clone.lock() {
                    if let Err(err) = mv.commit() {
                        tracing::warn!(?err, "enrichment checkpoint commit failed");
                    }
                }
            },
        );
    });

    EnrichmentHandle {
        handle,
        thread: Some(thread),
    }
}

/// Start a background enrichment worker with embedding generation.
///
/// Similar to `start_enrichment_worker` but also generates embeddings
/// for frames that need them.
///
/// # Arguments
/// * `memvid` - Arc-wrapped Memvid instance for thread-safe access
/// * `embedder` - The embedding model to use
/// * `config` - Optional worker configuration (uses defaults if None)
///
/// # Type Parameters
/// * `E` - Embedder type implementing `VecEmbedder + Send + 'static`
pub fn start_enrichment_worker_with_embeddings<E>(
    memvid: Arc<Mutex<Memvid>>,
    embedder: E,
    config: Option<EnrichmentWorkerConfig>,
) -> EnrichmentHandle
where
    E: VecEmbedder + Send + 'static,
{
    let config = config.unwrap_or_default();
    let handle = EnrichmentWorkerHandle::new();
    let worker_handle = handle.clone_handle();
    let batch_size = config.embedding_batch_size;

    let thread = std::thread::spawn(move || {
        worker_handle.set_running(true);
        tracing::info!("enrichment worker with embeddings started");

        // Process all enrichment with embeddings
        match memvid.lock() {
            Ok(mut mv) => {
                match mv.process_enrichment_with_embeddings(embedder, batch_size) {
                    Ok((frames, embeddings)) => {
                        // Update stats
                        for _ in 0..frames {
                            worker_handle.inc_frames_processed();
                        }
                        worker_handle.inc_embeddings(embeddings as u64);

                        // Commit changes
                        if let Err(err) = mv.commit() {
                            tracing::warn!(?err, "final commit failed");
                            worker_handle.inc_errors();
                        }
                    }
                    Err(err) => {
                        tracing::error!(?err, "enrichment with embeddings failed");
                        worker_handle.inc_errors();
                    }
                }
            }
            Err(err) => {
                tracing::error!(?err, "failed to acquire lock for enrichment");
                worker_handle.inc_errors();
            }
        }

        worker_handle.set_running(false);
        tracing::info!(
            frames_processed = worker_handle.stats().frames_processed,
            embeddings_generated = worker_handle.stats().embeddings_generated,
            "enrichment worker with embeddings stopped"
        );
    });

    EnrichmentHandle {
        handle,
        thread: Some(thread),
    }
}

impl Memvid {
    /// Get the number of frames pending enrichment.
    #[must_use]
    pub fn enrichment_queue_len(&self) -> usize {
        self.toc.enrichment_queue.len()
    }

    /// Check if any frames need enrichment.
    #[must_use]
    pub fn has_pending_enrichment(&self) -> bool {
        !self.toc.enrichment_queue.is_empty()
    }

    /// Get the next task from the enrichment queue.
    #[must_use]
    pub fn next_enrichment_task(&self) -> Option<EnrichmentTask> {
        self.toc.enrichment_queue.tasks.first().cloned()
    }

    /// Mark an enrichment task as complete.
    pub fn complete_enrichment_task(&mut self, frame_id: FrameId) {
        self.toc.enrichment_queue.remove(frame_id);
        self.dirty = true;
    }

    /// Read frame data needed for enrichment.
    ///
    /// Returns (`search_text`, `is_skim`, `needs_embedding`) if frame exists.
    #[must_use]
    pub fn read_frame_for_enrichment(&self, frame_id: FrameId) -> Option<(String, bool, bool)> {
        let frame = self
            .toc
            .frames
            .iter()
            .find(|f| f.id == frame_id && f.status == FrameStatus::Active)?;

        let search_text = frame.search_text.clone().unwrap_or_default();

        // Check if this is a skim extraction by looking at extra_metadata
        let is_skim = frame
            .extra_metadata
            .get("skim")
            .is_some_and(|v| v == "true");

        // Check if embeddings are needed
        let needs_embedding = frame.enrichment_state == EnrichmentState::Searchable;

        Some((search_text, is_skim, needs_embedding))
    }

    /// Perform full text extraction for a frame.
    ///
    /// This re-extracts the full text from the frame's payload without time budget.
    pub fn extract_full_text(&mut self, frame_id: FrameId) -> Result<String> {
        // Clone the frame to avoid borrow conflicts
        let frame = self
            .toc
            .frames
            .iter()
            .find(|f| f.id == frame_id && f.status == FrameStatus::Active)
            .cloned()
            .ok_or(crate::MemvidError::FrameNotFound { frame_id })?;

        // Read the payload
        let payload = self.read_frame_payload_bytes(&frame)?;

        // Extract with no time budget
        let mime_hint = frame.metadata.as_ref().and_then(|m| m.mime.as_deref());
        let uri_hint = frame.uri.as_deref();
        let budget = ExtractionBudget::unlimited();

        match crate::extract_budgeted::extract_with_budget(&payload, mime_hint, uri_hint, budget) {
            Ok(result) => Ok(result.text),
            Err(_) => {
                // Fall back to simple text extraction
                // If budgeted extraction fails, return any search text we already have
                Ok(frame.search_text.clone().unwrap_or_default())
            }
        }
    }

    /// Update the Tantivy index with enriched content.
    #[cfg(feature = "lex")]
    pub fn update_tantivy_for_enrichment(&mut self, frame_id: FrameId, text: &str) -> Result<()> {
        let tantivy = match self.tantivy.as_mut() {
            Some(t) => t,
            None => return Ok(()), // No Tantivy engine, nothing to update
        };

        // Find the frame
        let frame = self
            .toc
            .frames
            .iter()
            .find(|f| f.id == frame_id && f.status == FrameStatus::Active)
            .ok_or(crate::MemvidError::FrameNotFound { frame_id })?
            .clone();

        // Delete old document
        tantivy.delete_frame(frame_id)?;

        // Add updated document
        tantivy.add_frame(&frame, text)?;

        // Soft commit for immediate searchability
        tantivy.soft_commit()?;

        self.tantivy_dirty = true;
        Ok(())
    }

    #[cfg(not(feature = "lex"))]
    pub fn update_tantivy_for_enrichment(&mut self, _frame_id: FrameId, _text: &str) -> Result<()> {
        Ok(())
    }

    /// Update frame's enrichment state.
    pub fn mark_frame_enriched(&mut self, frame_id: FrameId) {
        if let Some(frame) = self
            .toc
            .frames
            .iter_mut()
            .find(|f| f.id == frame_id && f.status == FrameStatus::Active)
        {
            frame.enrichment_state = EnrichmentState::Enriched;
            self.dirty = true;
        }
    }

    /// Process a single enrichment task synchronously.
    ///
    /// This is useful for testing or when you don't want background processing.
    pub fn process_enrichment_task(&mut self, task: &EnrichmentTask) -> TaskResult {
        // Read frame data
        let frame_data = self.read_frame_for_enrichment(task.frame_id);

        // Process with closures that capture self
        let (search_text, is_skim, _needs_embedding) = match frame_data {
            Some(data) => data,
            None => {
                return TaskResult {
                    frame_id: task.frame_id,
                    re_extracted: false,
                    embeddings_generated: 0,
                    elapsed_ms: 0,
                    error: Some("Frame not found".to_string()),
                };
            }
        };

        let start = std::time::Instant::now();
        let mut result = TaskResult {
            frame_id: task.frame_id,
            re_extracted: false,
            embeddings_generated: 0,
            elapsed_ms: 0,
            error: None,
        };

        // Re-extract if this was a skim
        let final_text = if is_skim {
            match self.extract_full_text(task.frame_id) {
                Ok(full_text) => {
                    result.re_extracted = true;
                    full_text
                }
                Err(err) => {
                    tracing::warn!(
                        frame_id = task.frame_id,
                        ?err,
                        "re-extraction failed, using skim text"
                    );
                    search_text
                }
            }
        } else {
            search_text
        };

        // Update Tantivy index
        if let Err(err) = self.update_tantivy_for_enrichment(task.frame_id, &final_text) {
            result.error = Some(format!("Index update failed: {err}"));
        }

        // Mark frame as enriched
        self.mark_frame_enriched(task.frame_id);

        result.elapsed_ms = start.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
        result
    }

    /// Process all pending enrichment tasks synchronously.
    ///
    /// Returns the number of tasks processed.
    pub fn process_all_enrichment(&mut self) -> usize {
        let mut processed = 0;

        while let Some(task) = self.next_enrichment_task() {
            let result = self.process_enrichment_task(&task);
            self.complete_enrichment_task(task.frame_id);

            if result.error.is_some() {
                tracing::warn!(
                    frame_id = task.frame_id,
                    error = ?result.error,
                    "enrichment task failed"
                );
            } else {
                tracing::debug!(
                    frame_id = task.frame_id,
                    re_extracted = result.re_extracted,
                    elapsed_ms = result.elapsed_ms,
                    "enrichment task complete"
                );
            }

            processed += 1;
        }

        processed
    }

    /// Get enrichment statistics.
    #[must_use]
    pub fn enrichment_stats(&self) -> EnrichmentStats {
        let total_frames = self
            .toc
            .frames
            .iter()
            .filter(|f| f.status == FrameStatus::Active)
            .count();
        let enriched_frames = self
            .toc
            .frames
            .iter()
            .filter(|f| {
                f.status == FrameStatus::Active && f.enrichment_state == EnrichmentState::Enriched
            })
            .count();
        let pending_frames = self.enrichment_queue_len();

        EnrichmentStats {
            total_frames,
            enriched_frames,
            pending_frames,
            searchable_only: total_frames.saturating_sub(enriched_frames),
        }
    }

    /// Add embeddings to the vector index.
    ///
    /// This method adds new embeddings for frames that were enriched.
    /// It rebuilds the vector index to include the new embeddings.
    pub fn add_embeddings(&mut self, embeddings: Vec<(FrameId, Vec<f32>)>) -> Result<usize> {
        if embeddings.is_empty() {
            return Ok(0);
        }

        let count = embeddings.len();

        // Build new vector index with existing + new embeddings
        let mut builder = VecIndexBuilder::new();

        // Add existing embeddings from current index
        if let Some(ref vec_index) = self.vec_index {
            for (frame_id, embedding) in vec_index.entries() {
                // Skip if we're replacing this frame's embedding
                if !embeddings.iter().any(|(id, _)| *id == frame_id) {
                    builder.add_document(frame_id, embedding.to_vec());
                }
            }
        }

        // Add new embeddings
        for (frame_id, embedding) in embeddings {
            builder.add_document(frame_id, embedding);
        }

        // Finish building the index
        let artifact = builder.finish()?;
        if artifact.vector_count == 0 {
            return Ok(0);
        }

        // Decode and store the new index
        let new_index = crate::vec::VecIndex::decode(&artifact.bytes)?;
        self.vec_index = Some(new_index);

        // Update TOC with new manifest
        self.toc.indexes.vec = Some(crate::types::VecIndexManifest {
            vector_count: artifact.vector_count,
            dimension: artifact.dimension,
            bytes_offset: 0, // Will be set during commit
            bytes_length: artifact.bytes.len() as u64,
            checksum: artifact.checksum,
            compression_mode: crate::types::VectorCompression::None,
            model: self.vec_model.clone(),
        });

        self.dirty = true;
        self.vec_enabled = true;

        tracing::debug!(
            count,
            total_vectors = artifact.vector_count,
            dimension = artifact.dimension,
            "added embeddings to vector index"
        );

        Ok(count)
    }

    /// Process enrichment with embeddings using a batched embedder.
    ///
    /// This method processes the enrichment queue with embedding generation:
    /// 1. Re-extracts full text for skim frames
    /// 2. Generates embeddings in batches
    /// 3. Updates indexes
    /// 4. Marks frames as enriched
    ///
    /// Returns (`frames_processed`, `embeddings_generated`).
    pub fn process_enrichment_with_embeddings<E: VecEmbedder>(
        &mut self,
        embedder: E,
        batch_size: usize,
    ) -> Result<(usize, usize)> {
        let mut batcher = EmbeddingBatcher::new(embedder, batch_size);
        let mut frames_processed = 0;
        let mut embeddings_generated = 0;

        // Collect all tasks first to avoid borrow conflicts
        let tasks: Vec<_> = self.toc.enrichment_queue.tasks.clone();

        for task in tasks {
            // Read frame data
            let frame_data = self.read_frame_for_enrichment(task.frame_id);
            let (search_text, is_skim, needs_embedding) = match frame_data {
                Some(data) => data,
                None => continue, // Frame not found, skip
            };

            // Re-extract if this was a skim
            let final_text = if is_skim {
                match self.extract_full_text(task.frame_id) {
                    Ok(full_text) => full_text,
                    Err(err) => {
                        tracing::warn!(frame_id = task.frame_id, ?err, "re-extraction failed");
                        search_text
                    }
                }
            } else {
                search_text
            };

            // Update Tantivy index with enriched text
            if let Err(err) = self.update_tantivy_for_enrichment(task.frame_id, &final_text) {
                tracing::warn!(frame_id = task.frame_id, ?err, "tantivy update failed");
            }

            // Queue for embedding if needed
            if needs_embedding && !final_text.trim().is_empty() {
                batcher.add(task.frame_id, final_text);

                // Flush batch if ready
                if batcher.should_flush() {
                    match batcher.flush() {
                        Ok(count) => {
                            embeddings_generated += count;
                            // Store ready embeddings
                            let ready = batcher.take_embeddings();
                            if !ready.is_empty() {
                                if let Err(err) = self.add_embeddings(ready) {
                                    tracing::warn!(?err, "failed to add embeddings");
                                }
                            }
                        }
                        Err(err) => {
                            tracing::warn!(?err, "batch embedding failed");
                        }
                    }
                }
            }

            // Mark frame as enriched
            self.mark_frame_enriched(task.frame_id);
            self.complete_enrichment_task(task.frame_id);
            frames_processed += 1;

            // Update checkpoint in queue for crash recovery
            let chunks_done = u32::try_from(embeddings_generated).unwrap_or(u32::MAX);
            self.toc.enrichment_queue.update_checkpoint(
                task.frame_id,
                chunks_done,
                task.chunks_total.max(chunks_done),
            );
        }

        // Flush remaining batch
        if batcher.pending_count() > 0 {
            match batcher.flush() {
                Ok(count) => {
                    embeddings_generated += count;
                    let ready = batcher.take_embeddings();
                    if !ready.is_empty() {
                        if let Err(err) = self.add_embeddings(ready) {
                            tracing::warn!(?err, "failed to add final embeddings");
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(?err, "final batch embedding failed");
                }
            }
        }

        tracing::info!(
            frames_processed,
            embeddings_generated,
            "enrichment with embeddings complete"
        );

        Ok((frames_processed, embeddings_generated))
    }

    /// Check if vector embeddings are enabled.
    #[must_use]
    pub fn has_embeddings(&self) -> bool {
        self.vec_enabled && self.vec_index.is_some()
    }

    /// Get vector count from the index.
    #[must_use]
    pub fn vector_count(&self) -> usize {
        self.toc.indexes.vec.as_ref().map_or(0, |m| {
            #[allow(clippy::cast_possible_truncation)]
            let count = m.vector_count as usize;
            count
        })
    }
}

/// Statistics about enrichment state.
#[derive(Debug, Clone)]
pub struct EnrichmentStats {
    /// Total active frames.
    pub total_frames: usize,
    /// Frames that have been fully enriched.
    pub enriched_frames: usize,
    /// Frames pending enrichment.
    pub pending_frames: usize,
    /// Frames that are searchable but not enriched.
    pub searchable_only: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enrichment_stats_default() {
        let stats = EnrichmentStats {
            total_frames: 100,
            enriched_frames: 50,
            pending_frames: 10,
            searchable_only: 50,
        };
        assert_eq!(
            stats.enriched_frames + stats.searchable_only,
            stats.total_frames
        );
    }
}
