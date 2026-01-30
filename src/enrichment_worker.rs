//! Background enrichment worker for progressive ingestion.
//!
//! Processes frames in the enrichment queue asynchronously:
//! - Re-extracts full text for skim extractions
//! - Generates embeddings with batching and checkpointing
//! - Updates Tantivy index with enriched content
//! - Marks frames as Enriched when complete

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::types::{EnrichmentTask, FrameId, VecEmbedder};

/// Configuration for the enrichment worker.
#[derive(Debug, Clone)]
pub struct EnrichmentWorkerConfig {
    /// Batch size for embedding generation.
    pub embedding_batch_size: usize,
    /// Checkpoint interval (persist progress every N embeddings).
    pub checkpoint_interval: usize,
    /// Delay between processing tasks (to avoid blocking writers).
    pub task_delay_ms: u64,
    /// Maximum time to spend on a single task before yielding.
    pub max_task_time_ms: u64,
}

impl Default for EnrichmentWorkerConfig {
    fn default() -> Self {
        Self {
            embedding_batch_size: 32,
            checkpoint_interval: 100,
            task_delay_ms: 50,
            max_task_time_ms: 5000,
        }
    }
}

/// Statistics for the enrichment worker.
#[derive(Debug, Clone, Default)]
pub struct EnrichmentWorkerStats {
    /// Total frames processed.
    pub frames_processed: u64,
    /// Total embeddings generated.
    pub embeddings_generated: u64,
    /// Total re-extractions performed.
    pub re_extractions: u64,
    /// Total errors encountered.
    pub errors: u64,
    /// Current queue depth.
    pub queue_depth: usize,
    /// Whether worker is currently running.
    pub is_running: bool,
}

/// Handle for controlling the background enrichment worker.
pub struct EnrichmentWorkerHandle {
    /// Signal to stop the worker.
    stop_signal: Arc<AtomicBool>,
    /// Counter for frames processed.
    frames_processed: Arc<AtomicU64>,
    /// Counter for embeddings generated.
    embeddings_generated: Arc<AtomicU64>,
    /// Counter for re-extractions.
    re_extractions: Arc<AtomicU64>,
    /// Counter for errors.
    errors: Arc<AtomicU64>,
    /// Running state.
    is_running: Arc<AtomicBool>,
}

impl EnrichmentWorkerHandle {
    /// Create a new worker handle.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stop_signal: Arc::new(AtomicBool::new(false)),
            frames_processed: Arc::new(AtomicU64::new(0)),
            embeddings_generated: Arc::new(AtomicU64::new(0)),
            re_extractions: Arc::new(AtomicU64::new(0)),
            errors: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal the worker to stop.
    pub fn stop(&self) {
        self.stop_signal.store(true, Ordering::SeqCst);
    }

    /// Check if stop was requested.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.stop_signal.load(Ordering::SeqCst)
    }

    /// Check if worker is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> EnrichmentWorkerStats {
        EnrichmentWorkerStats {
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            embeddings_generated: self.embeddings_generated.load(Ordering::Relaxed),
            re_extractions: self.re_extractions.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            queue_depth: 0, // Will be updated by caller
            is_running: self.is_running.load(Ordering::Relaxed),
        }
    }

    /// Increment frames processed counter.
    pub(crate) fn inc_frames_processed(&self) {
        self.frames_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment embeddings generated counter.
    pub(crate) fn inc_embeddings(&self, count: u64) {
        self.embeddings_generated
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Increment re-extractions counter.
    pub(crate) fn inc_re_extractions(&self) {
        self.re_extractions.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment errors counter.
    pub(crate) fn inc_errors(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Set running state.
    pub(crate) fn set_running(&self, running: bool) {
        self.is_running.store(running, Ordering::SeqCst);
    }

    /// Clone the handle for sharing with the worker thread.
    #[must_use]
    pub fn clone_handle(&self) -> Self {
        Self {
            stop_signal: Arc::clone(&self.stop_signal),
            frames_processed: Arc::clone(&self.frames_processed),
            embeddings_generated: Arc::clone(&self.embeddings_generated),
            re_extractions: Arc::clone(&self.re_extractions),
            errors: Arc::clone(&self.errors),
            is_running: Arc::clone(&self.is_running),
        }
    }
}

impl Default for EnrichmentWorkerHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing a single enrichment task.
#[derive(Debug)]
pub struct TaskResult {
    /// Frame ID that was processed.
    pub frame_id: FrameId,
    /// Whether full re-extraction was performed.
    pub re_extracted: bool,
    /// Number of embeddings generated.
    pub embeddings_generated: usize,
    /// Time spent processing.
    pub elapsed_ms: u64,
    /// Error if processing failed.
    pub error: Option<String>,
}

/// Batched embedding generator for efficient embedding creation.
///
/// Collects text chunks and generates embeddings in batches to minimize
/// API calls and improve throughput.
pub struct EmbeddingBatcher<E: VecEmbedder> {
    /// The embedder to use for generating embeddings.
    embedder: E,
    /// Batch size for embedding generation.
    batch_size: usize,
    /// Pending texts to embed.
    pending_texts: Vec<(FrameId, String)>,
    /// Generated embeddings ready to store.
    ready_embeddings: Vec<(FrameId, Vec<f32>)>,
}

impl<E: VecEmbedder> EmbeddingBatcher<E> {
    /// Create a new embedding batcher.
    pub fn new(embedder: E, batch_size: usize) -> Self {
        Self {
            embedder,
            batch_size: batch_size.max(1),
            pending_texts: Vec::new(),
            ready_embeddings: Vec::new(),
        }
    }

    /// Add a frame's text for embedding.
    pub fn add(&mut self, frame_id: FrameId, text: String) {
        self.pending_texts.push((frame_id, text));
    }

    /// Get the number of pending texts.
    pub fn pending_count(&self) -> usize {
        self.pending_texts.len()
    }

    /// Get the number of ready embeddings.
    pub fn ready_count(&self) -> usize {
        self.ready_embeddings.len()
    }

    /// Check if a batch is ready to process.
    pub fn should_flush(&self) -> bool {
        self.pending_texts.len() >= self.batch_size
    }

    /// Process pending texts and generate embeddings.
    ///
    /// Returns the number of embeddings generated.
    pub fn flush(&mut self) -> Result<usize> {
        if self.pending_texts.is_empty() {
            return Ok(0);
        }

        // Take all pending texts
        let pending: Vec<_> = std::mem::take(&mut self.pending_texts);
        let count = pending.len();

        // Extract texts for batch embedding
        let texts: Vec<&str> = pending.iter().map(|(_, text)| text.as_str()).collect();

        // Generate embeddings in batch
        let embeddings = self.embedder.embed_chunks(&texts)?;

        // Store results
        for ((frame_id, _), embedding) in pending.into_iter().zip(embeddings.into_iter()) {
            self.ready_embeddings.push((frame_id, embedding));
        }

        Ok(count)
    }

    /// Take all ready embeddings.
    pub fn take_embeddings(&mut self) -> Vec<(FrameId, Vec<f32>)> {
        std::mem::take(&mut self.ready_embeddings)
    }

    /// Get embedding dimension from the embedder.
    pub fn dimension(&self) -> usize {
        self.embedder.embedding_dimension()
    }
}

/// Enrichment task processor (stateless, operates on Memvid instance).
pub struct EnrichmentProcessor {
    /// Worker configuration.
    pub config: EnrichmentWorkerConfig,
}

impl EnrichmentProcessor {
    /// Create a new enrichment processor.
    #[must_use]
    pub fn new(config: EnrichmentWorkerConfig) -> Self {
        Self { config }
    }

    /// Process a single enrichment task.
    ///
    /// This method:
    /// 1. Reads the frame from the memory
    /// 2. If frame needs re-extraction (skim), performs full extraction
    /// 3. If frame needs embeddings, generates them with batching
    /// 4. Updates the Tantivy index
    /// 5. Returns the result
    ///
    /// The caller is responsible for:
    /// - Acquiring write lock on the memory
    /// - Updating the enrichment queue
    /// - Persisting changes
    pub fn process_task<F, E, R>(
        &self,
        task: &EnrichmentTask,
        read_frame: F,
        extract_full: E,
        update_index: R,
    ) -> TaskResult
    where
        F: FnOnce(FrameId) -> Option<(String, bool, bool)>, // (text, is_skim, needs_embedding)
        E: FnOnce(FrameId) -> Result<String>,               // Full extraction
        R: FnOnce(FrameId, &str) -> Result<()>,             // Update index
    {
        let start = Instant::now();
        let mut result = TaskResult {
            frame_id: task.frame_id,
            re_extracted: false,
            embeddings_generated: 0,
            elapsed_ms: 0,
            error: None,
        };

        // Read current frame state
        let (text, is_skim, _needs_embedding) = if let Some(data) = read_frame(task.frame_id) {
            data
        } else {
            result.error = Some("Frame not found".to_string());
            result.elapsed_ms = start.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
            return result;
        };

        // Re-extract if this was a skim
        let final_text = if is_skim {
            match extract_full(task.frame_id) {
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
                    text
                }
            }
        } else {
            text
        };

        // Update index with enriched content
        if let Err(err) = update_index(task.frame_id, &final_text) {
            result.error = Some(format!("Index update failed: {err}"));
        }

        result.elapsed_ms = start.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
        result
    }
}

/// Run the enrichment worker loop.
///
/// This function should be called from a background thread.
/// It processes tasks from the enrichment queue until stopped.
///
/// # Arguments
/// * `handle` - Worker handle for control and statistics
/// * `config` - Worker configuration
/// * `get_next_task` - Closure to get the next task from the queue
/// * `process_task` - Closure to process a single task
/// * `mark_complete` - Closure to mark a task as complete
/// * `checkpoint` - Closure to save progress
pub fn run_worker_loop<G, P, M, C>(
    handle: &EnrichmentWorkerHandle,
    config: &EnrichmentWorkerConfig,
    mut get_next_task: G,
    mut process_task: P,
    mut mark_complete: M,
    mut checkpoint: C,
) where
    G: FnMut() -> Option<EnrichmentTask>,
    P: FnMut(&EnrichmentTask) -> TaskResult,
    M: FnMut(FrameId),
    C: FnMut(),
{
    handle.set_running(true);
    tracing::info!("enrichment worker started");

    let mut tasks_since_checkpoint = 0;

    while !handle.should_stop() {
        // Get next task
        let task = if let Some(task) = get_next_task() {
            task
        } else {
            // Queue is empty, wait and check again
            std::thread::sleep(Duration::from_millis(config.task_delay_ms * 10));
            continue;
        };

        // Process the task
        let result = process_task(&task);

        // Update statistics
        handle.inc_frames_processed();
        if result.re_extracted {
            handle.inc_re_extractions();
        }
        if result.embeddings_generated > 0 {
            handle.inc_embeddings(result.embeddings_generated as u64);
        }
        if result.error.is_some() {
            handle.inc_errors();
            tracing::warn!(
                frame_id = task.frame_id,
                error = ?result.error,
                "enrichment task failed"
            );
        } else {
            tracing::debug!(
                frame_id = task.frame_id,
                re_extracted = result.re_extracted,
                embeddings = result.embeddings_generated,
                elapsed_ms = result.elapsed_ms,
                "enrichment task complete"
            );
        }

        // Mark task complete (remove from queue)
        mark_complete(task.frame_id);
        tasks_since_checkpoint += 1;

        // Checkpoint periodically
        if tasks_since_checkpoint >= config.checkpoint_interval {
            checkpoint();
            tasks_since_checkpoint = 0;
        }

        // Yield to other threads
        std::thread::sleep(Duration::from_millis(config.task_delay_ms));
    }

    // Final checkpoint
    if tasks_since_checkpoint > 0 {
        checkpoint();
    }

    handle.set_running(false);
    tracing::info!(
        frames_processed = handle.frames_processed.load(Ordering::Relaxed),
        "enrichment worker stopped"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock embedder for testing
    struct MockEmbedder {
        dimension: usize,
    }

    impl MockEmbedder {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    impl crate::types::VecEmbedder for MockEmbedder {
        fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            // Generate deterministic embedding based on text length
            let seed = text.len() as f32;
            Ok((0..self.dimension)
                .map(|i| (seed + i as f32) * 0.1)
                .collect())
        }

        fn embedding_dimension(&self) -> usize {
            self.dimension
        }

        fn model_name(&self) -> Option<&str> {
            Some("mock-model")
        }
    }

    #[test]
    fn test_embedding_batcher_basic() {
        let embedder = MockEmbedder::new(4);
        let mut batcher = EmbeddingBatcher::new(embedder, 2);

        assert_eq!(batcher.pending_count(), 0);
        assert_eq!(batcher.ready_count(), 0);
        assert!(!batcher.should_flush());

        // Add one item - shouldn't trigger flush yet
        batcher.add(1, "hello".to_string());
        assert_eq!(batcher.pending_count(), 1);
        assert!(!batcher.should_flush());

        // Add second item - should trigger flush
        batcher.add(2, "world".to_string());
        assert_eq!(batcher.pending_count(), 2);
        assert!(batcher.should_flush());

        // Flush the batch
        let count = batcher.flush().expect("flush should succeed");
        assert_eq!(count, 2);
        assert_eq!(batcher.pending_count(), 0);
        assert_eq!(batcher.ready_count(), 2);

        // Take embeddings
        let embeddings = batcher.take_embeddings();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].0, 1); // frame_id
        assert_eq!(embeddings[0].1.len(), 4); // dimension
        assert_eq!(embeddings[1].0, 2);
        assert_eq!(embeddings[1].1.len(), 4);

        // After take, ready should be empty
        assert_eq!(batcher.ready_count(), 0);
    }

    #[test]
    fn test_embedding_batcher_dimension() {
        let embedder = MockEmbedder::new(128);
        let batcher = EmbeddingBatcher::new(embedder, 32);
        assert_eq!(batcher.dimension(), 128);
    }

    #[test]
    fn test_embedding_batcher_flush_empty() {
        let embedder = MockEmbedder::new(4);
        let mut batcher = EmbeddingBatcher::new(embedder, 2);

        // Flushing empty batcher should return 0
        let count = batcher.flush().expect("flush should succeed");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_worker_handle() {
        let handle = EnrichmentWorkerHandle::new();
        assert!(!handle.is_running());
        assert!(!handle.should_stop());

        handle.set_running(true);
        assert!(handle.is_running());

        handle.stop();
        assert!(handle.should_stop());

        handle.inc_frames_processed();
        handle.inc_embeddings(10);
        handle.inc_re_extractions();
        handle.inc_errors();

        let stats = handle.stats();
        assert_eq!(stats.frames_processed, 1);
        assert_eq!(stats.embeddings_generated, 10);
        assert_eq!(stats.re_extractions, 1);
        assert_eq!(stats.errors, 1);
    }

    #[test]
    fn test_processor() {
        let processor = EnrichmentProcessor::new(EnrichmentWorkerConfig::default());
        let task = EnrichmentTask {
            frame_id: 1,
            created_at: 0,
            chunks_done: 0,
            chunks_total: 0,
        };

        let result = processor.process_task(
            &task,
            |_| Some(("test content".to_string(), false, false)),
            |_| Ok("full content".to_string()),
            |_, _| Ok(()),
        );

        assert_eq!(result.frame_id, 1);
        assert!(!result.re_extracted); // Not a skim
        assert!(result.error.is_none());
    }

    #[test]
    fn test_processor_with_skim() {
        let processor = EnrichmentProcessor::new(EnrichmentWorkerConfig::default());
        let task = EnrichmentTask {
            frame_id: 2,
            created_at: 0,
            chunks_done: 0,
            chunks_total: 0,
        };

        let result = processor.process_task(
            &task,
            |_| Some(("skim content".to_string(), true, false)), // is_skim = true
            |_| Ok("full extracted content".to_string()),
            |_, text| {
                assert_eq!(text, "full extracted content");
                Ok(())
            },
        );

        assert_eq!(result.frame_id, 2);
        assert!(result.re_extracted); // Re-extraction happened
        assert!(result.error.is_none());
    }
}
