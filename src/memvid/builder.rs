#[cfg(feature = "parallel_segments")]
use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(feature = "parallel_segments")]
use tracing::info;

#[cfg(feature = "parallel_segments")]
use super::{
    lifecycle::Memvid,
    segments::{LexSegmentArtifact, TimeSegmentArtifact, VecSegmentArtifact},
    workers::SegmentResult,
};
#[cfg(feature = "parallel_segments")]
use crate::{
    MemvidError, Result,
    types::{PutOptions, SegmentKind, SegmentSpan, SegmentStats, VecIndexManifest},
};

#[cfg(feature = "parallel_segments")]
const DEFAULT_SEGMENT_TOKENS: usize = 2_048;
#[cfg(feature = "parallel_segments")]
const DEFAULT_SEGMENT_PAGES: usize = 4;
#[cfg(feature = "parallel_segments")]
const DEFAULT_MEMORY_CAP_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GiB
#[cfg(feature = "parallel_segments")]
const DEFAULT_QUEUE_DEPTH: usize = 64;

#[cfg(feature = "parallel_segments")]
#[derive(Debug, Clone)]
pub struct BuildOpts {
    pub segment_tokens: usize,
    pub segment_pages: usize,
    pub threads: usize,
    pub zstd_level: i32,
    pub memory_cap_bytes: u64,
    pub queue_depth: usize,
    pub vec_compression: crate::types::VectorCompression,
}

#[cfg(feature = "parallel_segments")]
impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            segment_tokens: DEFAULT_SEGMENT_TOKENS,
            segment_pages: DEFAULT_SEGMENT_PAGES,
            threads: default_worker_threads(),
            zstd_level: 3,
            memory_cap_bytes: DEFAULT_MEMORY_CAP_BYTES,
            queue_depth: DEFAULT_QUEUE_DEPTH,
            vec_compression: crate::types::VectorCompression::None,
        }
    }
}

#[cfg(feature = "parallel_segments")]
impl BuildOpts {
    pub fn sanitize(&mut self) {
        if self.segment_tokens == 0 {
            self.segment_tokens = DEFAULT_SEGMENT_TOKENS;
        }
        if self.segment_pages == 0 {
            self.segment_pages = DEFAULT_SEGMENT_PAGES;
        }
        if self.threads == 0 {
            self.threads = default_worker_threads();
        }
        if self.queue_depth == 0 {
            self.queue_depth = DEFAULT_QUEUE_DEPTH;
        }
        if self.memory_cap_bytes == 0 {
            self.memory_cap_bytes = DEFAULT_MEMORY_CAP_BYTES;
        }
        self.zstd_level = self.zstd_level.clamp(1, 9);
    }
}

#[cfg(feature = "parallel_segments")]
fn default_worker_threads() -> usize {
    num_cpus::get().saturating_sub(1).max(1)
}

/// Source bytes for pending parallel inputs.
#[cfg(feature = "parallel_segments")]
#[derive(Debug, Clone)]
pub enum ParallelPayload {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

/// Caller-specified payload + metadata for the parallel builder.
#[cfg(feature = "parallel_segments")]
#[derive(Debug, Clone)]
pub struct ParallelInput {
    pub payload: ParallelPayload,
    pub options: PutOptions,
    /// Embedding for the parent document (used when no chunking or single embedding).
    pub embedding: Option<Vec<f32>>,
    /// Pre-computed embeddings for each chunk. When provided, enables semantic search
    /// on all child chunks, not just the parent frame.
    pub chunk_embeddings: Option<Vec<Vec<f32>>>,
}

#[cfg(feature = "parallel_segments")]
impl Memvid {
    pub fn put_parallel<P>(&mut self, sources: &[P], mut opts: BuildOpts) -> Result<()>
    where
        P: AsRef<Path>,
    {
        opts.sanitize();
        if sources.is_empty() {
            return Ok(());
        }
        let mut inputs = Vec::with_capacity(sources.len());
        for source in sources {
            inputs.push(ParallelInput {
                payload: ParallelPayload::Path(source.as_ref().to_path_buf()),
                options: PutOptions::default(),
                embedding: None,
                chunk_embeddings: None,
            });
        }
        self.put_parallel_inputs(&inputs, opts).map(|_| ())
    }

    /// Ingests caller-supplied payloads (plus metadata) and seals them via the parallel builder.
    pub fn put_parallel_inputs(
        &mut self,
        inputs: &[ParallelInput],
        mut opts: BuildOpts,
    ) -> Result<Vec<u64>> {
        opts.sanitize();
        if inputs.is_empty() {
            info!(
                ingested_documents = 0,
                "parallel ingestion enqueued documents"
            );
            self.commit_parallel(opts)?;
            return Ok(Vec::new());
        }
        let mut seqs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let seq = match &input.payload {
                ParallelPayload::Path(path) => {
                    let bytes = fs::read(path)?;
                    self.ingest_parallel_bytes(&bytes, input)?
                }
                ParallelPayload::Bytes(bytes) => self.ingest_parallel_bytes(bytes, input)?,
            };
            seqs.push(seq);
        }
        info!(
            ingested_documents = seqs.len(),
            "parallel ingestion enqueued documents"
        );
        self.commit_parallel(opts)?;
        Ok(seqs)
    }

    pub fn commit_parallel(&mut self, mut opts: BuildOpts) -> Result<()> {
        opts.sanitize();
        self.commit_parallel_with_opts(&opts)
    }

    fn ingest_parallel_bytes(&mut self, bytes: &[u8], input: &ParallelInput) -> Result<u64> {
        // If chunk embeddings are provided, use put_with_chunk_embeddings for full semantic coverage
        if let Some(chunk_embeddings) = input.chunk_embeddings.as_ref() {
            self.put_with_chunk_embeddings(
                bytes,
                input.embedding.clone(),
                chunk_embeddings.clone(),
                input.options.clone(),
            )
        } else if let Some(embedding) = input.embedding.as_ref() {
            // Only parent embedding - use legacy path (chunks won't be searchable via semantic)
            self.put_with_embedding_and_options(bytes, embedding.clone(), input.options.clone())
        } else {
            self.put_bytes_with_options(bytes, input.options.clone())
        }
    }

    pub(crate) fn append_parallel_segments(&mut self, results: Vec<SegmentResult>) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }
        let mut appended_any = false;
        for result in results {
            let span = result.span;
            if let Some(segment) = result.lex {
                self.append_parallel_lex_segment(&segment.artifact, span, segment.stats)?;
                appended_any = true;
            }
            if let Some(segment) = result.vec {
                self.append_parallel_vec_segment(&segment.artifact, span, segment.stats)?;
                appended_any = true;
            }
            if let Some(segment) = result.time {
                self.append_parallel_time_segment(&segment.artifact, span, segment.stats)?;
                appended_any = true;
            }
        }
        if appended_any {
            self.dirty = true;
            if let Some(wal) = self.manifest_wal.as_mut() {
                wal.flush()?;
            }
        }
        Ok(())
    }

    fn append_parallel_lex_segment(
        &mut self,
        artifact: &LexSegmentArtifact,
        span: Option<SegmentSpan>,
        stats: SegmentStats,
    ) -> Result<()> {
        let segment_id = self.toc.segment_catalog.next_segment_id;
        let mut descriptor = self.append_lex_segment(artifact, segment_id)?;
        if let Some(span) = span {
            Self::decorate_segment_common(&mut descriptor.common, span);
        }
        let descriptor_for_manifest = descriptor.clone();
        self.toc.segment_catalog.lex_segments.push(descriptor);
        self.record_index_segment(SegmentKind::Lexical, descriptor_for_manifest.common, stats)?;
        self.toc.segment_catalog.version = self.toc.segment_catalog.version.max(1);
        self.toc.segment_catalog.next_segment_id = segment_id.saturating_add(1);
        self.lex_enabled = true;
        Ok(())
    }

    fn append_parallel_vec_segment(
        &mut self,
        artifact: &VecSegmentArtifact,
        span: Option<SegmentSpan>,
        stats: SegmentStats,
    ) -> Result<()> {
        if let Some(existing_dim) = self.effective_vec_index_dimension()? {
            if existing_dim != artifact.dimension {
                return Err(MemvidError::VecDimensionMismatch {
                    expected: existing_dim,
                    actual: artifact.dimension as usize,
                });
            }
        }

        let segment_id = self.toc.segment_catalog.next_segment_id;
        let mut descriptor = self.append_vec_segment(artifact, segment_id)?;
        if let Some(span) = span {
            Self::decorate_segment_common(&mut descriptor.common, span);
        }
        let descriptor_for_manifest = descriptor.clone();
        self.toc
            .segment_catalog
            .vec_segments
            .push(descriptor.clone());
        tracing::info!(
            segment_id,
            vec_count = artifact.vector_count,
            offset = descriptor.common.bytes_offset,
            length = descriptor.common.bytes_length,
            catalog_vec_segments = self.toc.segment_catalog.vec_segments.len(),
            "append_parallel_vec_segment: pushed descriptor to catalog"
        );
        self.record_index_segment(SegmentKind::Vector, descriptor_for_manifest.common, stats)?;
        self.toc.segment_catalog.version = self.toc.segment_catalog.version.max(1);
        self.toc.segment_catalog.next_segment_id = segment_id.saturating_add(1);

        // Keep the global vec manifest in sync for auto-detection and stats.
        // Segment-based vector storage uses `bytes_length == 0` as a placeholder.
        if self.toc.indexes.vec.is_none() {
            let empty_offset = self.data_end;
            let empty_checksum = *b"\xe3\xb0\xc4\x42\x98\xfc\x1c\x14\x9a\xfb\xf4\xc8\x99\x6f\xb9\x24\
                                    \x27\xae\x41\xe4\x64\x9b\x93\x4c\xa4\x95\x99\x1b\x78\x52\xb8\x55";
            self.toc.indexes.vec = Some(VecIndexManifest {
                vector_count: 0,
                dimension: 0,
                bytes_offset: empty_offset,
                bytes_length: 0,
                checksum: empty_checksum,
                compression_mode: self.vec_compression.clone(),
                model: self.vec_model.clone(),
            });
        }
        if let Some(manifest) = self.toc.indexes.vec.as_mut() {
            if manifest.dimension == 0 {
                manifest.dimension = artifact.dimension;
            }
            if manifest.bytes_length == 0 {
                manifest.vector_count = manifest.vector_count.saturating_add(artifact.vector_count);
                manifest.compression_mode = artifact.compression.clone();
            }
        }

        self.vec_enabled = true;
        Ok(())
    }

    fn append_parallel_time_segment(
        &mut self,
        artifact: &TimeSegmentArtifact,
        span: Option<SegmentSpan>,
        stats: SegmentStats,
    ) -> Result<()> {
        let segment_id = self.toc.segment_catalog.next_segment_id;
        let mut descriptor = self.append_time_segment(artifact, segment_id)?;
        if let Some(span) = span {
            Self::decorate_segment_common(&mut descriptor.common, span);
        }
        let descriptor_for_manifest = descriptor.clone();
        self.toc.segment_catalog.time_segments.push(descriptor);
        self.record_index_segment(SegmentKind::Time, descriptor_for_manifest.common, stats)?;
        self.toc.segment_catalog.version = self.toc.segment_catalog.version.max(1);
        self.toc.segment_catalog.next_segment_id = segment_id.saturating_add(1);
        Ok(())
    }
}

#[cfg(all(test, feature = "parallel_segments"))]
mod tests {
    use super::*;
    use crate::{MemvidError, memvid::lifecycle::Memvid, run_serial_test};
    use tempfile::tempdir;

    #[test]
    fn parallel_commit_persists_segments() -> Result<()> {
        run_serial_test(|| -> Result<()> {
            let dir = tempdir()?;
            let path = dir.path().join("parallel.mv2");
            let mut mem = Memvid::create(&path)?;
            mem.enable_lex()?;
            mem.enable_vec()?;
            mem.put_bytes(b"hello world")?;
            mem.put_bytes(b"another document")?;
            // Use minimal segment_tokens to ensure segments are created with small test data
            let mut opts = BuildOpts::default();
            opts.segment_tokens = 2; // Minimal threshold for 2 tiny documents
            mem.commit_parallel(opts)?;
            assert!(
                !mem.toc.segment_catalog.index_segments.is_empty(),
                "parallel commit should emit segments"
            );
            if let Some(wal) = mem.manifest_wal.as_ref() {
                assert!(
                    wal.replay()?.is_empty(),
                    "manifest wal should be flushed after commit"
                );
            }
            drop(mem);

            let reopened = Memvid::open(&path)?;
            assert!(
                !reopened.toc.segment_catalog.index_segments.is_empty(),
                "segments persist after reopen"
            );
            Ok(())
        })
    }

    #[test]
    fn parallel_vec_manifest_persists_dimension_and_count() -> Result<()> {
        run_serial_test(|| -> Result<()> {
            let dir = tempdir()?;
            let path = dir.path().join("parallel_vec.mv2");
            let mut mem = Memvid::create(&path)?;
            mem.enable_vec()?;

            let inputs = vec![ParallelInput {
                payload: ParallelPayload::Bytes(b"hello world".to_vec()),
                options: PutOptions::default(),
                embedding: Some(vec![0.0f32; 1536]),
                chunk_embeddings: None,
            }];

            let mut opts = BuildOpts::default();
            opts.segment_tokens = 2;
            mem.put_parallel_inputs(&inputs, opts)?;

            let manifest = mem.toc.indexes.vec.as_ref().expect("vec manifest");
            assert_eq!(manifest.dimension, 1536);
            assert!(manifest.vector_count > 0);

            drop(mem);

            let reopened = Memvid::open_read_only(&path)?;
            assert_eq!(reopened.vec_index_dimension(), Some(1536));
            assert_eq!(reopened.effective_vec_index_dimension()?, Some(1536));
            Ok(())
        })
    }

    #[test]
    fn effective_vec_dimension_falls_back_to_segments_when_manifest_zero() -> Result<()> {
        run_serial_test(|| -> Result<()> {
            let dir = tempdir()?;
            let path = dir.path().join("segment_dim.mv2");
            let mut mem = Memvid::create(&path)?;
            mem.enable_vec()?;

            let inputs = vec![ParallelInput {
                payload: ParallelPayload::Bytes(b"hello world".to_vec()),
                options: PutOptions::default(),
                embedding: Some(vec![0.0f32; 1536]),
                chunk_embeddings: None,
            }];

            let mut opts = BuildOpts::default();
            opts.segment_tokens = 2;
            mem.put_parallel_inputs(&inputs, opts)?;

            // Simulate older files that kept a placeholder vec manifest (dimension=0) even when
            // vector segments exist. Effective detection should still work.
            mem.toc
                .indexes
                .vec
                .as_mut()
                .expect("vec manifest")
                .dimension = 0;
            mem.rewrite_toc_footer()?;
            mem.header.toc_checksum = mem.toc.toc_checksum;
            crate::persist_header(&mut mem.file, &mem.header)?;
            mem.file.sync_all()?;

            drop(mem);

            let reopened = Memvid::open_read_only(&path)?;
            assert_eq!(reopened.vec_index_dimension(), None);
            assert_eq!(reopened.effective_vec_index_dimension()?, Some(1536));
            Ok(())
        })
    }

    #[test]
    fn vec_search_with_embedding_rejects_mismatch_for_segment_only_manifest_zero() -> Result<()> {
        run_serial_test(|| -> Result<()> {
            let dir = tempdir()?;
            let path = dir.path().join("segment_mismatch.mv2");
            let mut mem = Memvid::create(&path)?;
            mem.enable_vec()?;

            let inputs = vec![ParallelInput {
                payload: ParallelPayload::Bytes(b"hello world".to_vec()),
                options: PutOptions::default(),
                embedding: Some(vec![0.0f32; 1536]),
                chunk_embeddings: None,
            }];

            let mut opts = BuildOpts::default();
            opts.segment_tokens = 2;
            mem.put_parallel_inputs(&inputs, opts)?;

            mem.toc
                .indexes
                .vec
                .as_mut()
                .expect("vec manifest")
                .dimension = 0;
            mem.rewrite_toc_footer()?;
            mem.header.toc_checksum = mem.toc.toc_checksum;
            crate::persist_header(&mut mem.file, &mem.header)?;
            mem.file.sync_all()?;

            drop(mem);

            let mut reopened = Memvid::open_read_only(&path)?;
            let err = reopened
                .vec_search_with_embedding("hello", &vec![0.0f32; 384], 5, 240, None)
                .unwrap_err();
            match err {
                MemvidError::VecDimensionMismatch { expected, actual } => {
                    assert_eq!(expected, 1536);
                    assert_eq!(actual, 384);
                }
                other => panic!("expected VecDimensionMismatch, got {other:?}"),
            }
            Ok(())
        })
    }
}
