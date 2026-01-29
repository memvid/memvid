use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::lex::{LexIndex, LexIndexArtifact, LexIndexBuilder};
use crate::memvid::lifecycle::Memvid;
use crate::types::{Frame, FrameId, FrameStatus, VectorCompression};
use crate::{MemvidError, Result, VecIndex, VecIndexArtifact};

impl Memvid {
    #[allow(dead_code)]
    pub(crate) fn build_lex_artifact(&mut self) -> Result<Option<(LexIndexArtifact, LexIndex)>> {
        if !self.lex_enabled {
            return Ok(None);
        }
        let mut builder = LexIndexBuilder::new();
        let empty_tags = HashMap::new();
        let frames: Vec<Frame> = self
            .toc
            .frames
            .iter()
            .filter(|frame| frame.status == FrameStatus::Active)
            .cloned()
            .collect();
        for frame in frames {
            let content = self.frame_content(&frame)?;
            let uri = frame
                .uri
                .clone()
                .unwrap_or_else(|| crate::default_uri(frame.id));
            builder.add_document(
                frame.id,
                &uri,
                frame.title.as_deref(),
                &content,
                &empty_tags,
            );
        }
        let artifact = builder.finish()?;
        let index = LexIndex::decode(&artifact.bytes)?;
        Ok(Some((artifact, index)))
    }

    pub(crate) fn build_vec_artifact(
        &mut self,
        new_docs: &[(FrameId, Vec<f32>)],
    ) -> Result<Option<(VecIndexArtifact, VecIndex)>> {
        if !self.vec_enabled {
            return Ok(None);
        }
        let mut builder = VecIndexBuilder::new();
        if let Some(index) = self.vec_index.as_ref() {
            for (frame_id, embedding) in index.entries() {
                if self.frame_is_active(frame_id) {
                    builder.add_document(frame_id, embedding.to_vec());
                }
            }
        }
        for (frame_id, embedding) in new_docs {
            builder.add_document(*frame_id, embedding.clone());
        }
        let artifact = builder.finish()?;
        let index = VecIndex::decode(&artifact.bytes)?;
        Ok(Some((artifact, index)))
    }

    pub(crate) fn ensure_lex_index(&mut self) -> Result<()> {
        if self.lex_index.is_some() {
            return Ok(());
        }
        self.load_lex_index_from_manifest()
    }

    pub(crate) fn ensure_vec_index(&mut self) -> Result<()> {
        if self.vec_index.is_some() {
            return Ok(());
        }
        self.load_vec_index_from_manifest()?;

        // If no monolithic index was loaded but vec_segments exist, build from segments
        if self.vec_index.is_none()
            && self.vec_enabled
            && !self.toc.segment_catalog.vec_segments.is_empty()
        {
            self.build_vec_index_from_segments()?;
        }

        Ok(())
    }

    pub(crate) fn load_lex_index_from_manifest(&mut self) -> Result<()> {
        if let Some(manifest) = &self.toc.indexes.lex {
            // Empty manifest (placeholder for enabled but not yet populated index)
            if manifest.bytes_length == 0 {
                self.lex_index = None;
                return Ok(());
            }

            let bytes =
                if let Ok(bytes) = self.read_range(manifest.bytes_offset, manifest.bytes_length) {
                    bytes
                } else {
                    // Don't disable lex if loading fails - keep it enabled
                    self.lex_index = None;
                    return Ok(());
                };
            match LexIndex::decode(&bytes) {
                Ok(mut index) => {
                    self.hydrate_lex_index_metadata(&mut index);
                    self.lex_index = Some(index);
                }
                Err(_) => {
                    // Don't disable lex if decoding fails - keep it enabled
                    // CRITICAL: Don't modify self.toc during read-only operations!
                    // If dirty=true and Drop commits, it will corrupt the manifest.
                    self.lex_index = None;
                }
            }
        } else {
            self.lex_index = None;
        }
        Ok(())
    }

    pub(crate) fn load_vec_index_from_manifest(&mut self) -> Result<()> {
        // Load the model name from the manifest regardless of validation success
        self.vec_model = self.toc.indexes.vec.as_ref().and_then(|m| m.model.clone());

        if let Some(manifest) = &self.toc.indexes.vec {
            // Empty manifest (placeholder for enabled but not yet populated index)
            if manifest.bytes_length == 0 {
                self.vec_index = None;
                return Ok(());
            }

            let bytes =
                if let Ok(bytes) = self.read_range(manifest.bytes_offset, manifest.bytes_length) {
                    bytes
                } else {
                    self.vec_index = None;
                    // Don't disable vec if loading fails - keep it enabled
                    // self.vec_enabled = false;
                    return Ok(());
                };
            match catch_unwind(AssertUnwindSafe(|| VecIndex::decode(&bytes))) {
                Ok(Ok(index)) => self.vec_index = Some(index),
                Ok(Err(_)) | Err(_) => {
                    self.vec_index = None;
                    // Don't disable vec if decoding fails - keep it enabled
                    // CRITICAL: Don't modify self.toc during read-only operations!
                    // If dirty=true and Drop commits, it will corrupt the manifest.
                    // self.vec_enabled = false;
                }
            }
        } else {
            self.vec_index = None;
        }
        Ok(())
    }

    /// Load CLIP index from manifest.
    pub(crate) fn load_clip_index_from_manifest(&mut self) -> Result<()> {
        use crate::clip::ClipIndex;

        if let Some(manifest) = &self.toc.indexes.clip {
            // Empty manifest (placeholder for enabled but not yet populated index)
            if manifest.bytes_length == 0 {
                self.clip_index = None;
                return Ok(());
            }

            let bytes =
                if let Ok(bytes) = self.read_range(manifest.bytes_offset, manifest.bytes_length) {
                    bytes
                } else {
                    self.clip_index = None;
                    return Ok(());
                };
            match catch_unwind(AssertUnwindSafe(|| ClipIndex::decode(&bytes))) {
                Ok(Ok(index)) => self.clip_index = Some(index),
                Ok(Err(_)) | Err(_) => {
                    self.clip_index = None;
                }
            }
        } else {
            self.clip_index = None;
        }
        Ok(())
    }

    pub fn read_range(&mut self, offset: u64, length: u64) -> Result<Vec<u8>> {
        let file_len = self.file.metadata()?.len();
        let end = offset.checked_add(length).ok_or(MemvidError::InvalidToc {
            reason: "manifest range overflow".into(),
        })?;
        if end > file_len || length > crate::MAX_INDEX_BYTES {
            return Err(MemvidError::InvalidToc {
                reason: "manifest range invalid".into(),
            });
        }
        self.file.seek(SeekFrom::Start(offset))?;
        // Safe: length is checked against MAX_INDEX_BYTES above
        #[allow(clippy::cast_possible_truncation)]
        let mut buf = vec![0u8; length as usize];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    #[allow(dead_code)]
    fn disable_lex(&mut self) {
        self.lex_index = None;
        self.lex_enabled = false;
        self.toc.indexes.lex = None;
    }

    fn build_vec_index_from_segments(&mut self) -> Result<()> {
        use crate::vec::VecIndexBuilder;

        let mut builder = VecIndexBuilder::new();

        // Clone segments to avoid borrow checker issues
        let segments = self.toc.segment_catalog.vec_segments.clone();

        for segment_desc in &segments {
            let bytes = match self.read_range(
                segment_desc.common.bytes_offset,
                segment_desc.common.bytes_length,
            ) {
                Ok(bytes) => bytes,
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        segment_id = segment_desc.common.segment_id,
                        "failed to load vec segment, skipping"
                    );
                    continue;
                }
            };

            // Use the compression stored in the descriptor - it's already correct
            // The descriptor reflects the actual encoding used when the segment was written
            let compression_hint = segment_desc.vector_compression.clone();

            tracing::debug!(
                segment_id = segment_desc.common.segment_id,
                compression_hint = ?compression_hint,
                bytes_len = bytes.len(),
                "attempting to decode vec segment"
            );

            match VecIndex::decode_with_compression(&bytes, compression_hint) {
                Ok(segment_index) => {
                    for (frame_id, embedding) in segment_index.entries() {
                        if self.frame_is_active(frame_id) {
                            builder.add_document(frame_id, embedding.to_vec());
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        segment_id = segment_desc.common.segment_id,
                        "failed to decode vec segment, skipping"
                    );
                }
            }
        }

        let artifact = builder.finish()?;
        if artifact.vector_count > 0 {
            let index =
                VecIndex::decode_with_compression(&artifact.bytes, VectorCompression::None)?;
            self.vec_index = Some(index);
        }

        Ok(())
    }

    fn hydrate_lex_index_metadata(&self, index: &mut LexIndex) {
        for document in index.documents_mut() {
            let frame_idx = usize::try_from(document.frame_id).ok();
            let frame_meta = frame_idx.and_then(|idx| self.toc.frames.get(idx));

            if document.uri.is_none() {
                let derived = frame_meta
                    .and_then(|frame| frame.uri.clone())
                    .unwrap_or_else(|| crate::default_uri(document.frame_id));
                document.uri = Some(derived);
            }

            if document.title.is_none() {
                let title = frame_meta
                    .and_then(|frame| frame.title.clone())
                    .or_else(|| {
                        document
                            .uri
                            .as_ref()
                            .and_then(|uri| crate::infer_title_from_uri(uri))
                    });
                document.title = title;
            }
        }
    }
}

use crate::vec::VecIndexBuilder;
