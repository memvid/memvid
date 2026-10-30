//! Frame payload and preview helpers for `Memvid`.

use std::fs::File;
use std::io::{self, Cursor, Read, Seek, SeekFrom};

use crate::error::{MemvidError, Result};
use crate::memvid::lifecycle::Memvid;
use crate::types::{CanonicalEncoding, Frame, FrameId, FrameRole, FrameStatus, MediaManifest};

#[derive(Debug, Clone)]
pub(crate) struct ChunkInfo {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Streaming reader over canonical frame bytes. For binary payloads (e.g., video) the reader
/// clones the underlying file handle to avoid disturbing `Memvid`'s primary cursor.
pub struct BlobReader {
    inner: BlobReaderInner,
}

enum BlobReaderInner {
    File {
        file: File,
        start: u64,
        len: u64,
        pos: u64,
    },
    Memory(Cursor<Vec<u8>>),
}

impl BlobReader {
    fn from_file(file: File, start: u64, len: u64) -> Self {
        Self {
            inner: BlobReaderInner::File {
                file,
                start,
                len,
                pos: 0,
            },
        }
    }

    fn from_memory(bytes: Vec<u8>) -> Self {
        Self {
            inner: BlobReaderInner::Memory(Cursor::new(bytes)),
        }
    }

    #[must_use]
    pub fn len(&self) -> u64 {
        match &self.inner {
            BlobReaderInner::File { len, .. } => *len,
            BlobReaderInner::Memory(cursor) => cursor.get_ref().len() as u64,
        }
    }
}

impl Read for BlobReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match &mut self.inner {
            BlobReaderInner::File {
                file,
                start,
                len,
                pos,
            } => {
                let remaining = len.saturating_sub(*pos);
                if remaining == 0 {
                    return Ok(0);
                }
                #[allow(clippy::cast_possible_truncation)]
                let to_read = remaining.min(buf.len() as u64) as usize;
                file.seek(SeekFrom::Start(*start + *pos))?;
                let read = file.read(&mut buf[..to_read])?;
                *pos += read as u64;
                Ok(read)
            }
            BlobReaderInner::Memory(cursor) => cursor.read(buf),
        }
    }
}

impl Seek for BlobReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        match &mut self.inner {
            BlobReaderInner::File {
                file,
                start,
                len,
                pos,
            } => {
                let absolute = match position {
                    SeekFrom::Start(offset) => offset,
                    SeekFrom::End(delta) => {
                        let end = *len as i64;
                        let result = end.checked_add(delta).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidInput, "seek overflow")
                        })?;
                        if result < 0 {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                "seek before start",
                            ));
                        }
                        result as u64
                    }
                    SeekFrom::Current(delta) => {
                        let current = *pos as i64;
                        let result = current.checked_add(delta).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidInput, "seek overflow")
                        })?;
                        if result < 0 {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                "seek before start",
                            ));
                        }
                        result as u64
                    }
                };

                if absolute > *len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "seek beyond end",
                    ));
                }

                *pos = absolute;
                file.seek(SeekFrom::Start(*start + *pos))?;
                Ok(*pos)
            }
            BlobReaderInner::Memory(cursor) => cursor.seek(position),
        }
    }
}

fn mime_is_text(mime: &str) -> bool {
    let normalized = mime
        .split(';')
        .next()
        .unwrap_or(mime)
        .trim()
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        mime if mime.starts_with("text/")
            || mime == "application/json"
            || mime == "application/xml"
            || mime == "application/javascript"
            || mime == "application/xhtml+xml"
            || mime == "application/rss+xml"
            || mime == "application/rtf"
            || mime == "application/toml"
            || mime == "application/yaml"
            || mime == "application/x-yaml"
            || mime == "application/x-toml"
    )
}

impl Memvid {
    pub fn frame_by_id(&self, frame_id: FrameId) -> Result<Frame> {
        let index =
            usize::try_from(frame_id).map_err(|_| MemvidError::FrameNotFound { frame_id })?;
        self.toc
            .frames
            .get(index)
            .cloned()
            .ok_or(MemvidError::FrameNotFound { frame_id })
    }

    pub fn frame_by_uri(&self, uri: &str) -> Result<Frame> {
        let candidate = self
            .toc
            .frames
            .iter()
            .rev()
            .find(|frame| {
                frame
                    .uri
                    .as_deref()
                    .is_some_and(|candidate_uri| candidate_uri == uri)
                    && frame.status == FrameStatus::Active
            })
            .or_else(|| {
                self.toc
                    .frames
                    .iter()
                    .rev()
                    .find(|frame| frame.uri.as_deref() == Some(uri))
            })
            .cloned();

        candidate.ok_or_else(|| MemvidError::FrameNotFoundByUri {
            uri: uri.to_string(),
        })
    }

    /// Find an active frame by its content BLAKE3 hash.
    ///
    /// This is used for deduplication - if a frame with the same content hash already exists,
    /// we can skip re-ingestion. The hash is computed from the original file bytes.
    ///
    /// Returns `None` if no matching frame is found.
    #[must_use]
    pub fn find_frame_by_hash(&self, hash: &[u8; 32]) -> Option<&Frame> {
        self.toc
            .frames
            .iter()
            .rev()
            .find(|frame| frame.status == FrameStatus::Active && frame.checksum == *hash)
    }

    pub fn blob_reader(&mut self, frame_id: FrameId) -> Result<BlobReader> {
        let frame = self.frame_by_id(frame_id)?;
        self.blob_reader_from_frame(frame)
    }

    pub fn blob_reader_by_uri(&mut self, uri: &str) -> Result<BlobReader> {
        let frame = self.frame_by_uri(uri)?;
        self.blob_reader_from_frame(frame)
    }

    pub fn media_manifest(&self, frame_id: FrameId) -> Result<Option<MediaManifest>> {
        let frame = self.frame_by_id(frame_id)?;
        Ok(frame.metadata.and_then(|meta| meta.media))
    }

    pub fn media_manifest_by_uri(&self, uri: &str) -> Result<Option<MediaManifest>> {
        let frame = self.frame_by_uri(uri)?;
        Ok(frame.metadata.and_then(|meta| meta.media))
    }

    fn blob_reader_from_frame(&mut self, frame: Frame) -> Result<BlobReader> {
        match frame.canonical_encoding {
            CanonicalEncoding::Plain => {
                let mut file = self.file.try_clone()?;
                file.seek(SeekFrom::Start(frame.payload_offset))?;
                Ok(BlobReader::from_file(
                    file,
                    frame.payload_offset,
                    frame.payload_length,
                ))
            }
            CanonicalEncoding::Zstd => {
                let bytes = self.frame_canonical_bytes(&frame)?;
                Ok(BlobReader::from_memory(bytes))
            }
        }
    }

    pub fn frame_canonical_payload(&mut self, frame_id: FrameId) -> Result<Vec<u8>> {
        let frame = self.frame_by_id(frame_id)?;
        self.frame_canonical_bytes(&frame)
    }

    pub fn frame_preview_by_id(&mut self, frame_id: FrameId) -> Result<String> {
        let index = usize::try_from(frame_id).map_err(|_| MemvidError::InvalidTimeIndex {
            reason: "frame id too large".into(),
        })?;
        let frame = self
            .toc
            .frames
            .get(index)
            .cloned()
            .ok_or(MemvidError::InvalidTimeIndex {
                reason: "frame id out of range".into(),
            })?;
        self.frame_preview(&frame)
    }

    /// Get the full text content of a frame (no truncation).
    ///
    /// Unlike `frame_preview_by_id` which truncates for display purposes,
    /// this returns the complete text content suitable for LLM processing.
    pub fn frame_text_by_id(&mut self, frame_id: FrameId) -> Result<String> {
        let index = usize::try_from(frame_id).map_err(|_| MemvidError::InvalidTimeIndex {
            reason: "frame id too large".into(),
        })?;
        let frame = self
            .toc
            .frames
            .get(index)
            .cloned()
            .ok_or(MemvidError::InvalidTimeIndex {
                reason: "frame id out of range".into(),
            })?;
        self.frame_content(&frame)
    }

    pub(crate) fn frame_preview(&mut self, frame: &Frame) -> Result<String> {
        if let Some(text) = frame
            .metadata
            .as_ref()
            .and_then(crate::image_preview_from_metadata)
        {
            return Ok(text);
        }

        if let Some(video) = frame
            .metadata
            .as_ref()
            .and_then(|meta| meta.media.as_ref())
            .filter(|media| media.kind.eq_ignore_ascii_case("video"))
        {
            let mut segments = Vec::new();
            let label = video
                .filename
                .as_deref()
                .or(frame.title.as_deref())
                .or(frame.uri.as_deref())
                .unwrap_or("video");
            segments.push(format!("Video: {label}"));
            if let Some(duration) = video.duration_ms {
                let seconds = (duration as f64) / 1000.0;
                segments.push(format!("{seconds:.1}s"));
            }
            if let (Some(width), Some(height)) = (video.width, video.height) {
                segments.push(format!("{width}x{height}"));
            }
            if let Some(codec) = &video.codec {
                if !codec.trim().is_empty() {
                    segments.push(codec.clone());
                }
            }
            return Ok(segments.join(" · "));
        }

        if let Some(search) = &frame.search_text {
            return Ok(crate::truncate_preview(search));
        }
        if frame.payload_length == 0 {
            return Ok(String::new());
        }
        match self.frame_canonical_text(frame) {
            Ok(text) => Ok(crate::truncate_preview(&text)),
            Err(_) => Ok("<invalid frame>".into()),
        }
    }

    pub(crate) fn frame_content(&mut self, frame: &Frame) -> Result<String> {
        // Check search_text first - this handles no_raw mode where payload is empty
        // but search_text contains the indexed content
        if let Some(search) = &frame.search_text {
            if !search.is_empty() {
                return Ok(search.clone());
            }
        }
        if frame.payload_length == 0 && frame.chunk_manifest.is_none() {
            return Ok(String::new());
        }
        self.frame_canonical_text(frame)
    }

    pub fn frame_embedding(&mut self, frame_id: FrameId) -> Result<Option<Vec<f32>>> {
        if !self.vec_enabled {
            return Ok(None);
        }
        self.ensure_vec_index()?;
        Ok(self
            .vec_index
            .as_ref()
            .and_then(|index| index.embedding_for(frame_id).map(<[f32]>::to_vec)))
    }

    pub fn frame_context(&mut self, frame_id: FrameId, query: &str) -> Result<(String, usize)> {
        let index = usize::try_from(frame_id).map_err(|_| MemvidError::InvalidTimeIndex {
            reason: "frame id too large".into(),
        })?;
        let frame = self
            .toc
            .frames
            .get(index)
            .cloned()
            .ok_or(MemvidError::InvalidTimeIndex {
                reason: "frame id out of range".into(),
            })?;
        let preview = self.frame_preview(&frame)?;
        let content = self.frame_content(&frame)?;
        let lowered = content.to_lowercase();
        let count = query
            .split_whitespace()
            .filter(|token| !token.is_empty())
            .map(str::to_lowercase)
            .map(|needle| lowered.matches(&needle).count())
            .sum();
        Ok((preview, count))
    }

    pub(crate) fn frame_canonical_bytes(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        if frame.role == FrameRole::Document && frame.chunk_manifest.is_some() {
            let chunks = self.document_chunk_payloads(frame)?;
            let mut buffer = Vec::new();
            for (_, bytes) in chunks {
                buffer.extend_from_slice(&bytes);
            }
            return Ok(buffer);
        }
        let raw = self.read_frame_payload_bytes(frame)?;
        let decoded = crate::decode_canonical_bytes(&raw, frame.canonical_encoding, frame.id)?;
        if let Some(expected) = frame.canonical_length {
            if decoded.len() as u64 != expected {
                return Err(MemvidError::InvalidFrame {
                    frame_id: frame.id,
                    reason: "canonical length mismatch",
                });
            }
        }
        Ok(decoded)
    }

    pub(crate) fn frame_canonical_text(&mut self, frame: &Frame) -> Result<String> {
        if frame.role == FrameRole::Document && frame.chunk_manifest.is_some() {
            let bytes = self.frame_canonical_bytes(frame)?;
            return match String::from_utf8(bytes) {
                Ok(text) => Ok(text),
                Err(err) => {
                    let bytes = err.into_bytes();
                    Ok(Self::render_binary_summary(bytes.len()))
                }
            };
        }

        if let Some(search) = &frame.search_text {
            return Ok(search.clone());
        }

        if let Some(meta) = frame
            .metadata
            .as_ref()
            .and_then(|meta| meta.mime.as_deref())
        {
            if !mime_is_text(meta) {
                let logical = frame
                    .canonical_length
                    .or(Some(frame.payload_length))
                    .unwrap_or(frame.payload_length);
                #[allow(clippy::cast_possible_truncation)]
                return Ok(Self::render_binary_summary(logical as usize));
            }
        }

        let bytes = self.frame_canonical_bytes(frame)?;
        match String::from_utf8(bytes) {
            Ok(text) => Ok(text),
            Err(err) => {
                let bytes = err.into_bytes();
                Ok(Self::render_binary_summary(bytes.len()))
            }
        }
    }

    pub(crate) fn document_chunk_payloads(
        &mut self,
        frame: &Frame,
    ) -> Result<Vec<(Frame, Vec<u8>)>> {
        let Some(manifest) = frame.chunk_manifest.as_ref() else {
            return Err(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "document missing chunk manifest",
            });
        };
        let mut children = self.document_chunk_frames(frame.id);
        if children.is_empty() {
            return Err(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "document chunk manifest missing children",
            });
        }
        if children.len() != manifest.chunks.len() {
            return Err(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "chunk manifest length mismatch",
            });
        }
        children.sort_by_key(|child| (child.chunk_index.unwrap_or(u32::MAX), child.id));
        let mut payloads = Vec::with_capacity(children.len());
        for child in children {
            let raw = self.read_frame_payload_bytes(&child)?;
            let decoded = crate::decode_canonical_bytes(&raw, child.canonical_encoding, child.id)?;
            if let Some(expected) = child.canonical_length {
                if decoded.len() as u64 != expected {
                    return Err(MemvidError::InvalidFrame {
                        frame_id: child.id,
                        reason: "chunk canonical length mismatch",
                    });
                }
            }
            payloads.push((child, decoded));
        }
        Ok(payloads)
    }

    fn document_chunk_frames(&self, parent_id: FrameId) -> Vec<Frame> {
        let mut frames: Vec<Frame> = self
            .toc
            .frames
            .iter()
            .filter(|candidate| {
                candidate.status == FrameStatus::Active
                    && candidate.role == FrameRole::DocumentChunk
                    && candidate.parent_id == Some(parent_id)
            })
            .cloned()
            .collect();
        frames.sort_by_key(|frame| (frame.chunk_index.unwrap_or(u32::MAX), frame.id));
        frames
    }

    pub(crate) fn resolve_chunk_context(&mut self, frame: &Frame) -> Result<ChunkInfo> {
        match frame.role {
            FrameRole::Document => {
                if frame.chunk_manifest.is_some() {
                    let payloads = self.document_chunk_payloads(frame)?;
                    if payloads.is_empty() {
                        return Err(MemvidError::InvalidFrame {
                            frame_id: frame.id,
                            reason: "document chunk manifest missing payloads",
                        });
                    }
                    let (_, bytes) = match payloads.into_iter().next() {
                        Some(entry) => entry,
                        None => {
                            return Err(MemvidError::InvalidFrame {
                                frame_id: frame.id,
                                reason: "document chunk manifest missing payloads",
                            });
                        }
                    };
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    let end = bytes.len();
                    Ok(ChunkInfo {
                        start: 0,
                        end,
                        text,
                    })
                } else if let Some(search_text) = frame.search_text.clone() {
                    let end = search_text.len();
                    Ok(ChunkInfo {
                        start: 0,
                        end,
                        text: search_text,
                    })
                } else {
                    let bytes = self.frame_canonical_bytes(frame)?;
                    let end = bytes.len();
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    Ok(ChunkInfo {
                        start: 0,
                        end,
                        text,
                    })
                }
            }
            FrameRole::DocumentChunk => {
                // Try to resolve via parent's chunk manifest (new format)
                if let Some(parent_id) = frame.parent_id {
                    // Safe frame lookup
                    if let Ok(index) = usize::try_from(parent_id) {
                        if let Some(parent) = self.toc.frames.get(index).cloned() {
                            if parent.chunk_manifest.is_some() {
                                if let Ok(payloads) = self.document_chunk_payloads(&parent) {
                                    if let Some(idx) = frame.chunk_index {
                                        let idx = idx as usize;
                                        if idx < payloads.len() {
                                            let mut offset = 0usize;
                                            for (_, bytes) in payloads.iter().take(idx) {
                                                offset += bytes.len();
                                            }
                                            let (_, bytes) = &payloads[idx];
                                            let text = String::from_utf8_lossy(bytes).into_owned();
                                            let end = offset + bytes.len();
                                            return Ok(ChunkInfo {
                                                start: offset,
                                                end,
                                                text,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Fallback for legacy chunks: use search_text or raw bytes directly
                if let Some(search_text) = frame.search_text.clone() {
                    let end = search_text.len();
                    Ok(ChunkInfo {
                        start: 0,
                        end,
                        text: search_text,
                    })
                } else {
                    let bytes = self.frame_canonical_bytes(frame)?;
                    let end = bytes.len();
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    Ok(ChunkInfo {
                        start: 0,
                        end,
                        text,
                    })
                }
            }
            FrameRole::ExtractedImage => {
                // Extracted images are binary - return empty text info
                // They are meant to be viewed as images, not as text
                Ok(ChunkInfo {
                    start: 0,
                    end: 0,
                    text: String::new(),
                })
            }
        }
    }

    pub(crate) fn read_frame_payload_bytes(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        self.validate_frame_bounds(frame)?;
        self.file.seek(SeekFrom::Start(frame.payload_offset))?;
        // Safe: guarded by MAX_FRAME_BYTES check
        #[allow(clippy::cast_possible_truncation)]
        let mut buf = vec![0u8; frame.payload_length as usize];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    pub(crate) fn validate_frame_bounds(&mut self, frame: &Frame) -> Result<()> {
        if frame.payload_length == 0 {
            return Ok(());
        }

        if frame.payload_length > crate::MAX_FRAME_BYTES {
            return Err(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "payload length exceeds maximum",
            });
        }

        let wal_region_end = self
            .header
            .wal_offset
            .checked_add(self.header.wal_size)
            .ok_or(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "wal region overflow",
            })?;

        if frame.payload_offset < wal_region_end {
            return Err(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "payload overlaps wal region",
            });
        }

        let frame_end = frame
            .payload_offset
            .checked_add(frame.payload_length)
            .ok_or(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "payload range overflow",
            })?;

        if frame_end > self.data_end {
            return Err(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "payload extends past data region",
            });
        }

        let file_len = self.file.metadata()?.len();
        if frame_end > file_len {
            return Err(MemvidError::InvalidFrame {
                frame_id: frame.id,
                reason: "payload extends past file length",
            });
        }

        Ok(())
    }
}
