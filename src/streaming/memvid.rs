//! Read-only streaming handle for `.mv2` files.

#![allow(clippy::cast_possible_truncation)]

use std::io::Cursor;

#[cfg(feature = "lex")]
use std::sync::OnceLock;

use crate::constants::HEADER_SIZE;
use crate::footer::{CommitFooter, FOOTER_SIZE};
use crate::io::header::HeaderCodec;
use crate::types::{
    CanonicalEncoding, Frame, FrameId, FrameStatus, Header, SearchEngineKind, SearchHit,
    SearchHitMetadata, SearchParams, SearchRequest, SearchResponse, Stats, Tier, TimelineEntry,
    TimelineQuery, Toc,
};

use super::error::StreamingError;
use super::source::{StreamingResult, StreamingSource};

/// Read-only streaming handle for `.mv2` files.
///
/// This struct provides read-only access to `.mv2` files hosted on CDNs or local storage
/// without requiring the entire file to be downloaded. It uses HTTP range requests (or
/// equivalent) to fetch only the data needed for each operation.
///
/// # Lazy Loading
///
/// Indexes are loaded lazily on first use:
/// - The header, footer, and TOC are loaded when `open()` is called
/// - The lexical index is loaded on the first `search()` call
/// - The vector index is loaded on the first `search_vec()` call
/// - Frame content is fetched on demand via `frame_content()`
///
/// # Thread Safety
///
/// The streaming source must be `Send + Sync`. Index caches are stored in `OnceLock`
/// for thread-safe lazy initialization.
pub struct StreamingMemvid<S: StreamingSource> {
    source: S,
    header: Header,
    toc: Toc,
    file_size: u64,
    /// Cache for lazy-loaded lex index bytes (future Tantivy integration)
    #[cfg(feature = "lex")]
    #[allow(dead_code)]
    lex_index_bytes: OnceLock<Vec<u8>>,
}

impl<S: StreamingSource> StreamingMemvid<S> {
    /// Opens a streaming connection to a `.mv2` file.
    ///
    /// This fetches the header (4KB), footer (56 bytes), and TOC from the source.
    /// The rest of the file is fetched on demand.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source cannot be read
    /// - The header is invalid or the file is not a valid `.mv2` file
    /// - The footer is missing or corrupted
    /// - The TOC cannot be decoded or has an invalid checksum
    pub fn open(source: S) -> StreamingResult<Self> {
        let file_size = source.total_size()?;

        // 1. Read header (first 4KB)
        let header_bytes = source.read_range(0, HEADER_SIZE as u64)?;
        if header_bytes.len() < HEADER_SIZE {
            return Err(StreamingError::UnexpectedEndOfData {
                expected: HEADER_SIZE as u64,
                actual: header_bytes.len() as u64,
            });
        }

        let header_array: [u8; HEADER_SIZE] =
            header_bytes
                .try_into()
                .map_err(|_| StreamingError::InvalidHeader {
                    reason: "header too short".into(),
                })?;

        let header =
            HeaderCodec::decode(&header_array).map_err(|e| StreamingError::InvalidHeader {
                reason: format!("{e}").into(),
            })?;

        // Check if file might be encrypted (magic check)
        if header.magic == *b"MV2E" {
            return Err(StreamingError::EncryptedFile);
        }

        // 2. Read footer (last 56 bytes)
        if file_size < FOOTER_SIZE as u64 {
            return Err(StreamingError::InvalidFooter {
                reason: "file too small for footer".into(),
            });
        }

        let footer_offset = file_size - FOOTER_SIZE as u64;
        let footer_bytes = source.read_range(footer_offset, FOOTER_SIZE as u64)?;
        if footer_bytes.len() < FOOTER_SIZE {
            return Err(StreamingError::UnexpectedEndOfData {
                expected: FOOTER_SIZE as u64,
                actual: footer_bytes.len() as u64,
            });
        }

        let footer =
            CommitFooter::decode(&footer_bytes).ok_or_else(|| StreamingError::InvalidFooter {
                reason: "failed to decode footer".into(),
            })?;

        // 3. Read TOC
        let toc_len = footer.toc_len;
        if toc_len == 0 || toc_len > footer_offset {
            return Err(StreamingError::InvalidToc {
                reason: format!("invalid TOC length: {toc_len}").into(),
            });
        }

        let toc_offset = footer_offset - toc_len;
        let toc_bytes = source.read_range(toc_offset, toc_len)?;

        // Verify TOC checksum
        if !footer.hash_matches(&toc_bytes) {
            return Err(StreamingError::ChecksumMismatch { context: "toc" });
        }

        // Decode TOC
        let toc = Toc::decode(&toc_bytes).map_err(|e| StreamingError::InvalidToc {
            reason: format!("{e}").into(),
        })?;

        Ok(Self {
            source,
            header,
            toc,
            file_size,
            #[cfg(feature = "lex")]
            lex_index_bytes: OnceLock::new(),
        })
    }

    /// Returns the source identifier (URL or path).
    #[must_use]
    pub fn source_id(&self) -> &str {
        self.source.source_id()
    }

    /// Returns the total file size in bytes.
    #[must_use]
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Returns statistics about the memory.
    #[must_use]
    pub fn stats(&self) -> Stats {
        let frame_count = self.toc.frames.len() as u64;
        let active_count = self
            .toc
            .frames
            .iter()
            .filter(|f| matches!(f.status, FrameStatus::Active))
            .count() as u64;

        let payload_bytes: u64 = self.toc.frames.iter().map(|f| f.payload_length).sum();

        Stats {
            frame_count,
            size_bytes: self.file_size,
            tier: Tier::Free,
            has_lex_index: self.toc.indexes.lex.is_some(),
            has_vec_index: self.toc.indexes.vec.is_some(),
            has_clip_index: self.toc.indexes.clip.is_some(),
            has_time_index: self.toc.time_index.is_some(),
            seq_no: None,
            capacity_bytes: 0,
            active_frame_count: active_count,
            payload_bytes,
            logical_bytes: payload_bytes,
            saved_bytes: 0,
            compression_ratio_percent: 100.0,
            savings_percent: 0.0,
            storage_utilisation_percent: 0.0,
            remaining_capacity_bytes: 0,
            average_frame_payload_bytes: if frame_count > 0 {
                payload_bytes / frame_count
            } else {
                0
            },
            average_frame_logical_bytes: if frame_count > 0 {
                payload_bytes / frame_count
            } else {
                0
            },
            wal_bytes: self.header.wal_size,
            lex_index_bytes: self.toc.indexes.lex.as_ref().map_or(0, |m| m.bytes_length),
            vec_index_bytes: self.toc.indexes.vec.as_ref().map_or(0, |m| m.bytes_length),
            time_index_bytes: self.toc.time_index.as_ref().map_or(0, |m| m.bytes_length),
            vector_count: self.toc.indexes.vec.as_ref().map_or(0, |m| m.vector_count),
            clip_image_count: self.toc.indexes.clip.as_ref().map_or(0, |m| m.vector_count),
        }
    }

    /// Returns a reference to the Table of Contents.
    #[must_use]
    pub fn toc(&self) -> &Toc {
        &self.toc
    }

    /// Returns a reference to the header.
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Returns the frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.toc.frames.len() as u64
    }

    /// Returns a frame by ID.
    pub fn frame_by_id(&self, frame_id: FrameId) -> StreamingResult<&Frame> {
        self.toc
            .frames
            .get(frame_id as usize)
            .ok_or(StreamingError::FrameNotFound { frame_id })
    }

    /// Returns a frame by URI.
    pub fn frame_by_uri(&self, uri: &str) -> StreamingResult<&Frame> {
        self.toc
            .frames
            .iter()
            .find(|f| f.uri.as_deref() == Some(uri))
            .ok_or_else(|| StreamingError::FrameNotFoundByUri {
                uri: uri.to_string(),
            })
    }

    /// Returns the canonical text content of a frame.
    ///
    /// This fetches the frame payload from the source and decompresses it if needed.
    pub fn frame_content(&self, frame: &Frame) -> StreamingResult<String> {
        // Check search_text first (handles no_raw mode)
        if let Some(search) = &frame.search_text {
            if !search.is_empty() {
                return Ok(search.clone());
            }
        }

        if frame.payload_length == 0 {
            return Ok(String::new());
        }

        let payload = self
            .source
            .read_range(frame.payload_offset, frame.payload_length)?;

        let canonical = match frame.canonical_encoding {
            CanonicalEncoding::Plain => payload,
            CanonicalEncoding::Zstd => {
                zstd::decode_all(Cursor::new(&payload)).map_err(|_| StreamingError::InvalidToc {
                    reason: format!("failed to decompress frame {}", frame.id).into(),
                })?
            }
        };

        String::from_utf8(canonical).map_err(|_| StreamingError::InvalidToc {
            reason: format!("frame {} is not valid UTF-8", frame.id).into(),
        })
    }

    /// Returns the raw bytes of a frame payload.
    pub fn frame_bytes(&self, frame: &Frame) -> StreamingResult<Vec<u8>> {
        if frame.payload_length == 0 {
            return Ok(Vec::new());
        }

        let payload = self
            .source
            .read_range(frame.payload_offset, frame.payload_length)?;

        match frame.canonical_encoding {
            CanonicalEncoding::Plain => Ok(payload),
            CanonicalEncoding::Zstd => {
                zstd::decode_all(Cursor::new(&payload)).map_err(|_| StreamingError::InvalidToc {
                    reason: format!("failed to decompress frame {}", frame.id).into(),
                })
            }
        }
    }

    /// Returns a timeline of frames, optionally filtered and paginated.
    pub fn timeline(&self, query: TimelineQuery) -> StreamingResult<Vec<TimelineEntry>> {
        let TimelineQuery {
            limit,
            since,
            until,
            reverse,
            #[cfg(feature = "temporal_track")]
                temporal: _,
        } = query;

        let mut entries: Vec<TimelineEntry> = self
            .toc
            .frames
            .iter()
            .filter(|f| matches!(f.status, FrameStatus::Active))
            .filter(|f| since.is_none_or(|ts| f.timestamp >= ts))
            .filter(|f| until.is_none_or(|ts| f.timestamp <= ts))
            .map(|f| {
                let preview = f
                    .search_text
                    .as_ref()
                    .map(|s| s.chars().take(120).collect())
                    .unwrap_or_default();

                TimelineEntry {
                    frame_id: f.id,
                    timestamp: f.timestamp,
                    preview,
                    uri: f.uri.clone(),
                    child_frames: Vec::new(),
                    #[cfg(feature = "temporal_track")]
                    temporal: None,
                }
            })
            .collect();

        if reverse {
            entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        } else {
            entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        }

        if let Some(limit) = limit {
            entries.truncate(limit.get() as usize);
        }

        Ok(entries)
    }

    /// Performs a lexical search over the indexed content.
    ///
    /// This method provides a simplified search implementation that works without
    /// loading the full Tantivy index. It scans the TOC frames and performs
    /// basic text matching.
    ///
    /// For full Tantivy-powered search, consider downloading the file locally
    /// and using the standard `Memvid::search()` method.
    #[cfg(feature = "lex")]
    pub fn search(&self, request: SearchRequest) -> StreamingResult<SearchResponse> {
        use std::time::Instant;

        let start = Instant::now();
        let query_lower = request.query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        if query_terms.is_empty() {
            return Ok(SearchResponse {
                query: request.query,
                elapsed_ms: start.elapsed().as_millis(),
                total_hits: 0,
                params: SearchParams {
                    top_k: request.top_k,
                    snippet_chars: request.snippet_chars,
                    cursor: request.cursor,
                },
                hits: Vec::new(),
                context: String::new(),
                next_cursor: None,
                engine: SearchEngineKind::LexFallback,
            });
        }

        // Simple scan-based search over frames
        let mut hits: Vec<SearchHit> = Vec::new();

        for frame in &self.toc.frames {
            if !matches!(frame.status, FrameStatus::Active) {
                continue;
            }

            // Apply URI filter
            if let Some(ref uri_filter) = request.uri {
                if frame.uri.as_deref() != Some(uri_filter.as_str()) {
                    continue;
                }
            }

            // Apply scope filter
            if let Some(ref scope) = request.scope {
                if !frame
                    .uri
                    .as_ref()
                    .is_some_and(|u| u.starts_with(scope.as_str()))
                {
                    continue;
                }
            }

            // Get searchable text
            let text = frame.search_text.clone().unwrap_or_default();

            if text.is_empty() {
                continue;
            }

            let text_lower = text.to_lowercase();

            // Count matching terms
            let match_count = query_terms
                .iter()
                .filter(|term| text_lower.contains(*term))
                .count();

            if match_count == 0 {
                continue;
            }

            // Find snippet around first match
            let (snippet, range) = if let Some(pos) = query_terms
                .iter()
                .filter_map(|term| text_lower.find(term))
                .min()
            {
                let half_snippet = request.snippet_chars / 2;
                let start = pos.saturating_sub(half_snippet);
                let end = (pos + half_snippet).min(text.len());
                let snippet = text[start..end].to_string();
                ((start, end), snippet)
            } else {
                let end = request.snippet_chars.min(text.len());
                ((0, end), text[..end].to_string())
            };

            hits.push(SearchHit {
                rank: 0,
                frame_id: frame.id,
                uri: frame.uri.clone().unwrap_or_default(),
                title: frame.title.clone(),
                range: snippet,
                text: range,
                matches: match_count,
                chunk_range: None,
                chunk_text: None,
                score: Some(match_count as f32 / query_terms.len() as f32),
                metadata: Some(SearchHitMetadata {
                    matches: match_count,
                    tags: frame.tags.clone(),
                    labels: frame.labels.clone(),
                    track: frame.track.clone(),
                    created_at: None,
                    content_dates: frame.content_dates.clone(),
                    entities: Vec::new(),
                    extra_metadata: frame.extra_metadata.clone(),
                    #[cfg(feature = "temporal_track")]
                    temporal: None,
                }),
            });
        }

        // Sort by score descending
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply pagination
        let total_hits = hits.len();
        hits.truncate(request.top_k);

        // Assign ranks
        for (i, hit) in hits.iter_mut().enumerate() {
            hit.rank = i + 1;
        }

        // Build context
        let context = hits
            .iter()
            .map(|h| h.text.clone())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        Ok(SearchResponse {
            query: request.query,
            elapsed_ms: start.elapsed().as_millis(),
            total_hits,
            params: SearchParams {
                top_k: request.top_k,
                snippet_chars: request.snippet_chars,
                cursor: request.cursor,
            },
            hits,
            context,
            next_cursor: None,
            engine: SearchEngineKind::LexFallback,
        })
    }

    /// Performs a lexical search (stub when lex feature is disabled).
    #[cfg(not(feature = "lex"))]
    pub fn search(&self, _request: SearchRequest) -> StreamingResult<SearchResponse> {
        Err(StreamingError::FeatureNotEnabled { feature: "lex" })
    }
}

impl<S: StreamingSource> std::fmt::Debug for StreamingMemvid<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingMemvid")
            .field("source_id", &self.source.source_id())
            .field("file_size", &self.file_size)
            .field("frame_count", &self.toc.frames.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Memvid;
    use crate::streaming::LocalStreamingSource;
    use tempfile::tempdir;

    #[test]
    fn test_streaming_memvid_opens_local_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mv2");

        // Create a test .mv2 file
        {
            let mut mem = Memvid::create(&path).unwrap();
            mem.put_bytes(b"Hello, streaming world!").unwrap();
            mem.commit().unwrap();
        }

        // Open via streaming
        let source = LocalStreamingSource::open(&path).unwrap();
        let streaming = StreamingMemvid::open(source).unwrap();

        assert_eq!(streaming.frame_count(), 1);
        let stats = streaming.stats();
        assert_eq!(stats.frame_count, 1);
    }

    #[test]
    fn test_streaming_memvid_reads_frame_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("content.mv2");

        let content = "This is test content for streaming.";

        {
            let mut mem = Memvid::create(&path).unwrap();
            mem.put_bytes(content.as_bytes()).unwrap();
            mem.commit().unwrap();
        }

        let source = LocalStreamingSource::open(&path).unwrap();
        let streaming = StreamingMemvid::open(source).unwrap();

        let frame = streaming.frame_by_id(0).unwrap();
        let text = streaming.frame_content(frame).unwrap();
        // Content may be enriched with metadata during ingestion
        assert!(text.starts_with(content));
    }

    #[test]
    fn test_streaming_memvid_timeline() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("timeline.mv2");

        {
            let mut mem = Memvid::create(&path).unwrap();
            mem.put_bytes(b"First frame").unwrap();
            mem.put_bytes(b"Second frame").unwrap();
            mem.commit().unwrap();
        }

        let source = LocalStreamingSource::open(&path).unwrap();
        let streaming = StreamingMemvid::open(source).unwrap();

        let timeline = streaming.timeline(TimelineQuery::default()).unwrap();
        assert_eq!(timeline.len(), 2);
    }

    #[cfg(feature = "lex")]
    #[test]
    fn test_streaming_memvid_search() {
        use crate::PutOptions;

        let dir = tempdir().unwrap();
        let path = dir.path().join("search.mv2");

        {
            let mut mem = Memvid::create(&path).unwrap();
            mem.enable_lex().unwrap();
            let opts = PutOptions::builder()
                .search_text("machine learning neural networks")
                .build();
            mem.put_bytes_with_options(b"ML content", opts).unwrap();

            let opts2 = PutOptions::builder()
                .search_text("database systems query optimization")
                .build();
            mem.put_bytes_with_options(b"DB content", opts2).unwrap();
            mem.commit().unwrap();
        }

        let source = LocalStreamingSource::open(&path).unwrap();
        let streaming = StreamingMemvid::open(source).unwrap();

        let response = streaming
            .search(SearchRequest {
                query: "machine learning".into(),
                top_k: 10,
                snippet_chars: 100,
                uri: None,
                scope: None,
                cursor: None,
                #[cfg(feature = "temporal_track")]
                temporal: None,
                as_of_frame: None,
                as_of_ts: None,
                no_sketch: false,
            })
            .unwrap();

        assert_eq!(response.hits.len(), 1);
        assert!(
            response.hits[0]
                .text
                .to_lowercase()
                .contains("machine learning")
        );
    }
}
