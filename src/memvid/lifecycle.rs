//! Lifecycle management for creating and opening `.mv2` memories.
//!
//! Responsibilities:
//! - Enforce single-file invariant (no sidecars) and take OS locks.
//! - Bootstrap headers, internal WAL, and TOC on create, and recover them on open.
//! - Validate TOC/footer layout, recover the latest valid footer when needed.
//! - Wire up index state (lex/vector/time) without mutating payload bytes.

use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::panic;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::constants::{MAGIC, SPEC_VERSION, WAL_OFFSET, WAL_SIZE_TINY};
use crate::error::{MemvidError, Result};
use crate::footer::{FooterSlice, find_last_valid_footer};
use crate::io::header::HeaderCodec;
#[cfg(feature = "parallel_segments")]
use crate::io::manifest_wal::ManifestWal;
use crate::io::wal::EmbeddedWal;
use crate::lock::{FileLock, LockMode};
#[cfg(feature = "lex")]
use crate::search::{EmbeddedLexStorage, TantivyEngine};
#[cfg(feature = "temporal_track")]
use crate::types::FrameId;
#[cfg(feature = "parallel_segments")]
use crate::types::IndexSegmentRef;
use crate::types::{
    FrameStatus, Header, IndexManifests, LogicMesh, MemoriesTrack, SchemaRegistry, SegmentCatalog,
    SketchTrack, TicketRef, Tier, Toc, VectorCompression,
};
#[cfg(feature = "temporal_track")]
use crate::{TemporalTrack, temporal_track_read};
use crate::{lex::LexIndex, vec::VecIndex};
use blake3::Hasher;
use memmap2::Mmap;

const DEFAULT_LOCK_TIMEOUT_MS: u64 = 250;
const DEFAULT_HEARTBEAT_MS: u64 = 2_000;
const DEFAULT_STALE_GRACE_MS: u64 = 10_000;

/// Primary handle for interacting with a `.mv2` memory file.
///
/// Holds the file descriptor, lock, header, TOC, and in-memory index state. Mutations
/// append to the embedded WAL and are materialized at commit time to keep the layout deterministic.
pub struct Memvid {
    pub(crate) file: File,
    pub(crate) path: PathBuf,
    pub(crate) lock: FileLock,
    pub(crate) read_only: bool,
    pub(crate) header: Header,
    pub(crate) toc: Toc,
    pub(crate) wal: EmbeddedWal,
    /// Number of frame inserts appended to WAL but not yet materialized into `toc.frames`.
    ///
    /// This lets frontends predict stable frame IDs before an explicit commit.
    pub(crate) pending_frame_inserts: u64,
    pub(crate) data_end: u64,
    pub(crate) generation: u64,
    pub(crate) lock_settings: LockSettings,
    pub(crate) lex_enabled: bool,
    pub(crate) lex_index: Option<LexIndex>,
    #[cfg(feature = "lex")]
    #[allow(dead_code)]
    pub(crate) lex_storage: Arc<RwLock<EmbeddedLexStorage>>,
    pub(crate) vec_enabled: bool,
    pub(crate) vec_compression: VectorCompression,
    pub(crate) vec_model: Option<String>,
    pub(crate) vec_index: Option<VecIndex>,
    /// CLIP visual embeddings index (separate from vec due to different dimensions)
    pub(crate) clip_enabled: bool,
    pub(crate) clip_index: Option<crate::clip::ClipIndex>,
    pub(crate) dirty: bool,
    #[cfg(feature = "lex")]
    pub(crate) tantivy: Option<TantivyEngine>,
    #[cfg(feature = "lex")]
    pub(crate) tantivy_dirty: bool,
    #[cfg(feature = "temporal_track")]
    pub(crate) temporal_track: Option<TemporalTrack>,
    #[cfg(feature = "parallel_segments")]
    pub(crate) manifest_wal: Option<ManifestWal>,
    /// In-memory track for structured memory cards.
    pub(crate) memories_track: MemoriesTrack,
    /// In-memory Logic-Mesh graph for entity-relationship traversal.
    pub(crate) logic_mesh: LogicMesh,
    /// In-memory sketch track for fast candidate generation.
    pub(crate) sketch_track: SketchTrack,
    /// Schema registry for predicate validation.
    pub(crate) schema_registry: SchemaRegistry,
    /// Whether to enforce strict schema validation on card insert.
    pub(crate) schema_strict: bool,
    /// Active replay session being recorded (if any).
    #[cfg(feature = "replay")]
    pub(crate) active_session: Option<crate::replay::ActiveSession>,
    /// Completed sessions stored in memory (until persisted to file).
    #[cfg(feature = "replay")]
    pub(crate) completed_sessions: Vec<crate::replay::ReplaySession>,
}

/// Controls read-only open behaviour for `.mv2` memories.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenReadOptions {
    pub allow_repair: bool,
}

#[derive(Debug, Clone)]
pub struct LockSettings {
    pub timeout_ms: u64,
    pub heartbeat_ms: u64,
    pub stale_grace_ms: u64,
    pub force_stale: bool,
    pub command: Option<String>,
}

impl Default for LockSettings {
    fn default() -> Self {
        Self {
            timeout_ms: DEFAULT_LOCK_TIMEOUT_MS,
            heartbeat_ms: DEFAULT_HEARTBEAT_MS,
            stale_grace_ms: DEFAULT_STALE_GRACE_MS,
            force_stale: false,
            command: None,
        }
    }
}

impl Memvid {
    /// Create a new, empty `.mv2` file with an embedded WAL and empty TOC.
    /// The file is locked exclusively for the lifetime of the handle.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        ensure_single_file(path_ref)?;

        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path_ref)?;
        let (mut file, lock) = FileLock::open_and_lock(path_ref)?;

        let header = Header {
            magic: MAGIC,
            version: SPEC_VERSION,
            footer_offset: WAL_OFFSET + WAL_SIZE_TINY,
            wal_offset: WAL_OFFSET,
            wal_size: WAL_SIZE_TINY,
            wal_checkpoint_pos: 0,
            wal_sequence: 0,
            toc_checksum: [0u8; 32],
        };

        let mut toc = empty_toc();
        // If lex feature is enabled, set the catalog flag immediately
        #[cfg(feature = "lex")]
        {
            toc.segment_catalog.lex_enabled = true;
        }
        file.set_len(header.footer_offset)?;
        HeaderCodec::write(&mut file, &header)?;

        let wal = EmbeddedWal::open(&file, &header)?;
        let data_end = header.footer_offset;
        #[cfg(feature = "lex")]
        let lex_storage = Arc::new(RwLock::new(EmbeddedLexStorage::new()));
        #[cfg(feature = "parallel_segments")]
        let manifest_wal = ManifestWal::open(manifest_wal_path(path_ref))?;
        #[cfg(feature = "parallel_segments")]
        let manifest_wal_entries = manifest_wal.replay()?;

        let mut memvid = Self {
            file,
            path: path_ref.to_path_buf(),
            lock,
            read_only: false,
            header,
            toc,
            wal,
            pending_frame_inserts: 0,
            data_end,
            generation: 0,
            lock_settings: LockSettings::default(),
            lex_enabled: cfg!(feature = "lex"), // Enable by default if feature is enabled
            lex_index: None,
            #[cfg(feature = "lex")]
            lex_storage,
            vec_enabled: cfg!(feature = "vec"), // Enable by default if feature is enabled
            vec_compression: VectorCompression::None,
            vec_model: None,
            vec_index: None,
            clip_enabled: cfg!(feature = "clip"), // Enable by default if feature is enabled
            clip_index: None,
            dirty: false,
            #[cfg(feature = "lex")]
            tantivy: None,
            #[cfg(feature = "lex")]
            tantivy_dirty: false,
            #[cfg(feature = "temporal_track")]
            temporal_track: None,
            #[cfg(feature = "parallel_segments")]
            manifest_wal: Some(manifest_wal),
            memories_track: MemoriesTrack::new(),
            logic_mesh: LogicMesh::new(),
            sketch_track: SketchTrack::default(),
            schema_registry: SchemaRegistry::new(),
            schema_strict: false,
            #[cfg(feature = "replay")]
            active_session: None,
            #[cfg(feature = "replay")]
            completed_sessions: Vec::new(),
        };

        #[cfg(feature = "lex")]
        memvid.init_tantivy()?;

        #[cfg(feature = "parallel_segments")]
        memvid.load_manifest_segments(manifest_wal_entries);

        memvid.bootstrap_segment_catalog();

        // Create empty manifests for enabled indexes so they persist across open/close
        let empty_offset = memvid.data_end;
        let empty_checksum = *b"\xe3\xb0\xc4\x42\x98\xfc\x1c\x14\x9a\xfb\xf4\xc8\x99\x6f\xb9\x24\
                                \x27\xae\x41\xe4\x64\x9b\x93\x4c\xa4\x95\x99\x1b\x78\x52\xb8\x55";

        #[cfg(feature = "lex")]
        if memvid.lex_enabled && memvid.toc.indexes.lex.is_none() {
            memvid.toc.indexes.lex = Some(crate::types::LexIndexManifest {
                doc_count: 0,
                generation: 0,
                bytes_offset: empty_offset,
                bytes_length: 0,
                checksum: empty_checksum,
            });
        }

        #[cfg(feature = "vec")]
        if memvid.vec_enabled && memvid.toc.indexes.vec.is_none() {
            memvid.toc.indexes.vec = Some(crate::types::VecIndexManifest {
                vector_count: 0,
                dimension: 0,
                bytes_offset: empty_offset,
                bytes_length: 0,
                checksum: empty_checksum,
                compression_mode: memvid.vec_compression.clone(),
                model: memvid.vec_model.clone(),
            });
        }

        memvid.rewrite_toc_footer()?;
        memvid.header.toc_checksum = memvid.toc.toc_checksum;
        crate::persist_header(&mut memvid.file, &memvid.header)?;
        memvid.file.sync_all()?;
        Ok(memvid)
    }

    #[must_use]
    pub fn lock_settings(&self) -> &LockSettings {
        &self.lock_settings
    }

    pub fn lock_settings_mut(&mut self) -> &mut LockSettings {
        &mut self.lock_settings
    }

    /// Set the vector compression mode for this memory
    /// Must be called before ingesting documents with embeddings
    pub fn set_vector_compression(&mut self, compression: VectorCompression) {
        self.vec_compression = compression;
    }

    /// Get the current vector compression mode
    #[must_use]
    pub fn vector_compression(&self) -> &VectorCompression {
        &self.vec_compression
    }

    /// Predict the next frame ID that would be assigned to a new insert.
    ///
    /// Frame IDs are dense indices into `toc.frames`. When a memory is mutable, inserts are first
    /// appended to the embedded WAL and only materialized into `toc.frames` on commit. This helper
    /// lets frontends allocate stable frame IDs before an explicit commit.
    #[must_use]
    pub fn next_frame_id(&self) -> u64 {
        (self.toc.frames.len() as u64).saturating_add(self.pending_frame_inserts)
    }

    /// Returns the total number of frames in the memory.
    ///
    /// This includes all frames regardless of status (active, deleted, etc.).
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.toc.frames.len()
    }

    fn open_locked(mut file: File, lock: FileLock, path_ref: &Path) -> Result<Self> {
        // Fast-path detection for encrypted capsules (.mv2e).
        // This avoids confusing "invalid header" errors and provides an actionable hint.
        let mut magic = [0u8; 4];
        let is_mv2e = file.read_exact(&mut magic).is_ok() && magic == *b"MV2E";
        file.seek(SeekFrom::Start(0))?;
        if is_mv2e {
            return Err(MemvidError::EncryptedFile {
                path: path_ref.to_path_buf(),
                hint: format!("Run: memvid unlock {}", path_ref.display()),
            });
        }

        let mut header = HeaderCodec::read(&mut file)?;
        let toc = match read_toc(&mut file, &header) {
            Ok(toc) => toc,
            Err(err @ (MemvidError::Decode(_) | MemvidError::InvalidToc { .. })) => {
                tracing::info!("toc decode failed ({}); attempting recovery", err);
                let (toc, recovered_offset) = recover_toc(&mut file, Some(header.footer_offset))?;
                if recovered_offset != header.footer_offset
                    || header.toc_checksum != toc.toc_checksum
                {
                    header.footer_offset = recovered_offset;
                    header.toc_checksum = toc.toc_checksum;
                    crate::persist_header(&mut file, &header)?;
                }
                toc
            }
            Err(err) => return Err(err),
        };
        let checksum_result = toc.verify_checksum();

        // Validate segment integrity early to catch corruption before loading indexes
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        if let Err(e) = validate_segment_integrity(&toc, &header, file_len) {
            tracing::warn!("Segment integrity validation failed: {}", e);
            // Don't fail file open - let doctor handle it
            // This is just an early warning system
        }
        ensure_non_overlapping_frames(&toc, file_len)?;

        let wal = EmbeddedWal::open(&file, &header)?;
        #[cfg(feature = "lex")]
        let lex_storage = Arc::new(RwLock::new(EmbeddedLexStorage::from_manifest(
            toc.indexes.lex.as_ref(),
            &toc.indexes.lex_segments,
        )));
        #[cfg(feature = "parallel_segments")]
        let manifest_wal = ManifestWal::open(manifest_wal_path(path_ref))?;
        #[cfg(feature = "parallel_segments")]
        let manifest_wal_entries = manifest_wal.replay()?;

        let generation = detect_generation(&file)?.unwrap_or(0);
        let read_only = lock.mode() == LockMode::Shared;

        let mut memvid = Self {
            file,
            path: path_ref.to_path_buf(),
            lock,
            read_only,
            header,
            toc,
            wal,
            pending_frame_inserts: 0,
            data_end: 0,
            generation,
            lock_settings: LockSettings::default(),
            lex_enabled: false,
            lex_index: None,
            #[cfg(feature = "lex")]
            lex_storage,
            vec_enabled: false,
            vec_compression: VectorCompression::None,
            vec_model: None,
            vec_index: None,
            clip_enabled: false,
            clip_index: None,
            dirty: false,
            #[cfg(feature = "lex")]
            tantivy: None,
            #[cfg(feature = "lex")]
            tantivy_dirty: false,
            #[cfg(feature = "temporal_track")]
            temporal_track: None,
            #[cfg(feature = "parallel_segments")]
            manifest_wal: Some(manifest_wal),
            memories_track: MemoriesTrack::new(),
            logic_mesh: LogicMesh::new(),
            sketch_track: SketchTrack::default(),
            schema_registry: SchemaRegistry::new(),
            schema_strict: false,
            #[cfg(feature = "replay")]
            active_session: None,
            #[cfg(feature = "replay")]
            completed_sessions: Vec::new(),
        };
        memvid.data_end = compute_data_end(&memvid.toc, &memvid.header);
        // Use consolidated helper for lex_enabled check
        memvid.lex_enabled = has_lex_index(&memvid.toc);
        if memvid.lex_enabled {
            memvid.load_lex_index_from_manifest()?;
        }
        #[cfg(feature = "lex")]
        {
            memvid.init_tantivy()?;
        }
        memvid.vec_enabled =
            memvid.toc.indexes.vec.is_some() || !memvid.toc.segment_catalog.vec_segments.is_empty();
        if memvid.vec_enabled {
            memvid.load_vec_index_from_manifest()?;
        }
        memvid.clip_enabled = memvid.toc.indexes.clip.is_some();
        if memvid.clip_enabled {
            memvid.load_clip_index_from_manifest()?;
        }
        memvid.recover_wal()?;
        #[cfg(feature = "parallel_segments")]
        memvid.load_manifest_segments(manifest_wal_entries);
        memvid.bootstrap_segment_catalog();
        #[cfg(feature = "temporal_track")]
        memvid.ensure_temporal_track_loaded()?;
        memvid.load_memories_track()?;
        memvid.load_logic_mesh()?;
        memvid.load_sketch_track()?;
        if checksum_result.is_err() {
            memvid.toc.verify_checksum()?;
            if memvid.toc.toc_checksum != memvid.header.toc_checksum {
                memvid.header.toc_checksum = memvid.toc.toc_checksum;
                crate::persist_header(&mut memvid.file, &memvid.header)?;
                memvid.file.sync_all()?;
            }
        }
        Ok(memvid)
    }

    /// Open an existing `.mv2` with exclusive access, performing recovery if needed.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        ensure_single_file(path_ref)?;

        let (file, lock) = FileLock::open_and_lock(path_ref)?;
        Self::open_locked(file, lock, path_ref)
    }

    pub fn open_read_only<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_read_only_with_options(path, OpenReadOptions::default())
    }

    pub fn open_read_only_with_options<P: AsRef<Path>>(
        path: P,
        options: OpenReadOptions,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        ensure_single_file(path_ref)?;

        if options.allow_repair {
            return Self::open(path_ref);
        }

        Self::open_read_only_snapshot(path_ref)
    }

    fn open_read_only_snapshot(path_ref: &Path) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path_ref)?;
        let TailSnapshot {
            toc,
            footer_offset,
            data_end,
            generation,
        } = load_tail_snapshot(&file)?;

        let mut header = HeaderCodec::read(&mut file)?;
        header.footer_offset = footer_offset;
        header.toc_checksum = toc.toc_checksum;

        let lock = FileLock::acquire_with_mode(&file, LockMode::Shared)?;
        let wal = EmbeddedWal::open_read_only(&file, &header)?;

        #[cfg(feature = "lex")]
        let lex_storage = Arc::new(RwLock::new(EmbeddedLexStorage::from_manifest(
            toc.indexes.lex.as_ref(),
            &toc.indexes.lex_segments,
        )));

        let mut memvid = Self {
            file,
            path: path_ref.to_path_buf(),
            lock,
            read_only: true,
            header,
            toc,
            wal,
            pending_frame_inserts: 0,
            data_end,
            generation,
            lock_settings: LockSettings::default(),
            lex_enabled: false,
            lex_index: None,
            #[cfg(feature = "lex")]
            lex_storage,
            vec_enabled: false,
            vec_compression: VectorCompression::None,
            vec_model: None,
            vec_index: None,
            clip_enabled: false,
            clip_index: None,
            dirty: false,
            #[cfg(feature = "lex")]
            tantivy: None,
            #[cfg(feature = "lex")]
            tantivy_dirty: false,
            #[cfg(feature = "temporal_track")]
            temporal_track: None,
            #[cfg(feature = "parallel_segments")]
            manifest_wal: None,
            memories_track: MemoriesTrack::new(),
            logic_mesh: LogicMesh::new(),
            sketch_track: SketchTrack::default(),
            schema_registry: SchemaRegistry::new(),
            schema_strict: false,
            #[cfg(feature = "replay")]
            active_session: None,
            #[cfg(feature = "replay")]
            completed_sessions: Vec::new(),
        };

        // Use consolidated helper for lex_enabled check
        memvid.lex_enabled = has_lex_index(&memvid.toc);
        if memvid.lex_enabled {
            memvid.load_lex_index_from_manifest()?;
        }
        #[cfg(feature = "lex")]
        memvid.init_tantivy()?;

        memvid.vec_enabled =
            memvid.toc.indexes.vec.is_some() || !memvid.toc.segment_catalog.vec_segments.is_empty();
        if memvid.vec_enabled {
            memvid.load_vec_index_from_manifest()?;
        }
        memvid.clip_enabled = memvid.toc.indexes.clip.is_some();
        if memvid.clip_enabled {
            memvid.load_clip_index_from_manifest()?;
        }
        // Load memories track, Logic-Mesh, and sketch track if present
        memvid.load_memories_track()?;
        memvid.load_logic_mesh()?;
        memvid.load_sketch_track()?;

        memvid.bootstrap_segment_catalog();
        #[cfg(feature = "temporal_track")]
        memvid.ensure_temporal_track_loaded()?;

        Ok(memvid)
    }

    pub(crate) fn try_open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        ensure_single_file(path_ref)?;

        let file = OpenOptions::new().read(true).write(true).open(path_ref)?;
        let lock = match FileLock::try_acquire(&file, path_ref)? {
            Some(lock) => lock,
            None => {
                return Err(MemvidError::Lock(
                    "exclusive access unavailable for doctor".to_string(),
                ));
            }
        };
        Self::open_locked(file, lock, path_ref)
    }

    fn bootstrap_segment_catalog(&mut self) {
        let catalog = &mut self.toc.segment_catalog;
        if catalog.version == 0 {
            catalog.version = 1;
        }
        if catalog.next_segment_id == 0 {
            let mut max_id = 0u64;
            for descriptor in &catalog.lex_segments {
                max_id = max_id.max(descriptor.common.segment_id);
            }
            for descriptor in &catalog.vec_segments {
                max_id = max_id.max(descriptor.common.segment_id);
            }
            for descriptor in &catalog.time_segments {
                max_id = max_id.max(descriptor.common.segment_id);
            }
            #[cfg(feature = "temporal_track")]
            for descriptor in &catalog.temporal_segments {
                max_id = max_id.max(descriptor.common.segment_id);
            }
            #[cfg(feature = "parallel_segments")]
            for descriptor in &catalog.index_segments {
                max_id = max_id.max(descriptor.common.segment_id);
            }
            if max_id > 0 {
                catalog.next_segment_id = max_id.saturating_add(1);
            }
        }
    }

    #[cfg(feature = "parallel_segments")]
    fn load_manifest_segments(&mut self, entries: Vec<IndexSegmentRef>) {
        if entries.is_empty() {
            return;
        }
        for entry in entries {
            let duplicate = self
                .toc
                .segment_catalog
                .index_segments
                .iter()
                .any(|existing| existing.common.segment_id == entry.common.segment_id);
            if !duplicate {
                self.toc.segment_catalog.index_segments.push(entry);
            }
        }
    }

    /// Load the memories track from the manifest if present.
    fn load_memories_track(&mut self) -> Result<()> {
        let manifest = match &self.toc.memories_track {
            Some(m) => m,
            None => return Ok(()),
        };

        // Read the compressed data from the file
        if manifest.bytes_length > crate::MAX_INDEX_BYTES {
            return Err(MemvidError::InvalidToc {
                reason: "memories track exceeds safety limit".into(),
            });
        }
        // Safe: guarded by MAX_INDEX_BYTES check above
        #[allow(clippy::cast_possible_truncation)]
        let mut buf = vec![0u8; manifest.bytes_length as usize];
        self.file
            .seek(std::io::SeekFrom::Start(manifest.bytes_offset))?;
        self.file.read_exact(&mut buf)?;

        // Verify checksum
        let actual_checksum: [u8; 32] = blake3::hash(&buf).into();
        if actual_checksum != manifest.checksum {
            return Err(MemvidError::InvalidToc {
                reason: "memories track checksum mismatch".into(),
            });
        }

        // Deserialize the memories track
        self.memories_track = MemoriesTrack::deserialize(&buf)?;

        Ok(())
    }

    /// Load the Logic-Mesh from the manifest if present.
    fn load_logic_mesh(&mut self) -> Result<()> {
        let manifest = match &self.toc.logic_mesh {
            Some(m) => m,
            None => return Ok(()),
        };

        // Read the serialized data from the file
        if manifest.bytes_length > crate::MAX_INDEX_BYTES {
            return Err(MemvidError::InvalidToc {
                reason: "logic mesh exceeds safety limit".into(),
            });
        }
        // Safe: guarded by MAX_INDEX_BYTES check above
        #[allow(clippy::cast_possible_truncation)]
        let mut buf = vec![0u8; manifest.bytes_length as usize];
        self.file
            .seek(std::io::SeekFrom::Start(manifest.bytes_offset))?;
        self.file.read_exact(&mut buf)?;

        // Verify checksum
        let actual_checksum: [u8; 32] = blake3::hash(&buf).into();
        if actual_checksum != manifest.checksum {
            return Err(MemvidError::InvalidToc {
                reason: "logic mesh checksum mismatch".into(),
            });
        }

        // Deserialize the logic mesh
        self.logic_mesh = LogicMesh::deserialize(&buf)?;

        Ok(())
    }

    /// Load the sketch track from the manifest if present.
    fn load_sketch_track(&mut self) -> Result<()> {
        let manifest = match &self.toc.sketch_track {
            Some(m) => m.clone(),
            None => return Ok(()),
        };

        // Read and deserialize the sketch track (read_sketch_track handles seeking and checksum)
        self.sketch_track = crate::types::read_sketch_track(
            &mut self.file,
            manifest.bytes_offset,
            manifest.bytes_length,
        )?;

        Ok(())
    }

    #[cfg(feature = "temporal_track")]
    pub(crate) fn ensure_temporal_track_loaded(&mut self) -> Result<()> {
        if self.temporal_track.is_some() {
            return Ok(());
        }
        let manifest = match &self.toc.temporal_track {
            Some(manifest) => manifest.clone(),
            None => return Ok(()),
        };
        if manifest.bytes_length == 0 {
            return Ok(());
        }
        let file_len = self.file.metadata()?.len();
        let Some(end) = manifest.bytes_offset.checked_add(manifest.bytes_length) else {
            return Ok(());
        };
        if end > file_len {
            return Ok(());
        }
        match temporal_track_read(&mut self.file, manifest.bytes_offset, manifest.bytes_length) {
            Ok(track) => self.temporal_track = Some(track),
            Err(MemvidError::InvalidTemporalTrack { .. }) => {
                return Ok(());
            }
            Err(err) => return Err(err),
        }
        Ok(())
    }

    #[cfg(feature = "temporal_track")]
    pub(crate) fn temporal_track_ref(&mut self) -> Result<Option<&TemporalTrack>> {
        self.ensure_temporal_track_loaded()?;
        Ok(self.temporal_track.as_ref())
    }

    #[cfg(feature = "temporal_track")]
    pub(crate) fn temporal_anchor_timestamp(&mut self, frame_id: FrameId) -> Result<Option<i64>> {
        self.ensure_temporal_track_loaded()?;
        let Some(track) = self.temporal_track.as_ref() else {
            return Ok(None);
        };
        if !track.capabilities().has_anchors {
            return Ok(None);
        }
        Ok(track
            .anchor_for_frame(frame_id)
            .map(|anchor| anchor.anchor_ts))
    }

    #[cfg(feature = "temporal_track")]
    pub(crate) fn clear_temporal_track_cache(&mut self) {
        self.temporal_track = None;
    }

    #[cfg(feature = "temporal_track")]
    pub(crate) fn effective_temporal_timestamp(
        &mut self,
        frame_id: FrameId,
        fallback: i64,
    ) -> Result<i64> {
        Ok(self
            .temporal_anchor_timestamp(frame_id)?
            .unwrap_or(fallback))
    }

    #[cfg(not(feature = "temporal_track"))]
    pub(crate) fn effective_temporal_timestamp(
        &mut self,
        _frame_id: crate::types::FrameId,
        fallback: i64,
    ) -> Result<i64> {
        Ok(fallback)
    }

    /// Get current memory binding information.
    ///
    /// Returns the binding if this file is bound to a dashboard memory,
    /// or None if unbound.
    #[must_use]
    pub fn get_memory_binding(&self) -> Option<&crate::types::MemoryBinding> {
        self.toc.memory_binding.as_ref()
    }

    /// Bind this file to a dashboard memory.
    ///
    /// This stores the binding in the TOC and applies a temporary ticket for initial binding.
    /// The caller should follow up with `apply_signed_ticket` for cryptographic verification.
    ///
    /// # Errors
    ///
    /// Returns `MemoryAlreadyBound` if this file is already bound to a different memory.
    #[allow(deprecated)]
    pub fn bind_memory(
        &mut self,
        binding: crate::types::MemoryBinding,
        ticket: crate::types::Ticket,
    ) -> Result<()> {
        // Check existing binding
        if let Some(existing) = self.get_memory_binding() {
            if existing.memory_id != binding.memory_id {
                return Err(MemvidError::MemoryAlreadyBound {
                    existing_memory_id: existing.memory_id,
                    existing_memory_name: existing.memory_name.clone(),
                    bound_at: existing.bound_at.to_rfc3339(),
                });
            }
        }

        // Apply ticket for capacity
        self.apply_ticket(ticket)?;

        // Store binding in TOC
        self.toc.memory_binding = Some(binding);
        self.dirty = true;

        Ok(())
    }

    /// Set only the memory binding without applying a ticket.
    ///
    /// This is used when the caller will immediately follow up with `apply_signed_ticket`
    /// to apply the cryptographically verified ticket. This avoids the sequence number
    /// conflict that occurs when using `bind_memory` with a temporary ticket.
    ///
    /// # Errors
    ///
    /// Returns `MemoryAlreadyBound` if this file is already bound to a different memory.
    pub fn set_memory_binding_only(&mut self, binding: crate::types::MemoryBinding) -> Result<()> {
        self.ensure_writable()?;

        // Check existing binding
        if let Some(existing) = self.get_memory_binding() {
            if existing.memory_id != binding.memory_id {
                return Err(MemvidError::MemoryAlreadyBound {
                    existing_memory_id: existing.memory_id,
                    existing_memory_name: existing.memory_name.clone(),
                    bound_at: existing.bound_at.to_rfc3339(),
                });
            }
        }

        // Store binding in TOC (without applying a ticket)
        self.toc.memory_binding = Some(binding);
        self.dirty = true;

        Ok(())
    }

    /// Unbind this file from its dashboard memory.
    ///
    /// This clears the binding and reverts to free tier capacity (1 GB).
    pub fn unbind_memory(&mut self) -> Result<()> {
        self.toc.memory_binding = None;
        // Revert to free tier
        self.toc.ticket_ref = crate::types::TicketRef {
            issuer: "free-tier".into(),
            seq_no: 1,
            expires_in_secs: 0,
            capacity_bytes: crate::types::Tier::Free.capacity_bytes(),
            verified: false,
        };
        self.dirty = true;
        Ok(())
    }
}

pub(crate) fn read_toc(file: &mut File, header: &Header) -> Result<Toc> {
    use crate::footer::{CommitFooter, FOOTER_SIZE};

    let len = file.metadata()?.len();
    if len < header.footer_offset {
        return Err(MemvidError::InvalidToc {
            reason: "footer offset beyond file length".into(),
        });
    }

    // Read the entire region from footer_offset to EOF (includes TOC + footer)
    file.seek(SeekFrom::Start(header.footer_offset))?;
    // Safe: total_size bounded by file length, and we check MAX_INDEX_BYTES before reading
    #[allow(clippy::cast_possible_truncation)]
    let total_size = (len - header.footer_offset) as usize;
    if total_size as u64 > crate::MAX_INDEX_BYTES {
        return Err(MemvidError::InvalidToc {
            reason: "toc region exceeds safety limit".into(),
        });
    }

    if total_size < FOOTER_SIZE {
        return Err(MemvidError::InvalidToc {
            reason: "region too small to contain footer".into(),
        });
    }

    let mut buf = Vec::with_capacity(total_size);
    file.read_to_end(&mut buf)?;

    // Parse the footer (last FOOTER_SIZE bytes)
    let footer_start = buf.len() - FOOTER_SIZE;
    let footer_bytes = &buf[footer_start..];
    let footer = CommitFooter::decode(footer_bytes).ok_or(MemvidError::InvalidToc {
        reason: "failed to decode commit footer".into(),
    })?;

    // Extract only the TOC bytes (excluding the footer)
    let toc_bytes = &buf[..footer_start];
    #[allow(clippy::cast_possible_truncation)]
    if toc_bytes.len() != footer.toc_len as usize {
        return Err(MemvidError::InvalidToc {
            reason: "toc length mismatch".into(),
        });
    }
    if !footer.hash_matches(toc_bytes) {
        return Err(MemvidError::InvalidToc {
            reason: "commit footer toc hash mismatch".into(),
        });
    }

    verify_toc_prefix(toc_bytes)?;
    let toc = Toc::decode(toc_bytes)?;
    Ok(toc)
}

fn verify_toc_prefix(bytes: &[u8]) -> Result<()> {
    const MAX_SEGMENTS: u64 = 1_000_000;
    const MAX_FRAMES: u64 = 1_000_000;
    const MIN_SEGMENT_META_BYTES: u64 = 32;
    const MIN_FRAME_BYTES: u64 = 64;
    // TOC trailer layout (little-endian):
    // [toc_version:u64][segments_len:u64][frames_len:u64]...
    let read_u64 = |range: std::ops::Range<usize>, context: &str| -> Result<u64> {
        let slice = bytes.get(range).ok_or_else(|| MemvidError::InvalidToc {
            reason: context.to_string().into(),
        })?;
        let array: [u8; 8] = slice.try_into().map_err(|_| MemvidError::InvalidToc {
            reason: context.to_string().into(),
        })?;
        Ok(u64::from_le_bytes(array))
    };

    if bytes.len() < 24 {
        return Err(MemvidError::InvalidToc {
            reason: "toc trailer too small".into(),
        });
    }
    let toc_version = read_u64(0..8, "toc version missing or truncated")?;
    if toc_version > 32 {
        return Err(MemvidError::InvalidToc {
            reason: "toc version unreasonable".into(),
        });
    }
    let segments_len = read_u64(8..16, "segment count missing or truncated")?;
    if segments_len > MAX_SEGMENTS {
        return Err(MemvidError::InvalidToc {
            reason: "segment count unreasonable".into(),
        });
    }
    let frames_len = read_u64(16..24, "frame count missing or truncated")?;
    if frames_len > MAX_FRAMES {
        return Err(MemvidError::InvalidToc {
            reason: "frame count unreasonable".into(),
        });
    }
    let required = segments_len
        .saturating_mul(MIN_SEGMENT_META_BYTES)
        .saturating_add(frames_len.saturating_mul(MIN_FRAME_BYTES));
    if required > bytes.len() as u64 {
        return Err(MemvidError::InvalidToc {
            reason: "toc payload inconsistent with counts".into(),
        });
    }
    Ok(())
}

/// Ensure frame payloads do not overlap each other or exceed file boundary.
///
/// Frames in the TOC are ordered by `frame_id`, not by `payload_offset`, so we must
/// sort by `payload_offset` before checking for overlaps.
///
/// Note: Frames with `payload_length` == 0 are "virtual" frames (e.g., document
/// frames that reference chunks) and are skipped from this check.
fn ensure_non_overlapping_frames(toc: &Toc, file_len: u64) -> Result<()> {
    // Collect active frames with actual payloads and sort by payload_offset
    let mut frames_by_offset: Vec<_> = toc
        .frames
        .iter()
        .filter(|f| f.status == FrameStatus::Active && f.payload_length > 0)
        .collect();
    frames_by_offset.sort_by_key(|f| f.payload_offset);

    let mut previous_end = 0u64;
    for frame in frames_by_offset {
        let end = frame
            .payload_offset
            .checked_add(frame.payload_length)
            .ok_or_else(|| MemvidError::InvalidToc {
                reason: "frame payload offsets overflow".into(),
            })?;
        if end > file_len {
            return Err(MemvidError::InvalidToc {
                reason: "frame payload exceeds file length".into(),
            });
        }
        if frame.payload_offset < previous_end {
            return Err(MemvidError::InvalidToc {
                reason: format!(
                    "frame {} payload overlaps with previous frame (offset {} < previous end {})",
                    frame.id, frame.payload_offset, previous_end
                )
                .into(),
            });
        }
        previous_end = end;
    }
    Ok(())
}

pub(crate) fn recover_toc(file: &mut File, hint: Option<u64>) -> Result<(Toc, u64)> {
    let len = file.metadata()?.len();
    // Safety: we only create a read-only mapping over stable file bytes.
    let mmap = unsafe { Mmap::map(&*file)? };
    tracing::debug!(file_len = len, "attempting toc recovery");

    // First, try to find a valid footer which includes validated TOC bytes
    if let Some(footer_slice) = find_last_valid_footer(&mmap) {
        tracing::debug!(
            footer_offset = footer_slice.footer_offset,
            toc_offset = footer_slice.toc_offset,
            toc_len = footer_slice.toc_bytes.len(),
            "found valid footer during recovery"
        );
        // The footer has already validated the TOC hash, so we can directly decode it
        match Toc::decode(footer_slice.toc_bytes) {
            Ok(toc) => {
                return Ok((toc, footer_slice.toc_offset as u64));
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "footer-validated TOC failed to decode, falling back to scan"
                );
            }
        }
    }

    // If we have a header-provided hint (`footer_offset`) but the commit footer itself is corrupted,
    // we can often still recover because the TOC bytes are intact. In that case, assume the TOC
    // spans from `hint` up to the final fixed-size commit footer and decode it best-effort.
    if let Some(hint_offset) = hint {
        use crate::footer::FOOTER_SIZE;

        // Safe: file successfully mmapped so length fits in usize
        #[allow(clippy::cast_possible_truncation)]
        let start = (hint_offset.min(len)) as usize;
        if mmap.len().saturating_sub(start) >= FOOTER_SIZE {
            let toc_end = mmap.len().saturating_sub(FOOTER_SIZE);
            if toc_end > start {
                let toc_bytes = &mmap[start..toc_end];
                if verify_toc_prefix(toc_bytes).is_ok() {
                    let attempt = panic::catch_unwind(|| Toc::decode(toc_bytes));
                    if let Ok(Ok(toc)) = attempt {
                        tracing::debug!(
                            recovered_offset = hint_offset,
                            recovered_frames = toc.frames.len(),
                            "recovered toc from hinted offset without validated footer"
                        );
                        return Ok((toc, hint_offset));
                    }
                }
            }
        }
    }

    // Fallback to manual scan if footer-based recovery failed
    let mut ranges = Vec::new();
    if let Some(hint_offset) = hint {
        // Safe: file successfully mmapped so length fits in usize
        #[allow(clippy::cast_possible_truncation)]
        let hint_idx = hint_offset.min(len) as usize;
        ranges.push((hint_idx, mmap.len()));
        if hint_idx > 0 {
            ranges.push((0, hint_idx));
        }
    } else {
        ranges.push((0, mmap.len()));
    }

    for (start, end) in ranges {
        if let Some(found) = scan_range_for_toc(&mmap, start, end) {
            return Ok(found);
        }
    }

    Err(MemvidError::InvalidToc {
        reason: "unable to recover table of contents from file trailer".into(),
    })
}

fn scan_range_for_toc(data: &[u8], start: usize, end: usize) -> Option<(Toc, u64)> {
    if start >= end || end > data.len() {
        return None;
    }
    const MAX_TOC_BYTES: usize = 64 * 1024 * 1024;
    const ZERO_CHECKSUM: [u8; 32] = [0u8; 32];

    // We only ever consider offsets where the candidate TOC slice would be <= MAX_TOC_BYTES,
    // otherwise the loop devolves into iterating over the entire file for large memories.
    let min_offset = data.len().saturating_sub(MAX_TOC_BYTES);
    let scan_start = start.max(min_offset);

    for offset in (scan_start..end).rev() {
        let slice = &data[offset..];
        if slice.len() < 16 {
            continue;
        }
        debug_assert!(slice.len() <= MAX_TOC_BYTES);

        // No footer found - try old format with checksum
        if slice.len() < ZERO_CHECKSUM.len() {
            continue;
        }
        let (body, stored_checksum) = slice.split_at(slice.len() - ZERO_CHECKSUM.len());
        let mut hasher = Hasher::new();
        hasher.update(body);
        hasher.update(&ZERO_CHECKSUM);
        if hasher.finalize().as_bytes() != stored_checksum {
            continue;
        }
        if verify_toc_prefix(slice).is_err() {
            continue;
        }
        let attempt = panic::catch_unwind(|| Toc::decode(slice));
        if let Ok(Ok(toc)) = attempt {
            let recovered_offset = offset as u64;
            tracing::debug!(
                recovered_offset,
                recovered_frames = toc.frames.len(),
                "recovered toc via scan"
            );
            return Some((toc, recovered_offset));
        }
    }
    None
}

pub(crate) fn prepare_toc_bytes(toc: &mut Toc) -> Result<Vec<u8>> {
    toc.toc_checksum = [0u8; 32];
    let bytes = toc.encode()?;
    let checksum = Toc::calculate_checksum(&bytes);
    toc.toc_checksum = checksum;
    toc.encode()
}

pub(crate) fn empty_toc() -> Toc {
    Toc {
        toc_version: 0,
        segments: Vec::new(),
        frames: Vec::new(),
        indexes: IndexManifests::default(),
        time_index: None,
        temporal_track: None,
        memories_track: None,
        logic_mesh: None,
        sketch_track: None,
        segment_catalog: SegmentCatalog::default(),
        ticket_ref: TicketRef {
            issuer: "free-tier".into(),
            seq_no: 1,
            expires_in_secs: 0,
            capacity_bytes: Tier::Free.capacity_bytes(),
            verified: false,
        },
        memory_binding: None,
        replay_manifest: None,
        enrichment_queue: crate::types::EnrichmentQueueManifest::default(),
        merkle_root: [0u8; 32],
        toc_checksum: [0u8; 32],
    }
}

pub(crate) fn compute_data_end(toc: &Toc, header: &Header) -> u64 {
    // `data_end` tracks the end of all data bytes that should not be overwritten by appends:
    // - frame payloads
    // - embedded indexes / metadata segments referenced by the TOC
    // - the current footer boundary (TOC offset), since callers may safely overwrite old TOCs
    //
    // Keeping this conservative prevents WAL replay / appends from corrupting embedded segments.
    let wal_region_end = header.wal_offset.saturating_add(header.wal_size);
    let mut max_end = wal_region_end.max(header.footer_offset);

    // Frame payloads (active only).
    for frame in toc
        .frames
        .iter()
        .filter(|f| f.status == FrameStatus::Active && f.payload_length > 0)
    {
        if let Some(end) = frame.payload_offset.checked_add(frame.payload_length) {
            max_end = max_end.max(end);
        }
    }

    // Segment catalog entries.
    let catalog = &toc.segment_catalog;
    for seg in &catalog.lex_segments {
        if let Some(end) = seg.common.bytes_offset.checked_add(seg.common.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    for seg in &catalog.vec_segments {
        if let Some(end) = seg.common.bytes_offset.checked_add(seg.common.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    for seg in &catalog.time_segments {
        if let Some(end) = seg.common.bytes_offset.checked_add(seg.common.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    #[cfg(feature = "temporal_track")]
    for seg in &catalog.temporal_segments {
        if let Some(end) = seg.common.bytes_offset.checked_add(seg.common.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    #[cfg(feature = "lex")]
    for seg in &catalog.tantivy_segments {
        if let Some(end) = seg.common.bytes_offset.checked_add(seg.common.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    #[cfg(feature = "parallel_segments")]
    for seg in &catalog.index_segments {
        if let Some(end) = seg.common.bytes_offset.checked_add(seg.common.bytes_length) {
            max_end = max_end.max(end);
        }
    }

    // Global manifests (non-segment storage paths).
    if let Some(manifest) = toc.indexes.lex.as_ref() {
        if let Some(end) = manifest.bytes_offset.checked_add(manifest.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    if let Some(manifest) = toc.indexes.vec.as_ref() {
        if let Some(end) = manifest.bytes_offset.checked_add(manifest.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    if let Some(manifest) = toc.indexes.clip.as_ref() {
        if let Some(end) = manifest.bytes_offset.checked_add(manifest.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    if let Some(manifest) = toc.time_index.as_ref() {
        if let Some(end) = manifest.bytes_offset.checked_add(manifest.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    #[cfg(feature = "temporal_track")]
    if let Some(track) = toc.temporal_track.as_ref() {
        if let Some(end) = track.bytes_offset.checked_add(track.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    if let Some(track) = toc.memories_track.as_ref() {
        if let Some(end) = track.bytes_offset.checked_add(track.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    if let Some(mesh) = toc.logic_mesh.as_ref() {
        if let Some(end) = mesh.bytes_offset.checked_add(mesh.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    if let Some(track) = toc.sketch_track.as_ref() {
        if let Some(end) = track.bytes_offset.checked_add(track.bytes_length) {
            max_end = max_end.max(end);
        }
    }
    #[cfg(feature = "replay")]
    if let Some(manifest) = toc.replay_manifest.as_ref() {
        if let Some(end) = manifest.segment_offset.checked_add(manifest.segment_size) {
            max_end = max_end.max(end);
        }
    }

    tracing::debug!(
        wal_region_end,
        footer_offset = header.footer_offset,
        computed_data_end = max_end,
        "compute_data_end"
    );

    max_end
}

struct TailSnapshot {
    toc: Toc,
    footer_offset: u64,
    data_end: u64,
    generation: u64,
}

fn locate_footer_window(mmap: &[u8]) -> Option<(FooterSlice<'_>, usize)> {
    const MAX_SEARCH_SIZE: usize = 16 * 1024 * 1024;
    if mmap.is_empty() {
        return None;
    }
    let mut window = MAX_SEARCH_SIZE.min(mmap.len());
    loop {
        let start = mmap.len() - window;
        if let Some(slice) = find_last_valid_footer(&mmap[start..]) {
            return Some((slice, start));
        }
        if window == mmap.len() {
            break;
        }
        window = (window * 2).min(mmap.len());
    }
    None
}

fn load_tail_snapshot(file: &File) -> Result<TailSnapshot> {
    // Safety: we only create a read-only mapping over the stable file bytes.
    let mmap = unsafe { Mmap::map(file)? };

    let (slice, offset_adjustment) =
        locate_footer_window(&mmap).ok_or_else(|| MemvidError::InvalidToc {
            reason: "no valid commit footer found".into(),
        })?;
    let toc = Toc::decode(slice.toc_bytes)?;
    toc.verify_checksum()?;

    Ok(TailSnapshot {
        toc,
        footer_offset: slice.footer_offset as u64 + offset_adjustment as u64,
        // Using toc_offset causes stale data_end that moves footer backwards on next commit
        data_end: slice.footer_offset as u64 + offset_adjustment as u64,
        generation: slice.footer.generation,
    })
}

fn detect_generation(file: &File) -> Result<Option<u64>> {
    // Safety: read-only mapping for footer inspection.
    let mmap = unsafe { Mmap::map(file)? };

    Ok(locate_footer_window(&mmap).map(|(slice, _)| slice.footer.generation))
}

pub(crate) fn ensure_single_file(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let forbidden = ["-wal", "-shm", "-lock", "-journal"];
        for suffix in forbidden {
            let candidate = parent.join(format!("{name}{suffix}"));
            if candidate.exists() {
                return Err(MemvidError::AuxiliaryFileDetected { path: candidate });
            }
        }
        let hidden_forbidden = [".wal", ".shm", ".lock", ".journal"];
        for suffix in hidden_forbidden {
            let candidate = parent.join(format!(".{name}{suffix}"));
            if candidate.exists() {
                return Err(MemvidError::AuxiliaryFileDetected { path: candidate });
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel_segments")]
fn manifest_wal_path(path: &Path) -> PathBuf {
    let mut wal_path = path.to_path_buf();
    wal_path.set_extension("manifest.wal");
    wal_path
}

#[cfg(feature = "parallel_segments")]
pub(crate) fn cleanup_manifest_wal_public(path: &Path) {
    let wal_path = manifest_wal_path(path);
    if wal_path.exists() {
        let _ = std::fs::remove_file(&wal_path);
    }
}

/// Single source of truth: does this TOC have a lexical index?
/// Checks all possible locations: old manifest, `lex_segments`, and `tantivy_segments`.
pub(crate) fn has_lex_index(toc: &Toc) -> bool {
    toc.segment_catalog.lex_enabled
        || toc.indexes.lex.is_some()
        || !toc.indexes.lex_segments.is_empty()
        || !toc.segment_catalog.tantivy_segments.is_empty()
}

/// Single source of truth: expected document count for lex index.
/// Returns None if we can't determine (e.g., Tantivy segments without manifest).
#[cfg(feature = "lex")]
pub(crate) fn lex_doc_count(
    toc: &Toc,
    lex_storage: &crate::search::EmbeddedLexStorage,
) -> Option<u64> {
    // First try old manifest
    if let Some(manifest) = &toc.indexes.lex {
        if manifest.doc_count > 0 {
            return Some(manifest.doc_count);
        }
    }

    // Then try lex_storage (contains info from lex_segments)
    let storage_count = lex_storage.doc_count();
    if storage_count > 0 {
        return Some(storage_count);
    }

    // For Tantivy files with segments but no manifest/storage doc_count,
    // we can't know doc count without loading the index.
    // Return None and let caller decide (init_tantivy should trust segments exist)
    None
}

/// Validates segment integrity on file open to catch corruption early.
/// This helps doctor by detecting issues before they cause problems.
#[allow(dead_code)]
fn validate_segment_integrity(toc: &Toc, header: &Header, file_len: u64) -> Result<()> {
    let data_limit = header.footer_offset;

    // Validate replay segment (if present). Replay is stored AT the footer boundary,
    // and footer_offset is moved forward after writing. So we only check against file_len,
    // not against footer_offset (which would be after the replay segment).
    #[cfg(feature = "replay")]
    if let Some(manifest) = toc.replay_manifest.as_ref() {
        if manifest.segment_size != 0 {
            let end = manifest
                .segment_offset
                .checked_add(manifest.segment_size)
                .ok_or_else(|| MemvidError::Doctor {
                    reason: format!(
                        "Replay segment offset overflow: {} + {}",
                        manifest.segment_offset, manifest.segment_size
                    ),
                })?;

            // Only check against file_len - replay segments sit at the footer boundary
            // and footer_offset is updated to point after them
            if end > file_len {
                return Err(MemvidError::Doctor {
                    reason: format!(
                        "Replay segment out of bounds: offset={}, length={}, end={}, file_len={}",
                        manifest.segment_offset, manifest.segment_size, end, file_len
                    ),
                });
            }
        }
    }

    // Validate Tantivy segments
    for (idx, seg) in toc.segment_catalog.tantivy_segments.iter().enumerate() {
        let offset = seg.common.bytes_offset;
        let length = seg.common.bytes_length;

        if length == 0 {
            continue; // Empty segments are okay
        }

        let end = offset
            .checked_add(length)
            .ok_or_else(|| MemvidError::Doctor {
                reason: format!("Tantivy segment {idx} offset overflow: {offset} + {length}"),
            })?;

        if end > file_len || end > data_limit {
            return Err(MemvidError::Doctor {
                reason: format!(
                    "Tantivy segment {idx} out of bounds: offset={offset}, length={length}, end={end}, file_len={file_len}, data_limit={data_limit}"
                ),
            });
        }
    }

    // Validate time index segments
    for (idx, seg) in toc.segment_catalog.time_segments.iter().enumerate() {
        let offset = seg.common.bytes_offset;
        let length = seg.common.bytes_length;

        if length == 0 {
            continue;
        }

        let end = offset
            .checked_add(length)
            .ok_or_else(|| MemvidError::Doctor {
                reason: format!("Time segment {idx} offset overflow: {offset} + {length}"),
            })?;

        if end > file_len || end > data_limit {
            return Err(MemvidError::Doctor {
                reason: format!(
                    "Time segment {idx} out of bounds: offset={offset}, length={length}, end={end}, file_len={file_len}, data_limit={data_limit}"
                ),
            });
        }
    }

    // Validate vec segments
    for (idx, seg) in toc.segment_catalog.vec_segments.iter().enumerate() {
        let offset = seg.common.bytes_offset;
        let length = seg.common.bytes_length;

        if length == 0 {
            continue;
        }

        let end = offset
            .checked_add(length)
            .ok_or_else(|| MemvidError::Doctor {
                reason: format!("Vec segment {idx} offset overflow: {offset} + {length}"),
            })?;

        if end > file_len || end > data_limit {
            return Err(MemvidError::Doctor {
                reason: format!(
                    "Vec segment {idx} out of bounds: offset={offset}, length={length}, end={end}, file_len={file_len}, data_limit={data_limit}"
                ),
            });
        }
    }

    log::debug!("✓ Segment integrity validation passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn toc_prefix_underflow_surfaces_reason() {
        let err = verify_toc_prefix(&[0u8; 8]).expect_err("should reject short toc prefix");
        match err {
            MemvidError::InvalidToc { reason } => {
                assert!(
                    reason.contains("trailer too small"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn ensure_single_file_blocks_sidecars() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("mem.mv2");
        std::fs::write(dir.path().join("mem.mv2-wal"), b"junk").expect("sidecar");
        let result = Memvid::create(&path);
        assert!(matches!(
            result,
            Err(MemvidError::AuxiliaryFileDetected { .. })
        ));
    }
}
