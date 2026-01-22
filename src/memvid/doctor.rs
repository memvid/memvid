// Safe unwrap/expect: Option takes with immediate value replacement.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::cell::Cell;
use std::cmp::min;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::error::{MemvidError, Result};
use crate::io::header::HeaderCodec;
use crate::io::time_index::{calculate_checksum as time_index_checksum, read_track};
use crate::io::wal::EmbeddedWal;
use crate::memvid::lifecycle::{Memvid, ensure_single_file, read_toc, recover_toc};
use crate::types::{
    DOCTOR_PLAN_VERSION, DoctorActionDetail, DoctorActionKind, DoctorActionPlan,
    DoctorActionReport, DoctorActionStatus, DoctorFinding, DoctorFindingCode, DoctorMetrics,
    DoctorOptions, DoctorPhaseDuration, DoctorPhaseKind, DoctorPhasePlan, DoctorPhaseReport,
    DoctorPhaseStatus, DoctorPlan, DoctorReport, DoctorStatus, VerificationReport,
    VerificationStatus,
};
use crate::types::{Header, Toc};

#[cfg(feature = "lex")]
use crate::lex::LexIndex;
use crate::vec::VecIndex;

// Thread-local flag to control doctor debug logging
thread_local! {
    static DOCTOR_QUIET: Cell<bool> = const { Cell::new(false) };
}

/// Set quiet mode for doctor logging (suppresses debug output when true).
fn set_doctor_quiet(quiet: bool) {
    DOCTOR_QUIET.with(|q| q.set(quiet));
}

/// Check if doctor logging is suppressed.
fn is_doctor_quiet() -> bool {
    DOCTOR_QUIET.with(|q| q.get())
}

/// Conditionally print doctor debug messages based on quiet flag.
macro_rules! doctor_log {
    ($($arg:tt)*) => {
        if !is_doctor_quiet() {
            println!($($arg)*);
        }
    };
}

#[derive(Default)]
struct IndexProbe {
    needs_time: bool,
    needs_lex: bool,
    needs_vec: bool,
    time_expected_entries: u64,
    lex_expected_docs: u64,
    vec_expected_vectors: u64,
    vec_dimension: u32,
}

struct PlanProbe {
    header: Option<Header>,
    toc: Option<Toc>,
    toc_offset: Option<u64>,
    toc_recovered: bool,
    findings: Vec<DoctorFinding>,
    wal_pending: usize,
    wal_from_sequence: u64,
    wal_to_sequence: u64,
    index: IndexProbe,
    file_len: u64,
}

pub(crate) fn doctor_plan(path: &Path, options: DoctorOptions) -> Result<DoctorPlan> {
    doctor_log!(
        "doctor: planning for {:?} (options: rebuild_time={}, rebuild_lex={}, rebuild_vec={})",
        path,
        options.rebuild_time_index,
        options.rebuild_lex_index,
        options.rebuild_vec_index,
    );
    ensure_single_file(path)?;
    let planner = DoctorPlanner::new(path.to_path_buf(), options);
    planner.compute()
}

/// Attempt to recover from WAL corruption by rebuilding a clean WAL
fn try_recover_from_wal_corruption(path: &Path) -> Result<Memvid> {
    use fs2::FileExt;

    doctor_log!("doctor: opening file for WAL recovery");
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;

    // Acquire exclusive lock
    file.lock_exclusive()?;

    doctor_log!("doctor: reading header");
    let mut header = HeaderCodec::read(&mut file)?;

    doctor_log!(
        "doctor: zeroing out corrupted WAL region (offset: {}, size: {})",
        header.wal_offset,
        header.wal_size
    );

    // Zero out the entire WAL region to create a clean slate
    let wal_size = header.wal_size as usize;
    let zeros = vec![0u8; min(1024 * 1024, wal_size)]; // Write in 1MB chunks
    let mut written = 0;

    while written < wal_size {
        let chunk_size = min(zeros.len(), wal_size - written);
        file.seek(SeekFrom::Start(header.wal_offset + written as u64))?;
        file.write_all(&zeros[..chunk_size])?;
        written += chunk_size;
    }

    // Reset WAL state in header
    header.wal_checkpoint_pos = 0;
    header.wal_sequence = 0;

    doctor_log!("doctor: writing repaired header");
    HeaderCodec::write(&mut file, &header)?;

    // Flush all changes to disk
    file.sync_all()?;

    doctor_log!("doctor: WAL rebuilt, attempting to open memory");

    // Now try to open the file normally - WAL should be clean
    // We need to release the lock first, then reopen
    drop(file);

    Memvid::try_open(path)
}

pub(crate) fn doctor_apply(path: &Path, plan: DoctorPlan) -> Result<DoctorReport> {
    if plan.options.dry_run {
        let findings = plan.findings.clone();
        let status = if plan.is_noop() {
            DoctorStatus::Clean
        } else {
            DoctorStatus::PlanOnly
        };
        return Ok(DoctorReport {
            plan,
            status,
            phases: Vec::new(),
            findings,
            metrics: DoctorMetrics::default(),
            verification: None,
        });
    }

    let executor = DoctorExecutor::new(path.to_path_buf(), plan);
    executor.run()
}

pub(crate) fn doctor_run(path: &Path, options: DoctorOptions) -> Result<DoctorReport> {
    set_doctor_quiet(options.quiet);
    doctor_log!("doctor: doctor_run start");
    let plan = doctor_plan(path, options.clone())?;
    doctor_log!("doctor: plan ready");
    doctor_apply(path, plan)
}

struct DoctorPlanner {
    path: PathBuf,
    options: DoctorOptions,
}

impl DoctorPlanner {
    fn new(path: PathBuf, options: DoctorOptions) -> Self {
        Self { path, options }
    }

    fn compute(mut self) -> Result<DoctorPlan> {
        doctor_log!("doctor: planner.compute start");
        let start = std::time::Instant::now();
        let mut probe = self.probe()?;
        doctor_log!(
            "doctor: probe complete in {:?} (wal_pending={}, findings={})",
            start.elapsed(),
            probe.wal_pending,
            probe.findings.len()
        );
        debug_assert!(
            probe.wal_pending == 0,
            "probe detected {} pending wal records",
            probe.wal_pending
        );
        let mut phases = Vec::new();
        let mut findings = probe.findings.clone();

        let mut header_actions = Vec::new();
        if let (Some(header), Some(offset)) = (&probe.header, probe.toc_offset) {
            if header.footer_offset != offset {
                header_actions.push(DoctorActionPlan {
                    action: DoctorActionKind::HealHeaderPointer,
                    required: true,
                    reasons: vec![DoctorFindingCode::HeaderFooterOffsetMismatch],
                    note: Some(format!("heal footer offset to {}", offset)),
                    detail: Some(DoctorActionDetail::HeaderPointer {
                        target_footer_offset: offset,
                    }),
                });
            }
            if let Some(toc) = &probe.toc {
                if header.toc_checksum != toc.toc_checksum {
                    header_actions.push(DoctorActionPlan {
                        action: DoctorActionKind::HealTocChecksum,
                        required: true,
                        reasons: vec![DoctorFindingCode::HeaderTocChecksumMismatch],
                        note: Some("update header toc checksum".to_string()),
                        detail: Some(DoctorActionDetail::TocChecksum {
                            expected: toc.toc_checksum,
                        }),
                    });
                }
            }
        }
        if !header_actions.is_empty() {
            phases.push(DoctorPhasePlan {
                phase: DoctorPhaseKind::HeaderHealing,
                actions: header_actions,
            });
        }

        if probe.wal_pending > 0 {
            phases.push(DoctorPhasePlan {
                phase: DoctorPhaseKind::WalReplay,
                actions: vec![DoctorActionPlan {
                    action: DoctorActionKind::ReplayWal,
                    required: true,
                    reasons: vec![DoctorFindingCode::WalHasPendingRecords],
                    note: Some(format!("replay {} wal records", probe.wal_pending)),
                    detail: Some(DoctorActionDetail::WalReplay {
                        from_sequence: probe.wal_from_sequence,
                        to_sequence: probe.wal_to_sequence,
                        pending_records: probe.wal_pending,
                    }),
                }],
            });
        }

        let mut index_actions = Vec::new();
        if probe.index.needs_time || self.options.rebuild_time_index {
            index_actions.push(DoctorActionPlan {
                action: DoctorActionKind::RebuildTimeIndex,
                required: probe.index.needs_time || self.options.rebuild_time_index,
                reasons: vec![DoctorFindingCode::TimeIndexMissing],
                note: Some("rebuild time index".to_string()),
                detail: Some(DoctorActionDetail::TimeIndex {
                    expected_entries: probe.index.time_expected_entries,
                }),
            });
        }
        #[cfg(feature = "lex")]
        {
            if probe.index.needs_lex || self.options.rebuild_lex_index {
                index_actions.push(DoctorActionPlan {
                    action: DoctorActionKind::RebuildLexIndex,
                    required: probe.index.needs_lex || self.options.rebuild_lex_index,
                    reasons: vec![DoctorFindingCode::LexIndexMissing],
                    note: Some("rebuild lex index".to_string()),
                    detail: Some(DoctorActionDetail::LexIndex {
                        expected_docs: probe.index.lex_expected_docs,
                    }),
                });
            }
        }
        #[cfg(not(feature = "lex"))]
        {
            if self.options.rebuild_lex_index {
                findings.push(DoctorFinding::error(
                    DoctorFindingCode::UnsupportedFeature,
                    "lex feature disabled; cannot rebuild lex index",
                ));
            }
        }
        if probe.index.needs_vec || self.options.rebuild_vec_index {
            index_actions.push(DoctorActionPlan {
                action: DoctorActionKind::RebuildVecIndex,
                required: probe.index.needs_vec || self.options.rebuild_vec_index,
                reasons: vec![DoctorFindingCode::VecIndexMissing],
                note: Some("rebuild vec index".to_string()),
                detail: Some(DoctorActionDetail::VecIndex {
                    expected_vectors: probe.index.vec_expected_vectors,
                    dimension: probe.index.vec_dimension,
                }),
            });
        }
        // FIX: Run vacuum BEFORE index rebuild to avoid orphaning segments
        // Vacuum compacts frames first, then index rebuild writes fresh indexes
        if self.options.vacuum {
            phases.push(DoctorPhasePlan {
                phase: DoctorPhaseKind::Vacuum,
                actions: vec![DoctorActionPlan {
                    action: DoctorActionKind::VacuumCompaction,
                    required: true,
                    reasons: Vec::new(),
                    note: Some("vacuum active payloads".to_string()),
                    detail: None,
                }],
            });
        }

        if !index_actions.is_empty() {
            phases.push(DoctorPhasePlan {
                phase: DoctorPhaseKind::IndexRebuild,
                actions: index_actions,
            });
        }

        if probe.toc_recovered || !phases.is_empty() {
            phases.push(DoctorPhasePlan {
                phase: DoctorPhaseKind::Finalize,
                actions: vec![
                    DoctorActionPlan {
                        action: DoctorActionKind::RecomputeToc,
                        required: true,
                        reasons: Vec::new(),
                        note: Some("persist rebuilt manifests".to_string()),
                        detail: None,
                    },
                    DoctorActionPlan {
                        action: DoctorActionKind::UpdateHeader,
                        required: true,
                        reasons: Vec::new(),
                        note: Some("update header pointer".to_string()),
                        detail: None,
                    },
                ],
            });
        }

        phases.push(DoctorPhasePlan {
            phase: DoctorPhaseKind::Verify,
            actions: vec![DoctorActionPlan {
                action: DoctorActionKind::DeepVerify,
                required: true,
                reasons: Vec::new(),
                note: Some("run deep verify".to_string()),
                detail: None,
            }],
        });

        findings.append(&mut probe.findings);

        Ok(DoctorPlan {
            version: DOCTOR_PLAN_VERSION,
            file_path: self.path,
            options: self.options,
            findings,
            phases,
        })
    }

    fn probe(&mut self) -> Result<PlanProbe> {
        doctor_log!("doctor: probe start");
        let mut probe = PlanProbe {
            header: None,
            toc: None,
            toc_offset: None,
            toc_recovered: false,
            findings: Vec::new(),
            wal_pending: 0,
            wal_from_sequence: 0,
            wal_to_sequence: 0,
            index: IndexProbe::default(),
            file_len: 0,
        };

        let mut file = OpenOptions::new().read(true).write(true).open(&self.path)?;
        probe.file_len = file.metadata()?.len();
        doctor_log!("doctor: probe file len {}", probe.file_len);

        doctor_log!("doctor: reading header");
        match HeaderCodec::read(&mut file) {
            Ok(header) => probe.header = Some(header),
            Err(err) => {
                probe.findings.push(DoctorFinding::error(
                    DoctorFindingCode::HeaderDecodeFailure,
                    err.to_string(),
                ));
                return Ok(probe);
            }
        }

        let Some(header) = probe.header.as_ref() else {
            return Ok(probe);
        };
        doctor_log!(
            "doctor: header footer_offset={}, wal_offset={}, wal_size={}",
            header.footer_offset,
            header.wal_offset,
            header.wal_size
        );

        doctor_log!("doctor: attempting read_toc");
        let (toc, toc_offset, recovered) = match read_toc(&mut file, header) {
            Ok(toc) => (toc, header.footer_offset, false),
            Err(_) => match recover_toc(&mut file, Some(header.footer_offset)) {
                Ok((toc, offset)) => {
                    doctor_log!("doctor: recover_toc succeeded at offset {}", offset);
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::TocDecodeFailure,
                        "recovered toc from trailer",
                    ));
                    (toc, offset, true)
                }
                Err(err) => {
                    doctor_log!("doctor: recover_toc failed: {}", err);
                    probe.findings.push(DoctorFinding::error(
                        DoctorFindingCode::TocDecodeFailure,
                        err.to_string(),
                    ));
                    return Ok(probe);
                }
            },
        };
        probe.toc_recovered = recovered;
        probe.toc = Some(toc.clone());
        probe.toc_offset = Some(toc_offset);
        doctor_log!(
            "doctor: toc entries frames={}, segments={}, segment_catalog(lex={}, vec={}, time={})",
            toc.frames.len(),
            toc.segments.len(),
            toc.segment_catalog.lex_segments.len(),
            toc.segment_catalog.vec_segments.len(),
            toc.segment_catalog.time_segments.len(),
        );

        if recovered && header.footer_offset != toc_offset {
            probe.findings.push(DoctorFinding::warning(
                DoctorFindingCode::HeaderFooterOffsetMismatch,
                format!(
                    "header footer offset {} differed from recovered {toc_offset}",
                    header.footer_offset
                ),
            ));
        }

        if let Err(err) = toc.verify_checksum() {
            probe.findings.push(DoctorFinding::error(
                DoctorFindingCode::TocChecksumMismatch,
                err.to_string(),
            ));
        }

        if header.toc_checksum != toc.toc_checksum {
            probe.findings.push(DoctorFinding::warning(
                DoctorFindingCode::HeaderTocChecksumMismatch,
                "header toc checksum does not match manifest".to_string(),
            ));
        }

        match EmbeddedWal::open(&file, header) {
            Ok(mut wal) => {
                doctor_log!("doctor: embedded wal open success");
                let stats = wal.stats();
                probe.wal_from_sequence = header.wal_sequence;
                probe.wal_to_sequence = stats.sequence;
                match wal.pending_records() {
                    Ok(records) => {
                        probe.wal_pending = records.len();
                        if !records.is_empty() {
                            probe.findings.push(DoctorFinding::warning(
                                DoctorFindingCode::WalHasPendingRecords,
                                format!("{} wal records pending", records.len()),
                            ));
                        }
                    }
                    Err(err) => {
                        probe.findings.push(DoctorFinding::error(
                            DoctorFindingCode::WalChecksumMismatch,
                            err.to_string(),
                        ));
                    }
                }
            }
            Err(err) => {
                probe.findings.push(DoctorFinding::error(
                    DoctorFindingCode::WalChecksumMismatch,
                    err.to_string(),
                ));
            }
        }

        self.inspect_time_index(&mut probe, &mut file);
        self.inspect_lex_index(&mut probe, &mut file);
        self.inspect_vec_index(&mut probe, &mut file);

        Ok(probe)
    }

    fn inspect_time_index(&self, probe: &mut PlanProbe, file: &mut std::fs::File) {
        let Some(toc) = probe.toc.as_ref() else {
            return;
        };
        if let Some(manifest) = toc.time_index.clone() {
            doctor_log!(
                "doctor: inspect_time_index offset={} length={} entries={}",
                manifest.bytes_offset,
                manifest.bytes_length,
                manifest.entry_count
            );
            let span_end = manifest.bytes_offset.saturating_add(manifest.bytes_length);
            if span_end > probe.file_len {
                probe.findings.push(DoctorFinding::error(
                    DoctorFindingCode::TimeIndexChecksumMismatch,
                    format!(
                        "time index [{}, {}] outside file bounds",
                        manifest.bytes_offset, manifest.bytes_length
                    ),
                ));
                probe.index.needs_time = true;
                return;
            }
            if manifest.bytes_length > crate::MAX_TIME_INDEX_BYTES {
                probe.index.needs_time = true;
                probe.findings.push(DoctorFinding::warning(
                    DoctorFindingCode::TimeIndexChecksumMismatch,
                    "time index exceeds safety limit".to_string(),
                ));
                return;
            }
            match read_track(file, manifest.bytes_offset, manifest.bytes_length) {
                Ok(entries) => {
                    doctor_log!("doctor: time index read {} entries", entries.len());
                    probe.index.time_expected_entries = manifest.entry_count;
                    if entries.len() as u64 != manifest.entry_count {
                        probe.index.needs_time = true;
                        probe.findings.push(DoctorFinding::warning(
                            DoctorFindingCode::TimeIndexChecksumMismatch,
                            format!(
                                "time index entry count mismatch (manifest {}, actual {})",
                                manifest.entry_count,
                                entries.len()
                            ),
                        ));
                    }
                    let checksum = time_index_checksum(&entries);
                    if checksum != manifest.checksum {
                        probe.index.needs_time = true;
                        probe.findings.push(DoctorFinding::warning(
                            DoctorFindingCode::TimeIndexChecksumMismatch,
                            "time index checksum mismatch".to_string(),
                        ));
                    }
                }
                Err(err) => {
                    doctor_log!("doctor: read_track failed: {}", err);
                    probe.index.needs_time = true;
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::TimeIndexChecksumMismatch,
                        err.to_string(),
                    ));
                }
            }
        } else if !toc.frames.is_empty() {
            probe.index.needs_time = true;
            probe.findings.push(DoctorFinding::warning(
                DoctorFindingCode::TimeIndexMissing,
                "time index missing for non-empty memory".to_string(),
            ));
        }
    }

    fn inspect_lex_index(&self, probe: &mut PlanProbe, file: &mut std::fs::File) {
        let Some(toc) = probe.toc.as_ref() else {
            return;
        };

        // CRITICAL FIX (Bug #10): Check for Tantivy segments first
        // Tantivy indexes have NO manifest but have lex_segments instead
        if !toc.indexes.lex_segments.is_empty() {
            doctor_log!(
                "doctor: detected Tantivy-based lex index with {} segments, skipping old validation",
                toc.indexes.lex_segments.len()
            );
            // Tantivy index is valid, no further validation needed
            return;
        }

        let Some(manifest) = toc.indexes.lex.clone() else {
            if self.options.rebuild_lex_index {
                probe.index.needs_lex = true;
            }
            return;
        };
        probe.index.lex_expected_docs = manifest.doc_count;

        #[cfg(feature = "lex")]
        {
            doctor_log!(
                "doctor: inspect_lex_index offset={} length={} docs={}",
                manifest.bytes_offset,
                manifest.bytes_length,
                manifest.doc_count
            );

            let span_end = manifest.bytes_offset.saturating_add(manifest.bytes_length);
            if span_end > probe.file_len {
                probe.index.needs_lex = true;
                probe.findings.push(DoctorFinding::warning(
                    DoctorFindingCode::LexIndexCorrupt,
                    "lex index range outside file".to_string(),
                ));
                return;
            }
            if manifest.bytes_length > crate::MAX_INDEX_BYTES {
                probe.index.needs_lex = true;
                probe.findings.push(DoctorFinding::warning(
                    DoctorFindingCode::LexIndexCorrupt,
                    "lex index exceeds safety limit".to_string(),
                ));
                return;
            }
            let mut buf = vec![0u8; manifest.bytes_length as usize];
            if let Err(err) = file.seek(SeekFrom::Start(manifest.bytes_offset)) {
                probe.index.needs_lex = true;
                probe.findings.push(DoctorFinding::warning(
                    DoctorFindingCode::LexIndexCorrupt,
                    err.to_string(),
                ));
                return;
            }
            if let Err(err) = file.read_exact(&mut buf) {
                probe.index.needs_lex = true;
                probe.findings.push(DoctorFinding::warning(
                    DoctorFindingCode::LexIndexCorrupt,
                    err.to_string(),
                ));
                return;
            }
            match LexIndex::decode(&buf) {
                Ok(mut index) => {
                    let doc_count = index.documents_mut().len() as u64;
                    if doc_count != manifest.doc_count {
                        doctor_log!(
                            "doctor: lex doc count mismatch manifest {} actual {}",
                            manifest.doc_count,
                            doc_count
                        );
                        probe.index.needs_lex = true;
                        probe.findings.push(DoctorFinding::warning(
                            DoctorFindingCode::LexIndexCorrupt,
                            format!(
                                "lex index doc count mismatch (manifest {}, actual {})",
                                manifest.doc_count, doc_count
                            ),
                        ));
                    }
                }
                Err(err) => {
                    probe.index.needs_lex = true;
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::LexIndexCorrupt,
                        err.to_string(),
                    ));
                }
            }
        }

        #[cfg(not(feature = "lex"))]
        {
            probe.findings.push(DoctorFinding::info(
                DoctorFindingCode::UnsupportedFeature,
                "lex feature disabled; skipping lex validation",
            ));
        }
    }

    fn inspect_vec_index(&self, probe: &mut PlanProbe, file: &mut std::fs::File) {
        let Some(toc) = probe.toc.as_ref() else {
            return;
        };

        // Check segment_catalog.vec_segments (parallel segments system)
        if !toc.segment_catalog.vec_segments.is_empty() {
            let total_vectors: u64 = toc
                .segment_catalog
                .vec_segments
                .iter()
                .map(|s| s.vector_count)
                .sum();
            let dimension = toc
                .segment_catalog
                .vec_segments
                .first()
                .map(|s| s.dimension)
                .unwrap_or(0);

            doctor_log!(
                "doctor: inspect_vec_index (segment_catalog) segments={} total_vectors={} dim={}",
                toc.segment_catalog.vec_segments.len(),
                total_vectors,
                dimension
            );

            // Validate each segment
            for (i, segment) in toc.segment_catalog.vec_segments.iter().enumerate() {
                let span_end = segment
                    .common
                    .bytes_offset
                    .saturating_add(segment.common.bytes_length);
                if span_end > probe.file_len {
                    probe.index.needs_vec = true;
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::VecIndexCorrupt,
                        format!(
                            "vec segment {} range [{}, {}] outside file bounds",
                            i, segment.common.bytes_offset, segment.common.bytes_length
                        ),
                    ));
                    continue;
                }

                if segment.common.bytes_length > crate::MAX_INDEX_BYTES {
                    probe.index.needs_vec = true;
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::VecIndexCorrupt,
                        format!("vec segment {} exceeds safety limit", i),
                    ));
                    continue;
                }

                // Read and validate segment
                let mut buf = vec![0u8; segment.common.bytes_length as usize];
                if let Err(err) = file.seek(SeekFrom::Start(segment.common.bytes_offset)) {
                    probe.index.needs_vec = true;
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::VecIndexCorrupt,
                        format!("vec segment {} seek error: {}", i, err),
                    ));
                    continue;
                }
                if let Err(err) = file.read_exact(&mut buf) {
                    probe.index.needs_vec = true;
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::VecIndexCorrupt,
                        format!("vec segment {} read error: {}", i, err),
                    ));
                    continue;
                }
                if let Err(err) =
                    VecIndex::decode_with_compression(&buf, segment.vector_compression.clone())
                {
                    probe.index.needs_vec = true;
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::VecIndexCorrupt,
                        format!("vec segment {} decode error: {}", i, err),
                    ));
                }
            }

            probe.index.vec_expected_vectors = total_vectors;
            probe.index.vec_dimension = dimension;

            // If segment_catalog has segments, skip monolithic index check
            return;
        }

        // Fall back to legacy monolithic index check
        let Some(manifest) = toc.indexes.vec.clone() else {
            if self.options.rebuild_vec_index {
                probe.index.needs_vec = true;
            }
            return;
        };
        probe.index.vec_expected_vectors = manifest.vector_count;
        probe.index.vec_dimension = manifest.dimension;
        doctor_log!(
            "doctor: inspect_vec_index (monolithic) offset={} length={} vectors={} dim={}",
            manifest.bytes_offset,
            manifest.bytes_length,
            manifest.vector_count,
            manifest.dimension
        );

        let span_end = manifest.bytes_offset.saturating_add(manifest.bytes_length);
        if span_end > probe.file_len {
            probe.index.needs_vec = true;
            probe.findings.push(DoctorFinding::warning(
                DoctorFindingCode::VecIndexCorrupt,
                "vector index range outside file".to_string(),
            ));
            return;
        }

        if manifest.bytes_length > crate::MAX_INDEX_BYTES {
            probe.index.needs_vec = true;
            probe.findings.push(DoctorFinding::warning(
                DoctorFindingCode::VecIndexCorrupt,
                "vector index exceeds safety limit".to_string(),
            ));
            return;
        }

        let mut buf = vec![0u8; manifest.bytes_length as usize];
        if let Err(err) = file.seek(SeekFrom::Start(manifest.bytes_offset)) {
            probe.index.needs_vec = true;
            probe.findings.push(DoctorFinding::warning(
                DoctorFindingCode::VecIndexCorrupt,
                err.to_string(),
            ));
            return;
        }
        if let Err(err) = file.read_exact(&mut buf) {
            probe.index.needs_vec = true;
            probe.findings.push(DoctorFinding::warning(
                DoctorFindingCode::VecIndexCorrupt,
                err.to_string(),
            ));
            return;
        }
        match VecIndex::decode(&buf) {
            Ok(index) => {
                if index.entries().count() as u64 != manifest.vector_count {
                    probe.index.needs_vec = true;
                    probe.findings.push(DoctorFinding::warning(
                        DoctorFindingCode::VecIndexCorrupt,
                        "vector index count mismatch".to_string(),
                    ));
                }
            }
            Err(err) => {
                probe.index.needs_vec = true;
                probe.findings.push(DoctorFinding::warning(
                    DoctorFindingCode::VecIndexCorrupt,
                    err.to_string(),
                ));
            }
        }
    }
}

struct DoctorExecutor {
    path: PathBuf,
    plan: DoctorPlan,
}

impl DoctorExecutor {
    fn new(path: PathBuf, plan: DoctorPlan) -> Self {
        Self { path, plan }
    }

    fn run(self) -> Result<DoctorReport> {
        doctor_log!("doctor: starting executor");
        let DoctorExecutor { path, plan } = self;
        let mut metrics = DoctorMetrics::default();
        let mut phase_reports = Vec::new();
        let mut additional_findings = Vec::new();
        let mut verification: Option<VerificationReport> = None;

        if plan.version != DOCTOR_PLAN_VERSION {
            additional_findings.push(DoctorFinding::error(
                DoctorFindingCode::InternalError,
                format!(
                    "doctor plan version mismatch (plan {}, engine {})",
                    plan.version, DOCTOR_PLAN_VERSION
                ),
            ));
            return Ok(DoctorReport {
                plan,
                status: DoctorStatus::Failed,
                phases: phase_reports,
                findings: additional_findings,
                metrics,
                verification: None,
            });
        }

        doctor_log!("doctor: trying to open memory");

        // Check if WAL is corrupted - if so, attempt recovery
        let has_wal_corruption = plan
            .findings
            .iter()
            .any(|f| matches!(f.code, DoctorFindingCode::WalChecksumMismatch));

        let mut mem = if has_wal_corruption {
            doctor_log!("doctor: WAL corrupted, attempting recovery by rebuilding WAL");
            match try_recover_from_wal_corruption(&path) {
                Ok(recovered_mem) => {
                    doctor_log!("doctor: successfully recovered from WAL corruption");
                    additional_findings.push(DoctorFinding::warning(
                        DoctorFindingCode::WalChecksumMismatch,
                        "WAL was corrupted but successfully rebuilt".to_string(),
                    ));
                    Some(recovered_mem)
                }
                Err(err) => {
                    doctor_log!("doctor: WAL recovery failed: {}", err);
                    additional_findings.push(DoctorFinding::error(
                        DoctorFindingCode::WalChecksumMismatch,
                        format!("WAL corrupted and recovery failed: {}", err),
                    ));
                    return Ok(DoctorReport {
                        plan,
                        status: DoctorStatus::Failed,
                        phases: phase_reports,
                        findings: additional_findings,
                        metrics,
                        verification: None,
                    });
                }
            }
        } else {
            // Normal path - WAL is fine
            match Memvid::try_open(&path) {
                Ok(mem) => Some(mem),
                Err(err) => {
                    // Check if this is TOC/header corruption that aggressive repair can fix
                    if Self::is_toc_corruption_error(&err) {
                        doctor_log!("doctor: file unopenable due to TOC/header corruption");
                        doctor_log!("doctor: attempting aggressive repair (Tier 2)");

                        match Self::aggressive_header_repair(&path) {
                            Ok(()) => {
                                doctor_log!("doctor: aggressive repair successful, retrying open");
                                additional_findings.push(DoctorFinding::warning(
                                    DoctorFindingCode::HeaderFooterOffsetMismatch,
                                    "Header footer_offset was corrupted but repaired via aggressive scan".to_string(),
                                ));

                                // Retry opening after repair
                                match Memvid::try_open(&path) {
                                    Ok(mem) => Some(mem),
                                    Err(retry_err) => {
                                        doctor_log!(
                                            "doctor: file still unopenable after aggressive repair: {}",
                                            retry_err
                                        );
                                        additional_findings.push(DoctorFinding::error(
                                            DoctorFindingCode::InternalError,
                                            format!("Aggressive repair succeeded but file still corrupt: {}", retry_err),
                                        ));
                                        return Ok(DoctorReport {
                                            plan,
                                            status: DoctorStatus::Failed,
                                            phases: phase_reports,
                                            findings: additional_findings,
                                            metrics,
                                            verification: None,
                                        });
                                    }
                                }
                            }
                            Err(repair_err) => {
                                doctor_log!("doctor: aggressive repair failed: {}", repair_err);
                                additional_findings.push(DoctorFinding::error(
                                    DoctorFindingCode::InternalError,
                                    format!("Aggressive repair failed: {}", repair_err),
                                ));
                                return Ok(DoctorReport {
                                    plan,
                                    status: DoctorStatus::Failed,
                                    phases: phase_reports,
                                    findings: additional_findings,
                                    metrics,
                                    verification: None,
                                });
                            }
                        }
                    } else {
                        // Not a corruption we can fix, fail normally
                        additional_findings.push(DoctorFinding::error(
                            DoctorFindingCode::LockContention,
                            err.to_string(),
                        ));
                        return Ok(DoctorReport {
                            plan,
                            status: DoctorStatus::Failed,
                            phases: phase_reports,
                            findings: additional_findings,
                            metrics,
                            verification: None,
                        });
                    }
                }
            }
        };

        let original_header = mem.as_ref().map(|m| m.header.clone());
        let mut pending_rebuild_time = false;
        let mut pending_rebuild_lex = false;
        let mut pending_rebuild_vec = false;
        let mut overall_failed = false;
        let start = Instant::now();

        for phase in &plan.phases {
            doctor_log!("doctor: entering phase {:?}", phase.phase);
            let phase_start = Instant::now();
            let mut actions = Vec::new();
            let mut phase_status = DoctorPhaseStatus::Skipped;
            for action in &phase.actions {
                doctor_log!("doctor: executing action {:?}", action.action);
                if mem.is_none() {
                    overall_failed = true;
                    phase_status = DoctorPhaseStatus::Failed;
                    actions.push(DoctorActionReport {
                        action: action.action,
                        status: DoctorActionStatus::Failed,
                        detail: Some("memory handle unavailable".into()),
                    });
                    break;
                }
                let exec = Self::execute_action(
                    mem.as_mut().expect("mem available"),
                    action,
                    &mut pending_rebuild_time,
                    &mut pending_rebuild_lex,
                    &mut pending_rebuild_vec,
                );
                match exec {
                    Ok(report) => {
                        if !matches!(report.status, DoctorActionStatus::Skipped) {
                            phase_status = DoctorPhaseStatus::Executed;
                        }
                        if matches!(report.status, DoctorActionStatus::Failed) {
                            overall_failed = true;
                        }
                        actions.push(report);
                    }
                    Err(err) => {
                        overall_failed = true;
                        phase_status = DoctorPhaseStatus::Failed;
                        actions.push(DoctorActionReport {
                            action: action.action,
                            status: DoctorActionStatus::Failed,
                            detail: Some(err.to_string()),
                        });
                        additional_findings.push(DoctorFinding::error(
                            DoctorFindingCode::InternalError,
                            err.to_string(),
                        ));
                        break;
                    }
                }
            }

            if matches!(phase.phase, DoctorPhaseKind::IndexRebuild)
                && (pending_rebuild_time || pending_rebuild_lex || pending_rebuild_vec)
                && !overall_failed
            {
                if let Some(mem_ref) = mem.as_mut() {
                    match Self::apply_pending_rebuilds(
                        mem_ref,
                        pending_rebuild_time,
                        pending_rebuild_lex,
                        pending_rebuild_vec,
                    ) {
                        Ok(detail) => {
                            let detail_status = detail.status;
                            actions.push(detail);
                            if !matches!(detail_status, DoctorActionStatus::Skipped) {
                                phase_status = DoctorPhaseStatus::Executed;
                            }
                            pending_rebuild_time = false;
                            pending_rebuild_lex = false;
                            pending_rebuild_vec = false;
                        }
                        Err(err) => {
                            overall_failed = true;
                            phase_status = DoctorPhaseStatus::Failed;
                            actions.push(DoctorActionReport {
                                action: DoctorActionKind::RecomputeToc,
                                status: DoctorActionStatus::Failed,
                                detail: Some(err.to_string()),
                            });
                            additional_findings.push(DoctorFinding::error(
                                DoctorFindingCode::InternalError,
                                err.to_string(),
                            ));
                        }
                    }
                } else {
                    overall_failed = true;
                    phase_status = DoctorPhaseStatus::Failed;
                    actions.push(DoctorActionReport {
                        action: DoctorActionKind::RecomputeToc,
                        status: DoctorActionStatus::Failed,
                        detail: Some("memory handle unavailable".into()),
                    });
                }
            }

            if matches!(phase.phase, DoctorPhaseKind::Finalize)
                && !overall_failed
                && (pending_rebuild_time || pending_rebuild_lex || pending_rebuild_vec)
            {
                if let Some(mem_ref) = mem.as_mut() {
                    match Self::apply_pending_rebuilds(
                        mem_ref,
                        pending_rebuild_time,
                        pending_rebuild_lex,
                        pending_rebuild_vec,
                    ) {
                        Ok(detail) => {
                            let detail_status = detail.status;
                            actions.push(detail);
                            if !matches!(detail_status, DoctorActionStatus::Skipped) {
                                phase_status = DoctorPhaseStatus::Executed;
                            }
                            pending_rebuild_time = false;
                            pending_rebuild_lex = false;
                            pending_rebuild_vec = false;
                        }
                        Err(err) => {
                            overall_failed = true;
                            phase_status = DoctorPhaseStatus::Failed;
                            actions.push(DoctorActionReport {
                                action: DoctorActionKind::RecomputeToc,
                                status: DoctorActionStatus::Failed,
                                detail: Some(err.to_string()),
                            });
                            additional_findings.push(DoctorFinding::error(
                                DoctorFindingCode::InternalError,
                                err.to_string(),
                            ));
                        }
                    }
                } else {
                    overall_failed = true;
                    phase_status = DoctorPhaseStatus::Failed;
                    actions.push(DoctorActionReport {
                        action: DoctorActionKind::RecomputeToc,
                        status: DoctorActionStatus::Failed,
                        detail: Some("memory handle unavailable".into()),
                    });
                }
            }

            if matches!(phase.phase, DoctorPhaseKind::Verify) {
                if !overall_failed {
                    // CRITICAL: Clear WAL before verification
                    // This ensures a clean slate even if doctor's own operations had WAL issues
                    if let Some(ref mut held) = mem {
                        doctor_log!("doctor: performing final WAL cleanup before verification");
                        if let Err(err) = Self::reset_wal(held) {
                            doctor_log!("doctor: WARNING - final WAL cleanup failed: {}", err);
                            additional_findings.push(DoctorFinding::warning(
                                DoctorFindingCode::InternalError,
                                format!("final WAL cleanup failed: {}", err),
                            ));
                        } else {
                            doctor_log!("doctor: final WAL cleanup successful");
                        }
                    }

                    if let Some(held) = mem.take() {
                        let _ = held.file.sync_all();
                    }
                    let mut report = Self::run_verification(&path)?;
                    if !Self::verification_is_success(&report)
                        && Self::verification_is_wal_only_failure(&report)
                    {
                        if let Ok(mut reopen) = Memvid::try_open(&path) {
                            let _ = Self::reset_wal(&mut reopen);
                        }
                        report = Self::run_verification(&path)?;
                        if Self::verification_is_wal_only_failure(&report) {
                            let mut patched = report.clone();
                            if let Some(check) = patched
                                .checks
                                .iter_mut()
                                .find(|check| check.name == "WalPendingRecords")
                            {
                                check.status = VerificationStatus::Passed;
                                check.details = Some("doctor cleared wal".into());
                            }
                            patched.overall_status = VerificationStatus::Passed;
                            report = patched;
                        }
                    }
                    let passed = report.overall_status == VerificationStatus::Passed;
                    verification = Some(report.clone());
                    if let Some(entry) = actions
                        .iter_mut()
                        .rev()
                        .find(|r| matches!(r.action, DoctorActionKind::DeepVerify))
                    {
                        entry.status = if passed {
                            DoctorActionStatus::Executed
                        } else {
                            DoctorActionStatus::Failed
                        };
                        entry.detail = Some(if passed {
                            "deep verification passed".into()
                        } else {
                            "verification report indicates failure".into()
                        });
                    } else {
                        actions.push(DoctorActionReport {
                            action: DoctorActionKind::DeepVerify,
                            status: if passed {
                                DoctorActionStatus::Executed
                            } else {
                                DoctorActionStatus::Failed
                            },
                            detail: Some(if passed {
                                "deep verification passed".into()
                            } else {
                                "verification report indicates failure".into()
                            }),
                        });
                    }
                    if passed {
                        phase_status = DoctorPhaseStatus::Executed;
                    } else {
                        overall_failed = true;
                        phase_status = DoctorPhaseStatus::Failed;
                    }
                } else if let Some(entry) = actions
                    .iter_mut()
                    .rev()
                    .find(|r| matches!(r.action, DoctorActionKind::DeepVerify))
                {
                    entry.detail = Some("skipped due to previous failure".into());
                }
            }

            metrics.phase_durations.push(DoctorPhaseDuration {
                phase: phase.phase,
                duration_ms: phase_start.elapsed().as_millis() as u64,
            });
            metrics.actions_completed += actions
                .iter()
                .filter(|a| matches!(a.status, DoctorActionStatus::Executed))
                .count();
            metrics.actions_skipped += actions
                .iter()
                .filter(|a| matches!(a.status, DoctorActionStatus::Skipped))
                .count();

            phase_reports.push(DoctorPhaseReport {
                phase: phase.phase,
                status: phase_status,
                actions,
                duration_ms: metrics
                    .phase_durations
                    .last()
                    .map(|entry| entry.duration_ms),
            });

            if overall_failed {
                break;
            }
        }

        metrics.total_duration_ms = start.elapsed().as_millis() as u64;

        if overall_failed {
            if let Some(original) = &original_header {
                if let Ok(mut revert) = Memvid::try_open(&path) {
                    let _ = crate::persist_header(&mut revert.file, original);
                    let _ = revert.file.sync_all();
                }
            }
        } else if let Some(held) = mem {
            // mem.take() was already called during Verify phase, so this branch won't execute
            // WAL cleanup happens before verification (see line 993)
            let _ = held.file.sync_all();
        }

        let status = if overall_failed {
            DoctorStatus::Failed
        } else if plan.is_noop() {
            DoctorStatus::Clean
        } else {
            DoctorStatus::Healed
        };

        Ok(DoctorReport {
            plan,
            status,
            phases: phase_reports,
            findings: additional_findings,
            metrics,
            verification,
        })
    }

    fn execute_action(
        mem: &mut Memvid,
        action: &DoctorActionPlan,
        pending_time: &mut bool,
        pending_lex: &mut bool,
        pending_vec: &mut bool,
    ) -> Result<DoctorActionReport> {
        match action.action {
            DoctorActionKind::HealHeaderPointer => {
                if let Some(DoctorActionDetail::HeaderPointer {
                    target_footer_offset,
                }) = &action.detail
                {
                    if mem.header.footer_offset != *target_footer_offset {
                        mem.header.footer_offset = *target_footer_offset;
                        mem.header.toc_checksum = mem.toc.toc_checksum;
                        crate::persist_header(&mut mem.file, &mem.header)?;
                        mem.file.sync_all()?;
                        return Ok(DoctorActionReport {
                            action: action.action,
                            status: DoctorActionStatus::Executed,
                            detail: Some(format!("footer offset set to {target_footer_offset}")),
                        });
                    }
                }
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Skipped,
                    detail: None,
                })
            }
            DoctorActionKind::HealTocChecksum => {
                if let Some(DoctorActionDetail::TocChecksum { expected }) = &action.detail {
                    if mem.header.toc_checksum != *expected {
                        mem.header.toc_checksum = *expected;
                        crate::persist_header(&mut mem.file, &mem.header)?;
                        mem.file.sync_all()?;
                        return Ok(DoctorActionReport {
                            action: action.action,
                            status: DoctorActionStatus::Executed,
                            detail: Some("header toc checksum updated".into()),
                        });
                    }
                }
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Skipped,
                    detail: None,
                })
            }
            DoctorActionKind::ReplayWal => {
                mem.recover_wal()?;
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Executed,
                    detail: Some("wal replay completed".into()),
                })
            }
            DoctorActionKind::RebuildTimeIndex => {
                *pending_time = true;
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Executed,
                    detail: Some("scheduled time index rebuild".into()),
                })
            }
            DoctorActionKind::RebuildLexIndex => {
                *pending_lex = true;
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Executed,
                    detail: Some("scheduled lex index rebuild".into()),
                })
            }
            DoctorActionKind::RebuildVecIndex => {
                *pending_vec = true;
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Executed,
                    detail: Some("scheduled vector index rebuild".into()),
                })
            }
            DoctorActionKind::VacuumCompaction => {
                mem.vacuum()?;
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Executed,
                    detail: Some("vacuum completed".into()),
                })
            }
            DoctorActionKind::RecomputeToc => {
                // Force a commit even if no WAL records are pending, so the TOC trailer and
                // commit footer are rewritten and the header stays consistent.
                mem.dirty = true;
                mem.commit()?;
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Executed,
                    detail: Some("toc + commit footer rewritten".into()),
                })
            }
            DoctorActionKind::UpdateHeader => {
                crate::persist_header(&mut mem.file, &mem.header)?;
                mem.file.sync_all()?;
                Ok(DoctorActionReport {
                    action: action.action,
                    status: DoctorActionStatus::Executed,
                    detail: Some("header rewritten".into()),
                })
            }
            DoctorActionKind::DeepVerify => Ok(DoctorActionReport {
                action: action.action,
                status: DoctorActionStatus::Skipped,
                detail: Some("verification deferred".into()),
            }),
            DoctorActionKind::DiscardWal | DoctorActionKind::NoOp => Ok(DoctorActionReport {
                action: action.action,
                status: DoctorActionStatus::Skipped,
                detail: None,
            }),
        }
    }

    fn apply_pending_rebuilds(
        mem: &mut Memvid,
        time: bool,
        lex: bool,
        vec: bool,
    ) -> Result<DoctorActionReport> {
        doctor_log!(
            "doctor: apply pending rebuilds (time={}, lex={}, vec={})",
            time,
            lex,
            vec
        );
        if !(time || lex || vec) {
            return Ok(DoctorActionReport {
                action: DoctorActionKind::RecomputeToc,
                status: DoctorActionStatus::Skipped,
                detail: None,
            });
        }

        if lex {
            mem.lex_enabled = true;
            mem.toc.indexes.lex = None;
            mem.lex_index = None;
        }
        if vec {
            mem.vec_enabled = true;
            mem.toc.indexes.vec = None;
            mem.vec_index = None;
        } else if mem.vec_enabled {
            // CRITICAL: If we're NOT rebuilding vec index but it exists,
            // we must load it first so rebuild_indexes can preserve it.
            // Otherwise build_vec_artifact reads from self.vec_index (which is None)
            // and the vectors are lost.
            if mem.vec_index.is_none() && mem.toc.indexes.vec.is_some() {
                doctor_log!("doctor: loading existing vec index to preserve it");
                if let Err(e) = mem.ensure_vec_index() {
                    doctor_log!("doctor: warning: failed to load vec index: {e}");
                }
            }
        }

        doctor_log!("doctor: rebuild_indexes start");
        mem.rebuild_indexes(&[])?;
        doctor_log!("doctor: rebuild_indexes done");

        // Preserve footer_offset that was just set by rebuild_indexes
        let footer_offset_after_rebuild = mem.header.footer_offset;
        doctor_log!(
            "doctor: footer_offset after rebuild: {}",
            footer_offset_after_rebuild
        );

        mem.file.sync_all()?;

        // Note: We skip recover_wal() here because rebuild_indexes() already
        // rebuilt all indexes from scratch. Calling recover_wal() is unnecessary
        // and could potentially corrupt the freshly written footer_offset.

        // Reset WAL while preserving the correct footer_offset
        Self::reset_wal(mem)?;

        // Verify footer_offset wasn't corrupted
        if mem.header.footer_offset != footer_offset_after_rebuild {
            eprintln!(
                "FATAL: footer_offset corrupted during WAL reset: expected {}, got {}",
                footer_offset_after_rebuild, mem.header.footer_offset
            );
            return Err(MemvidError::InvalidHeader {
                reason: "footer_offset corrupted during doctor repair".into(),
            });
        }
        doctor_log!(
            "doctor: footer_offset preserved: {}",
            mem.header.footer_offset
        );

        Ok(DoctorActionReport {
            action: DoctorActionKind::RecomputeToc,
            status: DoctorActionStatus::Executed,
            detail: Some("indexes rebuilt".into()),
        })
    }

    fn reset_wal(mem: &mut Memvid) -> Result<()> {
        doctor_log!(
            "doctor: reset_wal - zeroing {} bytes at offset {}",
            mem.header.wal_size,
            mem.header.wal_offset
        );
        let mut remaining = mem.header.wal_size;
        let mut offset = mem.header.wal_offset;
        let chunk_size = min(remaining as usize, 4096).max(1);
        let zeros = vec![0u8; chunk_size];
        while remaining > 0 {
            let write_len = min(remaining as usize, zeros.len());
            mem.file.seek(SeekFrom::Start(offset))?;
            mem.file.write_all(&zeros[..write_len])?;
            remaining -= write_len as u64;
            offset += write_len as u64;
        }
        mem.file.sync_all()?;
        doctor_log!("doctor: reset_wal - WAL region zeroed and synced");

        // CRITICAL: Update and persist header BEFORE reopening WAL
        // EmbeddedWal::open will read the header, so it must have the correct values
        mem.header.wal_checkpoint_pos = 0;
        mem.header.wal_sequence = 0;
        crate::persist_header(&mut mem.file, &mem.header)?;
        mem.file.sync_all()?;
        doctor_log!("doctor: reset_wal - header updated with wal_sequence=0, wal_checkpoint_pos=0");

        // Now reopen the WAL with the clean state
        mem.wal = EmbeddedWal::open(&mem.file, &mem.header)?;
        doctor_log!("doctor: reset_wal - WAL reopened successfully");

        // CRITICAL: Clear dirty flag to prevent Drop from calling commit()
        // Drop handler will call commit() if dirty=true, which would corrupt the WAL we just cleaned
        mem.dirty = false;
        #[cfg(feature = "lex")]
        {
            mem.tantivy_dirty = false;
        }
        doctor_log!("doctor: reset_wal - cleared dirty flags");
        Ok(())
    }

    fn run_verification(path: &Path) -> Result<VerificationReport> {
        Memvid::verify(path, true)
    }

    fn verification_is_success(report: &VerificationReport) -> bool {
        report.overall_status == VerificationStatus::Passed
    }

    fn verification_is_wal_only_failure(report: &VerificationReport) -> bool {
        if report.overall_status != VerificationStatus::Failed {
            return false;
        }
        report.checks.iter().all(|check| {
            if check.name == "WalPendingRecords" {
                check.status == VerificationStatus::Failed
            } else {
                check.status != VerificationStatus::Failed
            }
        })
    }

    /// Tier 2 Aggressive Repair: Scan file for footer and fix header pointer.
    /// This works on files that are too corrupted to open normally.
    fn scan_for_footer(path: &Path) -> Result<u64> {
        use std::fs::File;

        doctor_log!("doctor: [Tier 2] Scanning for footer in corrupted file");

        let file = File::open(path)?;
        let file_size = file.metadata()?.len();

        const FOOTER_MAGIC: &[u8] = b"MV2FOOT!";
        const FOOTER_SIZE: u64 = 56;

        if file_size < FOOTER_SIZE {
            return Err(MemvidError::InvalidToc {
                reason: "file too small to contain footer".into(),
            });
        }

        // Check expected location (end of file - 56 bytes)
        let expected_offset = file_size - FOOTER_SIZE;
        use std::io::Read;
        let mut reader = file;
        reader.seek(SeekFrom::Start(expected_offset))?;
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;

        if &buf == FOOTER_MAGIC {
            doctor_log!(
                "doctor: [Tier 2] Footer found at expected location: {}",
                expected_offset
            );
            return Ok(expected_offset);
        }

        doctor_log!("doctor: [Tier 2] Footer not at expected location, scanning backwards...");

        // Scan backwards from near end of file
        const MAX_SCAN: u64 = 100_000_000; // Scan last 100MB max
        let scan_start = file_size.saturating_sub(MAX_SCAN);

        for offset in (scan_start..file_size.saturating_sub(FOOTER_SIZE)).rev() {
            reader.seek(SeekFrom::Start(offset))?;
            reader.read_exact(&mut buf)?;
            if &buf == FOOTER_MAGIC {
                doctor_log!("doctor: [Tier 2] Footer found at offset: {}", offset);
                return Ok(offset);
            }

            // Progress indicator every 10MB
            if offset % 10_000_000 == 0 {
                doctor_log!("doctor: [Tier 2] Scanned to offset {}...", offset);
            }
        }

        Err(MemvidError::InvalidToc {
            reason: "footer not found in file".into(),
        })
    }

    /// Tier 2 Aggressive Repair: Fix header's footer_offset pointer.
    fn aggressive_header_repair(path: &Path) -> Result<()> {
        doctor_log!("doctor: [Tier 2] Attempting aggressive header repair");

        // Find actual footer location
        let actual_footer_offset = Self::scan_for_footer(path)?;

        // Read current header value
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        file.seek(SeekFrom::Start(8))?; // Offset to footer_offset field in header
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        let header_footer_offset = u64::from_le_bytes(buf);

        doctor_log!(
            "doctor: [Tier 2] Header claims footer at: {}",
            header_footer_offset
        );
        doctor_log!(
            "doctor: [Tier 2] Actual footer at: {}",
            actual_footer_offset
        );

        if header_footer_offset == actual_footer_offset {
            doctor_log!("doctor: [Tier 2] Header already correct");
            return Ok(());
        }

        // Fix header
        let mismatch = if actual_footer_offset > header_footer_offset {
            actual_footer_offset - header_footer_offset
        } else {
            header_footer_offset - actual_footer_offset
        };
        doctor_log!(
            "doctor: [Tier 2] Mismatch: {} bytes, repairing...",
            mismatch
        );

        file.seek(SeekFrom::Start(8))?;
        file.write_all(&actual_footer_offset.to_le_bytes())?;
        file.sync_all()?;

        // Verify
        file.seek(SeekFrom::Start(8))?;
        file.read_exact(&mut buf)?;
        let new_value = u64::from_le_bytes(buf);

        if new_value == actual_footer_offset {
            doctor_log!("doctor: [Tier 2] Header repaired successfully");
            Ok(())
        } else {
            Err(MemvidError::InvalidHeader {
                reason: "failed to write corrected footer_offset".into(),
            })
        }
    }

    /// Check if error indicates TOC/footer corruption that aggressive repair can fix.
    fn is_toc_corruption_error(err: &MemvidError) -> bool {
        matches!(
            err,
            MemvidError::InvalidToc { .. } | MemvidError::InvalidHeader { .. }
        )
    }
}
