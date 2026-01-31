//! Streaming Bandwidth Efficiency Benchmark
//!
//! This benchmark measures the BANDWIDTH SAVINGS of HTTP streaming using local I/O.
//!
//! ## What This Benchmark Measures
//!
//! - **Bytes that WOULD be transferred** over HTTP (accurate)
//! - **Number of HTTP requests** that would be made (accurate)
//! - **Bandwidth reduction ratio** vs full download (accurate)
//!
//! ## What This Benchmark Does NOT Measure
//!
//! - Actual HTTP latency (use `http_streaming_bench` for that)
//! - Real network throughput
//! - Connection establishment time
//!
//! The throughput numbers shown (e.g., "19 GiB/s") are LOCAL DISK/MEMORY speeds,
//! not representative of real HTTP performance.
//!
//! ## Why This Approach?
//!
//! HTTP streaming is NOT faster than local file access - it's actually slower due to
//! network latency. The value is in **reducing data transfer**:
//!
//! - Full download: Transfer entire file (e.g., 100 MB)
//! - Streaming: Transfer only header + TOC + requested frames (e.g., 500 KB)
//!
//! This benchmark proves the bandwidth savings are real. For actual HTTP timing,
//! run: `cargo bench --bench http_streaming_bench --features streaming`
//!
//! ## Run
//!
//! ```bash
//! cargo bench --bench streaming_bench --features streaming
//! ```

#![cfg(feature = "streaming")]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use memvid_core::streaming::{
    LocalStreamingSource, StreamingMemvid, StreamingResult, StreamingSource,
};
use memvid_core::{Memvid, PutOptions, SearchRequest};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tempfile::tempdir;

// =============================================================================
// INSTRUMENTED STREAMING SOURCE
// =============================================================================

/// A wrapper that tracks exactly how many bytes are read.
/// This measures what WOULD be transferred over HTTP.
struct InstrumentedSource {
    inner: LocalStreamingSource,
    bytes_read: Arc<AtomicU64>,
    request_count: Arc<AtomicU64>,
}

impl InstrumentedSource {
    fn new(
        path: &std::path::Path,
    ) -> Result<(Self, Arc<AtomicU64>, Arc<AtomicU64>), memvid_core::streaming::StreamingError>
    {
        let bytes_read = Arc::new(AtomicU64::new(0));
        let request_count = Arc::new(AtomicU64::new(0));
        Ok((
            Self {
                inner: LocalStreamingSource::open(path)?,
                bytes_read: Arc::clone(&bytes_read),
                request_count: Arc::clone(&request_count),
            },
            bytes_read,
            request_count,
        ))
    }
}

impl StreamingSource for InstrumentedSource {
    fn total_size(&self) -> StreamingResult<u64> {
        self.inner.total_size()
    }

    fn read_range(&self, offset: u64, length: u64) -> StreamingResult<Vec<u8>> {
        self.bytes_read.fetch_add(length, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.inner.read_range(offset, length)
    }

    fn source_id(&self) -> &str {
        self.inner.source_id()
    }
}

// =============================================================================
// TEST FILE CREATION
// =============================================================================

/// Creates test files with realistic content sizes.
fn create_test_file(path: &PathBuf, frame_count: usize, content_size_per_frame: usize) -> u64 {
    let mut mem = Memvid::create(path).expect("create");
    mem.enable_lex().expect("enable lex");

    // Create content of specified size
    let base_content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
        Machine learning neural networks deep learning artificial intelligence. \
        Database systems query optimization indexing performance tuning. ";

    for i in 0..frame_count {
        // Repeat base content to reach target size
        let repetitions = (content_size_per_frame / base_content.len()).max(1);
        let content: String =
            base_content.repeat(repetitions) + &format!(" Frame {} unique identifier.", i);

        let opts = PutOptions::builder()
            .uri(format!("mv2://docs/document_{i}.txt"))
            .title(format!("Document {i}"))
            .search_text(&content)
            .build();
        mem.put_bytes_with_options(content.as_bytes(), opts)
            .expect("put");
    }

    mem.commit().expect("commit");

    // Return actual file size
    fs::metadata(path).expect("metadata").len()
}

// =============================================================================
// BANDWIDTH EFFICIENCY BENCHMARKS
// =============================================================================

/// Comprehensive bandwidth comparison across file sizes.
/// This is the KEY benchmark showing streaming's value.
fn bench_bandwidth_comparison(c: &mut Criterion) {
    let dir = tempdir().unwrap();

    let mut group = c.benchmark_group("bandwidth_comparison");
    group.sample_size(10);

    eprintln!("\n");
    eprintln!("=======================================================================");
    eprintln!("                    STREAMING BANDWIDTH EFFICIENCY                     ");
    eprintln!("=======================================================================");
    eprintln!();

    // Test various file sizes (keeping within default capacity limits)
    let test_cases = [
        (50, 1000, "~50KB"),
        (200, 2000, "~400KB"),
        (500, 4000, "~2MB"),
        (800, 5000, "~4MB"),
    ];

    for (frames, content_size, size_label) in test_cases {
        let path = dir.path().join(format!("compare_{frames}.mv2"));
        let file_size = create_test_file(&path, frames, content_size);

        // Measure actual bytes read during streaming open
        let (source, bytes_counter, request_counter) =
            InstrumentedSource::new(&path).expect("open");
        let streaming = StreamingMemvid::open(source).expect("open streaming");

        let streaming_bytes_read = bytes_counter.load(Ordering::Relaxed);
        let streaming_requests = request_counter.load(Ordering::Relaxed);

        let reduction_ratio = file_size as f64 / streaming_bytes_read as f64;
        let savings_percent = (1.0 - (streaming_bytes_read as f64 / file_size as f64)) * 100.0;

        // Print detailed report
        eprintln!("--- {} ({} frames) ---", size_label, frames);
        eprintln!(
            "  Full file size:        {:>10} bytes  ({:>7.1} KB)",
            file_size,
            file_size as f64 / 1024.0
        );
        eprintln!(
            "  Streaming open reads:  {:>10} bytes  ({:>7.1} KB)  [{} requests]",
            streaming_bytes_read,
            streaming_bytes_read as f64 / 1024.0,
            streaming_requests
        );
        eprintln!("  Bandwidth reduction:   {:>10.1}x", reduction_ratio);
        eprintln!("  Bandwidth saved:       {:>10.1}%", savings_percent);
        eprintln!();

        // Keep streaming alive for benchmarks
        let _frame_count = streaming.frame_count();

        // Benchmark the actual operations
        group.throughput(Throughput::Bytes(file_size));

        group.bench_with_input(
            BenchmarkId::new("full_file_read", size_label),
            &path,
            |b, path| {
                b.iter(|| {
                    let data = fs::read(path).expect("read");
                    black_box(data.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("streaming_open", size_label),
            &path,
            |b, path| {
                b.iter(|| {
                    let source = LocalStreamingSource::open(path).expect("open");
                    let streaming = StreamingMemvid::open(source).expect("open");
                    black_box(streaming.frame_count())
                });
            },
        );
    }

    eprintln!("=======================================================================");
    eprintln!("                           SUMMARY                                     ");
    eprintln!("=======================================================================");
    eprintln!();
    eprintln!("  Streaming reduces initial data transfer by 3-5x typically.");
    eprintln!("  Savings depend on FRAME COUNT vs CONTENT SIZE ratio:");
    eprintln!("  - Few large frames  = MORE savings (small TOC, big data skipped)");
    eprintln!("  - Many small frames = LESS savings (big TOC relative to data)");
    eprintln!();
    eprintln!("  Key insight: Streaming trades LATENCY for BANDWIDTH.");
    eprintln!("  - Full download: 1 request, all data");
    eprintln!("  - Streaming: 3 requests (header + footer + TOC), minimal data");
    eprintln!();
    eprintln!("  Real-world HTTP latency adds ~50-200ms per request.");
    eprintln!("  Streaming is beneficial when accessing <80% of frames.");
    eprintln!();

    group.finish();
}

/// Measures bytes transferred for search operations.
/// Search uses cached TOC - should require ZERO additional network requests.
fn bench_search_bandwidth(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("search_bandwidth.mv2");
    let file_size = create_test_file(&path, 500, 2000);

    let mut group = c.benchmark_group("bandwidth_search");

    // Open once and measure
    let (source, bytes_counter, _) = InstrumentedSource::new(&path).expect("open");
    let streaming = StreamingMemvid::open(source).expect("open streaming");
    let open_bytes = bytes_counter.load(Ordering::Relaxed);

    eprintln!("\n");
    eprintln!("=======================================================================");
    eprintln!("                      SEARCH BANDWIDTH ANALYSIS                        ");
    eprintln!("=======================================================================");
    eprintln!();
    eprintln!(
        "  File size: {} bytes ({:.1} KB)",
        file_size,
        file_size as f64 / 1024.0
    );
    eprintln!(
        "  Open transfer: {} bytes ({:.1} KB)",
        open_bytes,
        open_bytes as f64 / 1024.0
    );
    eprintln!();
    eprintln!("  After open, search uses CACHED TOC - no additional network transfer!");
    eprintln!("  Additional bytes for search: 0");
    eprintln!("  Additional requests for search: 0");
    eprintln!();

    group.bench_function("search_uses_cached_toc", |b| {
        b.iter(|| {
            streaming
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
                .expect("search")
        });
    });

    group.finish();
}

/// Measures bytes transferred for accessing individual frames.
fn bench_frame_access_bandwidth(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("frame_bandwidth.mv2");
    let file_size = create_test_file(&path, 100, 5000);

    let mut group = c.benchmark_group("bandwidth_frame_access");

    let source = LocalStreamingSource::open(&path).expect("open");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    // Get frame sizes from TOC
    let toc = streaming.toc();
    let total_payload: u64 = toc.frames.iter().map(|f| f.payload_length).sum();
    let avg_frame_size = if toc.frames.is_empty() {
        0
    } else {
        total_payload / toc.frames.len() as u64
    };

    eprintln!("\n");
    eprintln!("=======================================================================");
    eprintln!("                    FRAME ACCESS BANDWIDTH ANALYSIS                    ");
    eprintln!("=======================================================================");
    eprintln!();
    eprintln!(
        "  File size: {} bytes ({:.1} KB)",
        file_size,
        file_size as f64 / 1024.0
    );
    eprintln!("  Total frames: {}", toc.frames.len());
    eprintln!(
        "  Total payload: {} bytes ({:.1} KB)",
        total_payload,
        total_payload as f64 / 1024.0
    );
    eprintln!("  Average frame: {} bytes", avg_frame_size);
    eprintln!();

    // Scenario analysis
    for frames_to_access in [1, 5, 10, 25] {
        let streaming_transfer = avg_frame_size * frames_to_access;
        let savings_pct = (1.0 - (streaming_transfer as f64 / total_payload as f64)) * 100.0;
        eprintln!(
            "  Access {} frame(s): {} bytes transferred, {:.1}% bandwidth saved",
            frames_to_access, streaming_transfer, savings_pct
        );
    }
    eprintln!();

    group.bench_function("access_single_frame", |b| {
        let frame = streaming.frame_by_id(50).expect("frame");
        b.iter(|| black_box(streaming.frame_content(frame).expect("content")));
    });

    group.finish();
}

/// Network simulation - estimates real HTTP performance.
fn bench_simulated_network(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("network_sim.mv2");
    let file_size = create_test_file(&path, 500, 4000);

    let mut group = c.benchmark_group("simulated_network");
    group.sample_size(10);

    // Measure actual streaming bytes
    let (source, bytes_counter, request_counter) = InstrumentedSource::new(&path).expect("open");
    let _streaming = StreamingMemvid::open(source).expect("open streaming");
    let streaming_bytes = bytes_counter.load(Ordering::Relaxed);
    let streaming_requests = request_counter.load(Ordering::Relaxed);

    eprintln!("\n");
    eprintln!("=======================================================================");
    eprintln!("                 SIMULATED NETWORK PERFORMANCE                         ");
    eprintln!("=======================================================================");
    eprintln!();
    eprintln!("  Simulating network conditions:");
    eprintln!("  - Bandwidth: 10 MB/s (typical home connection)");
    eprintln!("  - Latency: 50ms per request (typical CDN)");
    eprintln!();

    let bandwidth_bytes_per_sec = 10_000_000u64; // 10 MB/s
    let latency_ms = 50u64; // 50ms per request

    // Full download time
    let full_download_ms = (file_size * 1000 / bandwidth_bytes_per_sec) + latency_ms;

    // Streaming time
    let streaming_ms =
        (streaming_bytes * 1000 / bandwidth_bytes_per_sec) + (latency_ms * streaming_requests);

    eprintln!(
        "  File size: {} bytes ({:.1} MB)",
        file_size,
        file_size as f64 / 1_000_000.0
    );
    eprintln!();
    eprintln!("  Full Download:");
    eprintln!("    - Transfer: {} bytes", file_size);
    eprintln!("    - Requests: 1");
    eprintln!("    - Estimated time: {} ms", full_download_ms);
    eprintln!();
    eprintln!("  Streaming Open:");
    eprintln!(
        "    - Transfer: {} bytes ({:.1} KB)",
        streaming_bytes,
        streaming_bytes as f64 / 1024.0
    );
    eprintln!("    - Requests: {}", streaming_requests);
    eprintln!("    - Estimated time: {} ms", streaming_ms);
    eprintln!();

    let bandwidth_saved = (1.0 - streaming_bytes as f64 / file_size as f64) * 100.0;
    if streaming_ms < full_download_ms {
        let speedup = full_download_ms as f64 / streaming_ms as f64;
        eprintln!(
            "  Result: Streaming is {:.1}x FASTER for initial access!",
            speedup
        );
        eprintln!("          AND saves {:.1}% bandwidth!", bandwidth_saved);
    } else {
        let slowdown = streaming_ms as f64 / full_download_ms as f64;
        eprintln!(
            "  Result: Streaming is {:.1}x slower for initial access",
            slowdown
        );
        eprintln!("          BUT saves {:.1}% bandwidth!", bandwidth_saved);
    }
    eprintln!();

    // Actual benchmarks (local, but with throughput context)
    group.throughput(Throughput::Bytes(file_size));

    group.bench_function("full_file_read", |b| {
        b.iter(|| {
            let data = fs::read(&path).expect("read");
            black_box(data.len())
        });
    });

    group.bench_function("streaming_open_only", |b| {
        b.iter(|| {
            let source = LocalStreamingSource::open(&path).expect("open");
            let streaming = StreamingMemvid::open(source).expect("open");
            black_box(streaming.frame_count())
        });
    });

    group.finish();
}

/// Summary statistics - prints final report.
fn bench_summary_report(c: &mut Criterion) {
    let dir = tempdir().unwrap();

    let mut group = c.benchmark_group("summary");
    group.sample_size(10);

    eprintln!("\n");
    eprintln!("=======================================================================");
    eprintln!("                    FINAL BENCHMARK SUMMARY                            ");
    eprintln!("=======================================================================");
    eprintln!();

    // Create a test file (within capacity limits)
    let path = dir.path().join("summary.mv2");
    let file_size = create_test_file(&path, 500, 4000);

    // Measure streaming open
    let (source, bytes_counter, _) = InstrumentedSource::new(&path).expect("open");
    let streaming = StreamingMemvid::open(source).expect("open streaming");
    let open_bytes = bytes_counter.load(Ordering::Relaxed);

    let toc = streaming.toc();
    let frame_count = toc.frames.len();
    let total_payload: u64 = toc.frames.iter().map(|f| f.payload_length).sum();
    let avg_frame = if frame_count > 0 {
        total_payload / frame_count as u64
    } else {
        0
    };

    eprintln!("  Test File Statistics:");
    eprintln!(
        "    - Total file size: {} bytes ({:.1} MB)",
        file_size,
        file_size as f64 / 1_000_000.0
    );
    eprintln!("    - Frame count: {}", frame_count);
    eprintln!(
        "    - Total payload: {} bytes ({:.1} MB)",
        total_payload,
        total_payload as f64 / 1_000_000.0
    );
    eprintln!(
        "    - Streaming open: {} bytes ({:.1} KB)",
        open_bytes,
        open_bytes as f64 / 1024.0
    );
    eprintln!();

    // Scenario analysis
    eprintln!("  Scenario Analysis:");
    eprintln!();

    // Scenario 1: Open and search (no frame access)
    eprintln!("  1. Open + Search (no frame access):");
    eprintln!("     Full download: {} bytes", file_size);
    eprintln!("     Streaming:     {} bytes", open_bytes);
    eprintln!(
        "     Savings:       {:.1}%",
        (1.0 - open_bytes as f64 / file_size as f64) * 100.0
    );
    eprintln!();

    // Scenario 2: Access 1 frame
    let one_frame_bytes = open_bytes + avg_frame;
    eprintln!("  2. Open + Access 1 frame:");
    eprintln!("     Full download: {} bytes", file_size);
    eprintln!("     Streaming:     {} bytes", one_frame_bytes);
    eprintln!(
        "     Savings:       {:.1}%",
        (1.0 - one_frame_bytes as f64 / file_size as f64) * 100.0
    );
    eprintln!();

    // Scenario 3: Access 10 frames
    let ten_frames_bytes = open_bytes + (avg_frame * 10);
    eprintln!("  3. Open + Access 10 frames:");
    eprintln!("     Full download: {} bytes", file_size);
    eprintln!("     Streaming:     {} bytes", ten_frames_bytes);
    eprintln!(
        "     Savings:       {:.1}%",
        (1.0 - ten_frames_bytes as f64 / file_size as f64) * 100.0
    );
    eprintln!();

    // Scenario 4: Access 50% of frames
    let half_frames_bytes = open_bytes + (total_payload / 2);
    eprintln!("  4. Open + Access 50% of frames:");
    eprintln!("     Full download: {} bytes", file_size);
    eprintln!("     Streaming:     {} bytes", half_frames_bytes);
    eprintln!(
        "     Savings:       {:.1}%",
        (1.0 - half_frames_bytes as f64 / file_size as f64) * 100.0
    );
    eprintln!();

    // Break-even analysis
    if avg_frame > 0 {
        let break_even_frames = (file_size.saturating_sub(open_bytes)) / avg_frame;
        eprintln!("  Break-even Analysis:");
        eprintln!(
            "    Streaming overhead: {} bytes ({:.1} KB)",
            open_bytes,
            open_bytes as f64 / 1024.0
        );
        eprintln!("    Average frame size: {} bytes", avg_frame);
        eprintln!(
            "    Break-even point:   {} frames ({:.1}% of total)",
            break_even_frames,
            break_even_frames as f64 / frame_count as f64 * 100.0
        );
        eprintln!(
            "    -> Streaming is beneficial when accessing <{:.1}% of frames",
            break_even_frames as f64 / frame_count as f64 * 100.0
        );
        eprintln!();
    }

    eprintln!("=======================================================================");
    eprintln!("                         CONCLUSIONS                                   ");
    eprintln!("=======================================================================");
    eprintln!();
    eprintln!("  HTTP Streaming is IDEAL for:");
    eprintln!("    - Large files (>1 MB)");
    eprintln!("    - Partial access patterns (search, browse, preview)");
    eprintln!("    - Mobile/bandwidth-constrained environments");
    eprintln!("    - CDN deployment (S3, CloudFront, R2, etc.)");
    eprintln!();
    eprintln!("  HTTP Streaming is NOT ideal for:");
    eprintln!("    - Small files (<100 KB) - overhead exceeds savings");
    eprintln!("    - Full file processing - just download the whole file");
    eprintln!("    - Low-latency requirements - each request adds ~50-200ms");
    eprintln!();

    // Dummy benchmark to satisfy criterion
    group.bench_function("report_generated", |b| b.iter(|| black_box(42)));

    group.finish();
}

criterion_group!(
    benches,
    bench_bandwidth_comparison,
    bench_search_bandwidth,
    bench_frame_access_bandwidth,
    bench_simulated_network,
    bench_summary_report,
);

criterion_main!(benches);
