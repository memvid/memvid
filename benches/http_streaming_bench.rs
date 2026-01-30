//! Real HTTP Streaming Benchmark
//!
//! This benchmark tests ACTUAL HTTP streaming performance, not simulated.
//!
//! ## Requirements
//!
//! Set the `MEMVID_TEST_URL` environment variable to a .mv2 file hosted on an HTTP server.
//!
//! ## Quick Local Test Setup
//!
//! ```bash
//! # Terminal 1: Start a simple HTTP server
//! cd /tmp && python3 -m http.server 8080
//!
//! # Terminal 2: Create test file and run benchmark
//! cargo run --example http_streaming --features streaming -- create /tmp/test.mv2 100
//! export MEMVID_TEST_URL="http://localhost:8080/test.mv2"
//! cargo bench --bench http_streaming_bench --features streaming
//! ```
//!
//! ## CDN Test Setup
//!
//! ```bash
//! # Upload to S3
//! aws s3 cp test.mv2 s3://your-bucket/test.mv2 --acl public-read
//! export MEMVID_TEST_URL="https://your-bucket.s3.amazonaws.com/test.mv2"
//!
//! # Run benchmark
//! cargo bench --bench http_streaming_bench --features streaming
//! ```
//!
//! ## What This Benchmark Measures
//!
//! Unlike the bandwidth benchmark (streaming_bench.rs) which measures bytes transferred,
//! this benchmark measures:
//!
//! - Real HTTP latency
//! - Actual network throughput
//! - Range request overhead
//! - Connection establishment time

#![cfg(feature = "streaming")]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use memvid_core::SearchRequest;
use memvid_core::streaming::{HttpConfig, HttpStreamingSource, StreamingMemvid, StreamingSource};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// =============================================================================
// INSTRUMENTED HTTP SOURCE
// =============================================================================

/// Wrapper that tracks bytes and requests for HTTP source.
struct InstrumentedHttpSource {
    inner: HttpStreamingSource,
    bytes_read: Arc<AtomicU64>,
    request_count: Arc<AtomicU64>,
}

impl InstrumentedHttpSource {
    #[allow(clippy::type_complexity)]
    fn new(
        url: &str,
    ) -> Result<(Self, Arc<AtomicU64>, Arc<AtomicU64>), Box<dyn std::error::Error>> {
        let bytes_read = Arc::new(AtomicU64::new(0));
        let request_count = Arc::new(AtomicU64::new(0));

        // Use shorter timeout for benchmarks
        let config = HttpConfig {
            timeout_secs: 60,
            max_retries: 2,
            ..Default::default()
        };

        let inner = HttpStreamingSource::with_config(url, config)?;

        Ok((
            Self {
                inner,
                bytes_read: Arc::clone(&bytes_read),
                request_count: Arc::clone(&request_count),
            },
            bytes_read,
            request_count,
        ))
    }
}

impl StreamingSource for InstrumentedHttpSource {
    fn total_size(&self) -> memvid_core::streaming::StreamingResult<u64> {
        self.inner.total_size()
    }

    fn read_range(
        &self,
        offset: u64,
        length: u64,
    ) -> memvid_core::streaming::StreamingResult<Vec<u8>> {
        self.bytes_read.fetch_add(length, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.inner.read_range(offset, length)
    }

    fn source_id(&self) -> &str {
        self.inner.source_id()
    }
}

// =============================================================================
// HTTP STREAMING BENCHMARKS
// =============================================================================

/// Benchmark real HTTP streaming open operation.
fn bench_http_streaming_open(c: &mut Criterion) {
    let url = match std::env::var("MEMVID_TEST_URL") {
        Ok(u) => u,
        Err(_) => {
            eprintln!("\n");
            eprintln!("=======================================================================");
            eprintln!("              SKIPPING HTTP BENCHMARK - NO URL SET                     ");
            eprintln!("=======================================================================");
            eprintln!();
            eprintln!("  Set MEMVID_TEST_URL environment variable to run HTTP benchmarks.");
            eprintln!();
            eprintln!("  Quick local test:");
            eprintln!("    1. python3 -m http.server 8080 &");
            eprintln!(
                "    2. cargo run --example http_streaming --features streaming -- create /tmp/test.mv2 100"
            );
            eprintln!("    3. export MEMVID_TEST_URL=\"http://localhost:8080/test.mv2\"");
            eprintln!("    4. cargo bench --bench http_streaming_bench --features streaming");
            eprintln!();
            return;
        }
    };

    eprintln!("\n");
    eprintln!("=======================================================================");
    eprintln!("                   REAL HTTP STREAMING BENCHMARK                       ");
    eprintln!("=======================================================================");
    eprintln!();
    eprintln!("  URL: {}", url);
    eprintln!();

    // Verify connectivity first
    let config = HttpConfig {
        timeout_secs: 30,
        max_retries: 1,
        ..Default::default()
    };

    let source = match HttpStreamingSource::with_config(&url, config.clone()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  ERROR: Failed to connect to URL: {}", e);
            eprintln!("  Ensure the server is running and the URL is correct.");
            eprintln!();
            return;
        }
    };

    let file_size = match source.total_size() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  ERROR: Failed to get file size: {}", e);
            return;
        }
    };

    eprintln!(
        "  File size: {} bytes ({:.2} KB, {:.2} MB)",
        file_size,
        file_size as f64 / 1024.0,
        file_size as f64 / 1_000_000.0
    );
    eprintln!();

    // Warm-up and measure initial latency
    eprintln!("  Measuring HTTP characteristics...");
    let start = Instant::now();
    let (warm_source, bytes_counter, request_counter) = match InstrumentedHttpSource::new(&url) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  ERROR: {}", e);
            return;
        }
    };
    let connection_time = start.elapsed();

    let start = Instant::now();
    let streaming = match StreamingMemvid::open(warm_source) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  ERROR: Failed to open streaming: {}", e);
            return;
        }
    };
    let open_time = start.elapsed();

    let bytes_read = bytes_counter.load(Ordering::Relaxed);
    let requests = request_counter.load(Ordering::Relaxed);

    eprintln!("  Connection time: {:?}", connection_time);
    eprintln!("  Open time:       {:?}", open_time);
    eprintln!("  Total time:      {:?}", connection_time + open_time);
    eprintln!();
    eprintln!(
        "  Bytes transferred: {} bytes ({:.1} KB)",
        bytes_read,
        bytes_read as f64 / 1024.0
    );
    eprintln!("  HTTP requests:     {}", requests);
    eprintln!(
        "  Avg request size:  {} bytes",
        if requests > 0 {
            bytes_read / requests
        } else {
            0
        }
    );
    eprintln!();

    // Calculate effective throughput
    let total_ms = (connection_time + open_time).as_secs_f64() * 1000.0;
    let effective_throughput = if total_ms > 0.0 {
        bytes_read as f64 / total_ms * 1000.0 // bytes per second
    } else {
        0.0
    };
    let avg_latency_ms = if requests > 0 {
        total_ms / requests as f64
    } else {
        0.0
    };

    eprintln!(
        "  Effective throughput: {:.2} KB/s ({:.2} MB/s)",
        effective_throughput / 1024.0,
        effective_throughput / 1_000_000.0
    );
    eprintln!("  Avg latency/request: {:.2} ms", avg_latency_ms);
    eprintln!();

    // Compare to full download scenario
    eprintln!("  Comparison to full download:");
    eprintln!("    Full download:    {} bytes (100%)", file_size);
    eprintln!(
        "    Streaming open:   {} bytes ({:.1}%)",
        bytes_read,
        bytes_read as f64 / file_size as f64 * 100.0
    );
    eprintln!(
        "    Bandwidth saved:  {:.1}%",
        (1.0 - bytes_read as f64 / file_size as f64) * 100.0
    );
    eprintln!();

    // Benchmark section
    let mut group = c.benchmark_group("http_streaming");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));
    group.throughput(Throughput::Bytes(bytes_read)); // Measure actual bytes transferred

    // Get frame count for label
    let frame_count = streaming.frame_count();
    let label = format!("{}frames_{}KB", frame_count, file_size / 1024);

    group.bench_with_input(BenchmarkId::new("http_open", &label), &url, |b, url| {
        b.iter(|| {
            let source = HttpStreamingSource::with_config(url, config.clone()).expect("connect");
            let streaming = StreamingMemvid::open(source).expect("open");
            black_box(streaming.frame_count())
        });
    });

    // Benchmark search (uses cached TOC, no additional HTTP requests)
    group.bench_with_input(
        BenchmarkId::new("http_search_cached", &label),
        &streaming,
        |b, streaming| {
            b.iter(|| {
                streaming
                    .search(SearchRequest {
                        query: "test".into(),
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
                    .ok()
            });
        },
    );

    group.finish();

    // Full download comparison (if file is small enough)
    if file_size < 50_000_000 {
        // < 50MB
        let mut full_group = c.benchmark_group("http_full_download");
        full_group.sample_size(10);
        full_group.measurement_time(std::time::Duration::from_secs(30));
        full_group.throughput(Throughput::Bytes(file_size));

        full_group.bench_with_input(BenchmarkId::new("full_download", &label), &url, |b, url| {
            b.iter(|| {
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(60))
                    .build()
                    .expect("client");
                let response = client.get(url.as_str()).send().expect("get");
                let data = response.bytes().expect("bytes");
                black_box(data.len())
            });
        });

        full_group.finish();
    }

    eprintln!("=======================================================================");
    eprintln!("                     HTTP BENCHMARK COMPLETE                           ");
    eprintln!("=======================================================================");
    eprintln!();
}

/// Benchmark frame access over HTTP.
fn bench_http_frame_access(c: &mut Criterion) {
    let url = match std::env::var("MEMVID_TEST_URL") {
        Ok(u) => u,
        Err(_) => return, // Skip if no URL
    };

    let config = HttpConfig {
        timeout_secs: 30,
        max_retries: 2,
        ..Default::default()
    };

    let source = match HttpStreamingSource::with_config(&url, config.clone()) {
        Ok(s) => s,
        Err(_) => return,
    };

    let streaming = match StreamingMemvid::open(source) {
        Ok(s) => s,
        Err(_) => return,
    };

    let frame_count = streaming.frame_count();
    if frame_count == 0 {
        return;
    }

    eprintln!("\n");
    eprintln!("=======================================================================");
    eprintln!("                  HTTP FRAME ACCESS BENCHMARK                          ");
    eprintln!("=======================================================================");
    eprintln!();
    eprintln!("  Total frames: {}", frame_count);
    eprintln!();

    let toc = streaming.toc();
    let total_payload: u64 = toc.frames.iter().map(|f| f.payload_length).sum();
    let avg_frame_size = if frame_count > 0 {
        total_payload / frame_count as u64
    } else {
        0
    };
    eprintln!("  Total payload: {} bytes", total_payload);
    eprintln!("  Avg frame size: {} bytes", avg_frame_size);
    eprintln!();

    let mut group = c.benchmark_group("http_frame_access");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Benchmark accessing a single frame
    let mid_frame_id = frame_count / 2;
    if let Ok(frame) = streaming.frame_by_id(mid_frame_id) {
        group.throughput(Throughput::Bytes(frame.payload_length));

        group.bench_function("single_frame_http", |b| {
            b.iter(|| black_box(streaming.frame_content(frame).ok()));
        });
    }

    group.finish();
}

/// Network quality analysis.
fn bench_http_latency_analysis(c: &mut Criterion) {
    let url = match std::env::var("MEMVID_TEST_URL") {
        Ok(u) => u,
        Err(_) => return,
    };

    eprintln!("\n");
    eprintln!("=======================================================================");
    eprintln!("                   HTTP LATENCY ANALYSIS                               ");
    eprintln!("=======================================================================");
    eprintln!();

    // Measure raw HTTP latency with minimal data
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("client");

    // HEAD request latency (no body)
    let mut head_latencies = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let _ = client.head(&url).send();
        head_latencies.push(start.elapsed());
    }

    let avg_head = head_latencies.iter().map(|d| d.as_millis()).sum::<u128>() / 5;
    let min_head = head_latencies
        .iter()
        .map(|d| d.as_millis())
        .min()
        .unwrap_or(0);
    let max_head = head_latencies
        .iter()
        .map(|d| d.as_millis())
        .max()
        .unwrap_or(0);

    eprintln!("  HEAD request latency (5 samples):");
    eprintln!("    Min: {} ms", min_head);
    eprintln!("    Avg: {} ms", avg_head);
    eprintln!("    Max: {} ms", max_head);
    eprintln!();

    // Range request latency (1 byte)
    let mut range_latencies = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let _ = client.get(&url).header("Range", "bytes=0-0").send();
        range_latencies.push(start.elapsed());
    }

    let avg_range = range_latencies.iter().map(|d| d.as_millis()).sum::<u128>() / 5;
    let min_range = range_latencies
        .iter()
        .map(|d| d.as_millis())
        .min()
        .unwrap_or(0);
    let max_range = range_latencies
        .iter()
        .map(|d| d.as_millis())
        .max()
        .unwrap_or(0);

    eprintln!("  Range request latency (1 byte, 5 samples):");
    eprintln!("    Min: {} ms", min_range);
    eprintln!("    Avg: {} ms", avg_range);
    eprintln!("    Max: {} ms", max_range);
    eprintln!();

    // Benchmark the range requests
    let mut group = c.benchmark_group("http_latency");
    group.sample_size(20);

    group.bench_function("head_request", |b| {
        b.iter(|| black_box(client.head(&url).send().ok()));
    });

    group.bench_function("range_1byte", |b| {
        b.iter(|| black_box(client.get(&url).header("Range", "bytes=0-0").send().ok()));
    });

    group.bench_function("range_4kb", |b| {
        b.iter(|| {
            black_box(
                client
                    .get(&url)
                    .header("Range", "bytes=0-4095")
                    .send()
                    .and_then(|r| r.bytes())
                    .ok(),
            )
        });
    });

    group.finish();

    eprintln!("=======================================================================");
    eprintln!();
}

criterion_group!(
    benches,
    bench_http_streaming_open,
    bench_http_frame_access,
    bench_http_latency_analysis,
);

criterion_main!(benches);
