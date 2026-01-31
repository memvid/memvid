//! Integration tests for HTTP streaming support.
//!
//! These tests verify that `.mv2` files can be read through the streaming interface
//! using a local file source (which mimics HTTP range requests).

#![cfg(feature = "streaming")]

use memvid_core::streaming::{LocalStreamingSource, StreamingMemvid};
use memvid_core::{Memvid, PutOptions, SearchRequest, TimelineQuery};
use tempfile::tempdir;

/// Creates a test .mv2 file with sample content.
fn create_test_file(path: &std::path::Path, frame_count: usize) {
    let mut mem = Memvid::create(path).expect("create");

    for i in 0..frame_count {
        let content = format!(
            "Frame {} content with some searchable text about topic {}",
            i,
            i % 5
        );
        let opts = PutOptions::builder()
            .uri(format!("mv2://test/frame{i}.txt"))
            .title(format!("Frame {i}"))
            .search_text(&content)
            .build();
        mem.put_bytes_with_options(content.as_bytes(), opts)
            .expect("put");
    }

    mem.commit().expect("commit");
}

#[test]
fn streaming_opens_local_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.mv2");

    create_test_file(&path, 3);

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    assert_eq!(streaming.frame_count(), 3);

    let stats = streaming.stats();
    assert_eq!(stats.frame_count, 3);
    // Note: lex index may or may not be present depending on feature flags
}

#[test]
fn streaming_reads_frame_content() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("content.mv2");

    let content = "This is test content for streaming functionality.";

    {
        let mut mem = Memvid::create(&path).expect("create");
        mem.put_bytes(content.as_bytes()).expect("put");
        mem.commit().expect("commit");
    }

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    let frame = streaming.frame_by_id(0).expect("frame by id");
    let text = streaming.frame_content(frame).expect("frame content");

    // Content may be enriched with metadata during ingestion
    assert!(text.starts_with(content));
}

#[test]
fn streaming_reads_frame_by_uri() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("uri.mv2");

    {
        let mut mem = Memvid::create(&path).expect("create");

        let opts = PutOptions::builder()
            .uri("mv2://docs/readme.md")
            .title("README")
            .build();
        mem.put_bytes_with_options(b"README content", opts)
            .expect("put");

        let opts2 = PutOptions::builder()
            .uri("mv2://docs/license.txt")
            .title("LICENSE")
            .build();
        mem.put_bytes_with_options(b"LICENSE content", opts2)
            .expect("put");

        mem.commit().expect("commit");
    }

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    let frame = streaming
        .frame_by_uri("mv2://docs/readme.md")
        .expect("frame by uri");
    assert_eq!(frame.title.as_deref(), Some("README"));

    let frame2 = streaming
        .frame_by_uri("mv2://docs/license.txt")
        .expect("frame by uri");
    assert_eq!(frame2.title.as_deref(), Some("LICENSE"));

    // Non-existent URI should error
    let err = streaming.frame_by_uri("mv2://docs/missing.txt");
    assert!(err.is_err());
}

#[test]
fn streaming_timeline_pagination() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("timeline.mv2");

    create_test_file(&path, 10);

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    // Get all entries
    let all = streaming
        .timeline(TimelineQuery::default())
        .expect("timeline");
    assert_eq!(all.len(), 10);

    // Get limited entries
    let limited = streaming
        .timeline(
            TimelineQuery::builder()
                .limit(std::num::NonZeroU64::new(3).unwrap())
                .build(),
        )
        .expect("timeline limited");
    assert_eq!(limited.len(), 3);

    // Get reversed entries
    let reversed = streaming
        .timeline(TimelineQuery::builder().reverse(true).build())
        .expect("timeline reversed");
    assert_eq!(reversed.len(), 10);
    assert!(reversed[0].timestamp >= reversed[9].timestamp);
}

#[test]
fn streaming_search_basic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("search.mv2");

    {
        let mut mem = Memvid::create(&path).expect("create");
        mem.enable_lex().expect("enable lex");

        let opts1 = PutOptions::builder()
            .uri("mv2://docs/ml.txt")
            .search_text("machine learning neural networks deep learning")
            .build();
        mem.put_bytes_with_options(b"ML content", opts1)
            .expect("put");

        let opts2 = PutOptions::builder()
            .uri("mv2://docs/db.txt")
            .search_text("database systems query optimization indexing")
            .build();
        mem.put_bytes_with_options(b"DB content", opts2)
            .expect("put");

        let opts3 = PutOptions::builder()
            .uri("mv2://docs/web.txt")
            .search_text("web development javascript typescript frontend")
            .build();
        mem.put_bytes_with_options(b"Web content", opts3)
            .expect("put");

        mem.commit().expect("commit");
    }

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    // Search for ML terms
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
        .expect("search");

    assert_eq!(response.hits.len(), 1);
    assert_eq!(response.hits[0].uri, "mv2://docs/ml.txt");

    // Search for database terms
    let response2 = streaming
        .search(SearchRequest {
            query: "database query".into(),
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
        .expect("search");

    assert_eq!(response2.hits.len(), 1);
    assert_eq!(response2.hits[0].uri, "mv2://docs/db.txt");
}

#[test]
fn streaming_search_with_uri_filter() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("filter.mv2");

    {
        let mut mem = Memvid::create(&path).expect("create");
        mem.enable_lex().expect("enable lex");

        // Both contain "rust"
        let opts1 = PutOptions::builder()
            .uri("mv2://docs/rust-guide.txt")
            .search_text("rust programming language systems")
            .build();
        mem.put_bytes_with_options(b"Rust guide", opts1)
            .expect("put");

        let opts2 = PutOptions::builder()
            .uri("mv2://blog/rust-news.txt")
            .search_text("rust updates and news")
            .build();
        mem.put_bytes_with_options(b"Rust news", opts2)
            .expect("put");

        mem.commit().expect("commit");
    }

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    // Search without filter - should find both
    let response = streaming
        .search(SearchRequest {
            query: "rust".into(),
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
        .expect("search");
    assert_eq!(response.hits.len(), 2);

    // Search with URI filter
    let response2 = streaming
        .search(SearchRequest {
            query: "rust".into(),
            top_k: 10,
            snippet_chars: 100,
            uri: Some("mv2://docs/rust-guide.txt".into()),
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
        })
        .expect("search");
    assert_eq!(response2.hits.len(), 1);
    assert_eq!(response2.hits[0].uri, "mv2://docs/rust-guide.txt");

    // Search with scope filter
    let response3 = streaming
        .search(SearchRequest {
            query: "rust".into(),
            top_k: 10,
            snippet_chars: 100,
            uri: None,
            scope: Some("mv2://blog/".into()),
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
        })
        .expect("search");
    assert_eq!(response3.hits.len(), 1);
    assert_eq!(response3.hits[0].uri, "mv2://blog/rust-news.txt");
}

#[test]
fn streaming_stats_accuracy() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stats.mv2");

    {
        let mut mem = Memvid::create(&path).expect("create");

        for i in 0..5 {
            let content = format!("Content for frame {i}");
            mem.put_bytes(content.as_bytes()).expect("put");
        }

        mem.commit().expect("commit");
    }

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    let stats = streaming.stats();

    assert_eq!(stats.frame_count, 5);
    assert_eq!(stats.active_frame_count, 5);
    assert!(stats.size_bytes > 0);
    assert!(stats.wal_bytes > 0);
}

#[test]
fn streaming_toc_access() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("toc.mv2");

    create_test_file(&path, 5);

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    let toc = streaming.toc();

    assert_eq!(toc.frames.len(), 5);

    // Verify frame metadata
    for (i, frame) in toc.frames.iter().enumerate() {
        assert_eq!(frame.id, i as u64);
        assert!(frame.uri.is_some());
        assert!(frame.title.is_some());
    }
}

#[test]
fn streaming_debug_impl() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("debug.mv2");

    create_test_file(&path, 2);

    let source = LocalStreamingSource::open(&path).expect("open source");
    let streaming = StreamingMemvid::open(source).expect("open streaming");

    let debug_str = format!("{:?}", streaming);

    assert!(debug_str.contains("StreamingMemvid"));
    assert!(debug_str.contains("frame_count"));
    assert!(debug_str.contains("2"));
}

#[test]
#[ignore] // Requires network
fn http_streaming_from_url() {
    // This test requires a real HTTP endpoint
    // Uncomment and configure URL to run manually

    // use memvid_core::streaming::HttpStreamingSource;
    // let source = HttpStreamingSource::from_url("https://example.com/test.mv2").expect("http source");
    // let streaming = StreamingMemvid::open(source).expect("open");
    // assert!(streaming.frame_count() > 0);
}

#[test]
fn real_bandwidth_benchmark() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Create test file
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.mv2");

    let mut mem = memvid_core::Memvid::create(&path).unwrap();
    let base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Machine learning neural networks. ";
    for i in 0..200 {
        let content = base.repeat(20) + &format!(" Frame {} unique.", i);
        let opts = memvid_core::PutOptions::builder()
            .uri(format!("mv2://doc_{i}.txt"))
            .search_text(&content)
            .build();
        mem.put_bytes_with_options(content.as_bytes(), opts)
            .unwrap();
    }
    mem.commit().unwrap();

    let file_size = std::fs::metadata(&path).unwrap().len();

    // Measure bytes read
    struct Counter {
        inner: memvid_core::streaming::LocalStreamingSource,
        bytes: Arc<AtomicU64>,
        reqs: Arc<AtomicU64>,
    }
    impl memvid_core::streaming::StreamingSource for Counter {
        fn total_size(&self) -> memvid_core::streaming::StreamingResult<u64> {
            self.inner.total_size()
        }
        fn read_range(&self, o: u64, l: u64) -> memvid_core::streaming::StreamingResult<Vec<u8>> {
            self.bytes.fetch_add(l, Ordering::Relaxed);
            self.reqs.fetch_add(1, Ordering::Relaxed);
            self.inner.read_range(o, l)
        }
        fn source_id(&self) -> &str {
            self.inner.source_id()
        }
    }

    let bytes = Arc::new(AtomicU64::new(0));
    let reqs = Arc::new(AtomicU64::new(0));
    let source = Counter {
        inner: memvid_core::streaming::LocalStreamingSource::open(&path).unwrap(),
        bytes: Arc::clone(&bytes),
        reqs: Arc::clone(&reqs),
    };

    let streaming = memvid_core::streaming::StreamingMemvid::open(source).unwrap();
    let bytes_read = bytes.load(Ordering::Relaxed);
    let requests = reqs.load(Ordering::Relaxed);

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════╗");
    eprintln!("║          REAL STREAMING BANDWIDTH BENCHMARK RESULTS               ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║ File size:          {:>10} bytes ({:>7.2} KB)               ║",
        file_size,
        file_size as f64 / 1024.0
    );
    eprintln!(
        "║ Frames:             {:>10}                                   ║",
        streaming.frame_count()
    );
    eprintln!("╠═══════════════════════════════════════════════════════════════════╣");
    eprintln!("║ STREAMING OPEN:                                                   ║");
    eprintln!(
        "║   Bytes transferred: {:>9} bytes ({:>7.2} KB)               ║",
        bytes_read,
        bytes_read as f64 / 1024.0
    );
    eprintln!(
        "║   HTTP requests:     {:>9}                                   ║",
        requests
    );
    eprintln!("╠═══════════════════════════════════════════════════════════════════╣");
    eprintln!("║ BANDWIDTH SAVINGS:                                                ║");
    eprintln!(
        "║   Reduction ratio:   {:>9.1}x                                   ║",
        file_size as f64 / bytes_read as f64
    );
    eprintln!(
        "║   Bandwidth saved:   {:>9.1}%                                   ║",
        (1.0 - bytes_read as f64 / file_size as f64) * 100.0
    );
    eprintln!("╠═══════════════════════════════════════════════════════════════════╣");
    eprintln!("║ SIMULATED NETWORK (10 MB/s, 50ms latency):                        ║");
    let full_ms = (file_size as f64 / 10_000_000.0 * 1000.0) + 50.0;
    let stream_ms = (bytes_read as f64 / 10_000_000.0 * 1000.0) + (50.0 * requests as f64);
    eprintln!(
        "║   Full download:     {:>9.0} ms                                 ║",
        full_ms
    );
    eprintln!(
        "║   Streaming open:    {:>9.0} ms                                 ║",
        stream_ms
    );
    if stream_ms < full_ms {
        eprintln!(
            "║   Result:            {:>9.1}x FASTER                            ║",
            full_ms / stream_ms
        );
    }
    eprintln!("╚═══════════════════════════════════════════════════════════════════╝");
    eprintln!();
}
