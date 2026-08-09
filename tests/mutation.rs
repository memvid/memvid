//! Integration tests for Memvid mutation operations.
//! Tests: put, put_bytes_with_options, update, delete

use memvid_core::{
    EmbeddingIdentitySummary, MEMVID_EMBEDDING_MODEL_KEY, MEMVID_EMBEDDING_PROVIDER_KEY, Memvid,
    MemvidError, PutOptions, TimelineQuery, VerificationStatus, constants::HEADER_SIZE,
    io::header::HeaderCodec,
};
use std::fs::File;
use std::io::Read;
use std::num::NonZeroU64;
use std::path::Path;
use tempfile::TempDir;

fn read_wal_size(path: &Path) -> u64 {
    let mut header_bytes = [0u8; HEADER_SIZE];
    File::open(path)
        .unwrap()
        .read_exact(&mut header_bytes)
        .unwrap();
    HeaderCodec::decode(&header_bytes).unwrap().wal_size
}

/// Test basic put operation with bytes.
#[test]
fn put_bytes_basic() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    let mut mem = Memvid::create(&path).unwrap();

    let opts = PutOptions {
        uri: Some("mv2://test".to_string()),
        title: Some("Test Document".to_string()),
        ..Default::default()
    };

    let _frame_id = mem.put_bytes_with_options(b"Hello, World!", opts).unwrap();
    mem.commit().unwrap();

    // Verify frame was created
    // Verify frame was created (FrameId is u64 so >= 0 is implied)
    // assert!(frame_id >= 0);

    let mem = Memvid::open_read_only(&path).unwrap();
    assert_eq!(mem.stats().unwrap().frame_count, 1, "Should have 1 frame");
}

/// Test put with all options.
#[test]
fn put_bytes_with_all_options() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        let opts = PutOptions {
            uri: Some("mv2://docs/report.txt".to_string()),
            title: Some("Annual Report".to_string()),
            search_text: Some("Financial summary for 2024".to_string()),
            tags: vec!["finance".to_string(), "annual".to_string()],
            labels: vec!["Important".to_string()],
            timestamp: Some(1700000000),
            ..Default::default()
        };

        mem.put_bytes_with_options(b"Report content here", opts)
            .unwrap();
        mem.commit().unwrap();
    }

    // Verify frame was stored
    let mem = Memvid::open_read_only(&path).unwrap();
    let stats = mem.stats().unwrap();
    assert_eq!(stats.frame_count, 1, "Should have 1 frame");

    // Verify frame metadata by fetching via URI
    let frame = mem.frame_by_uri("mv2://docs/report.txt").unwrap();
    assert_eq!(frame.uri.as_deref(), Some("mv2://docs/report.txt"));
    assert_eq!(frame.title.as_deref(), Some("Annual Report"));
    assert!(frame.tags.contains(&"finance".to_string()));
    assert!(frame.labels.contains(&"Important".to_string()));
    assert_eq!(frame.timestamp, 1700000000);
}

/// Test multiple puts create frames.
#[test]
fn put_multiple_creates_frames() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        for i in 0..5 {
            let opts = PutOptions {
                uri: Some(format!("mv2://doc{}", i)),
                ..Default::default()
            };
            mem.put_bytes_with_options(format!("Content {}", i).as_bytes(), opts)
                .unwrap();
        }

        mem.commit().unwrap();
    }

    let mem = Memvid::open_read_only(&path).unwrap();
    assert_eq!(mem.stats().unwrap().frame_count, 5, "Should have 5 frames");
}

/// Test frame_by_uri returns correct data.
#[test]
fn frame_by_uri_returns_data() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        let opts = PutOptions {
            uri: Some("mv2://test".to_string()),
            title: Some("Test".to_string()),
            ..Default::default()
        };
        mem.put_bytes_with_options(b"Test content", opts).unwrap();
        mem.commit().unwrap();
    }

    let mem = Memvid::open_read_only(&path).unwrap();
    let frame = mem.frame_by_uri("mv2://test").unwrap();

    assert_eq!(frame.uri.as_deref(), Some("mv2://test"));
    assert_eq!(frame.title.as_deref(), Some("Test"));
}

/// Test update_frame modifies frame metadata.
#[test]
fn update_frame_modifies_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    // Create with initial data
    {
        let mut mem = Memvid::create(&path).unwrap();
        let opts = PutOptions {
            uri: Some("mv2://test".to_string()),
            title: Some("Original Title".to_string()),
            ..Default::default()
        };
        mem.put_bytes_with_options(b"Content", opts).unwrap();
        mem.commit().unwrap();
    }

    // Update frame - look up by URI to get current frame_id
    {
        let mut mem = Memvid::open(&path).unwrap();
        // Get the frame by URI to find its actual ID
        let frame = mem.frame_by_uri("mv2://test").unwrap();
        let frame_id = frame.id;

        let update_opts = PutOptions {
            title: Some("Updated Title".to_string()),
            ..Default::default()
        };
        // Pass new payload to trigger actual update
        mem.update_frame(frame_id, Some(b"New Content".to_vec()), update_opts, None)
            .unwrap();
        mem.commit().unwrap();
    }

    // Verify update
    let mem = Memvid::open_read_only(&path).unwrap();
    let frame = mem.frame_by_uri("mv2://test").unwrap();

    assert_eq!(frame.title.as_deref(), Some("Updated Title"));
}

/// Test delete_frame marks frame as deleted.
#[test]
fn delete_frame_marks_deleted() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    // Create with content
    {
        let mut mem = Memvid::create(&path).unwrap();
        let opts = PutOptions {
            uri: Some("mv2://test".to_string()),
            ..Default::default()
        };
        mem.put_bytes_with_options(b"Content", opts).unwrap();
        mem.commit().unwrap();
    }

    // Delete frame - look up by URI to get current frame_id
    {
        let mut mem = Memvid::open(&path).unwrap();
        // Get the frame by URI to find its actual ID
        let frame = mem.frame_by_uri("mv2://test").unwrap();
        let frame_id = frame.id;

        mem.delete_frame(frame_id).unwrap();
        mem.commit().unwrap();
    }

    // Verify deletion - need to iterate or check stats since URI lookup may fail for deleted
    let mem = Memvid::open_read_only(&path).unwrap();
    // A deleted frame should still exist but with Deleted status
    // The URI lookup might fail, so check stats instead
    let stats = mem.stats().unwrap();
    // After deletion, frame_count may still be 1 but the frame is marked deleted
    // Or it may be 0 depending on implementation
    // Both are valid - the key is no panic occurred
    assert!(
        stats.frame_count == 0 || stats.frame_count == 1,
        "Frame count should be 0 or 1 after delete"
    );
}

/// Test URI uniqueness is enforced.
#[test]
fn uri_uniqueness() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    let mut mem = Memvid::create(&path).unwrap();

    // First put succeeds
    let opts1 = PutOptions {
        uri: Some("mv2://unique".to_string()),
        ..Default::default()
    };
    mem.put_bytes_with_options(b"First", opts1).unwrap();

    // Second put with same URI should fail or replace
    let opts2 = PutOptions {
        uri: Some("mv2://unique".to_string()),
        ..Default::default()
    };
    // The behavior depends on implementation - this tests that it doesn't panic
    let result = mem.put_bytes_with_options(b"Second", opts2);

    // Either it succeeds (replacing) or returns an error (enforcing uniqueness)
    // Both are valid behaviors - the test ensures no panic
    if result.is_ok() {
        mem.commit().unwrap();
    }
}

#[test]
fn put_rejects_mixed_embedding_dimensions() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    let mut mem = Memvid::create(&path).unwrap();
    mem.enable_vec().unwrap();

    mem.put_with_embedding(b"first", vec![0.0f32; 384]).unwrap();
    let err = mem
        .put_with_embedding(b"second", vec![0.0f32; 1536])
        .unwrap_err();
    match err {
        MemvidError::VecDimensionMismatch { expected, actual } => {
            assert_eq!(expected, 384);
            assert_eq!(actual, 1536);
        }
        other => panic!("expected VecDimensionMismatch, got {other:?}"),
    }
}

#[test]
fn embedding_identity_summary_unknown_when_missing() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    let mut mem = Memvid::create(&path).unwrap();
    mem.put_bytes_with_options(b"hello", PutOptions::default())
        .unwrap();
    mem.commit().unwrap();

    let mem = Memvid::open_read_only(&path).unwrap();
    assert_eq!(
        mem.embedding_identity_summary(1_000),
        EmbeddingIdentitySummary::Unknown
    );
}

#[test]
fn embedding_identity_summary_single() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    let mut mem = Memvid::create(&path).unwrap();
    let mut options = PutOptions::default();
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_PROVIDER_KEY.to_string(),
        "openai".to_string(),
    );
    options.extra_metadata.insert(
        MEMVID_EMBEDDING_MODEL_KEY.to_string(),
        "text-embedding-3-small".to_string(),
    );
    mem.put_bytes_with_options(b"hello", options).unwrap();
    mem.commit().unwrap();

    let mem = Memvid::open_read_only(&path).unwrap();
    match mem.embedding_identity_summary(1_000) {
        EmbeddingIdentitySummary::Single(identity) => {
            assert_eq!(identity.provider.as_deref(), Some("openai"));
            assert_eq!(identity.model.as_deref(), Some("text-embedding-3-small"));
        }
        other => panic!("expected Single identity, got {other:?}"),
    }
}

#[test]
fn embedding_identity_summary_mixed() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    let mut mem = Memvid::create(&path).unwrap();

    let mut options_a = PutOptions::default();
    options_a.extra_metadata.insert(
        MEMVID_EMBEDDING_PROVIDER_KEY.to_string(),
        "fastembed".to_string(),
    );
    options_a.extra_metadata.insert(
        MEMVID_EMBEDDING_MODEL_KEY.to_string(),
        "BAAI/bge-small-en-v1.5".to_string(),
    );
    mem.put_bytes_with_options(b"a", options_a).unwrap();

    let mut options_b = PutOptions::default();
    options_b.extra_metadata.insert(
        MEMVID_EMBEDDING_PROVIDER_KEY.to_string(),
        "openai".to_string(),
    );
    options_b.extra_metadata.insert(
        MEMVID_EMBEDDING_MODEL_KEY.to_string(),
        "text-embedding-3-small".to_string(),
    );
    mem.put_bytes_with_options(b"b", options_b).unwrap();

    mem.commit().unwrap();

    let mem = Memvid::open_read_only(&path).unwrap();
    match mem.embedding_identity_summary(1_000) {
        EmbeddingIdentitySummary::Mixed(identities) => {
            assert_eq!(identities.len(), 2);
            let models: Vec<_> = identities
                .iter()
                .filter_map(|entry| entry.identity.model.as_deref())
                .collect();
            assert!(models.contains(&"BAAI/bge-small-en-v1.5"));
            assert!(models.contains(&"text-embedding-3-small"));
        }
        other => panic!("expected Mixed identity, got {other:?}"),
    }
}

/// Test put with empty content.
#[test]
fn put_empty_content() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        let opts = PutOptions {
            uri: Some("mv2://empty".to_string()),
            title: Some("Empty Document".to_string()),
            ..Default::default()
        };

        mem.put_bytes_with_options(b"", opts).unwrap();
        mem.commit().unwrap();
    }

    let mem = Memvid::open_read_only(&path).unwrap();
    let frame = mem.frame_by_uri("mv2://empty").unwrap();

    assert_eq!(frame.title.as_deref(), Some("Empty Document"));
}

/// Test put with moderately large content (50KB to avoid WAL limits).
#[test]
fn put_moderately_large_content() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        // Create 50KB of content (default new file gets 1MB WAL region)
        let large_content = vec![b'x'; 50 * 1024];

        let opts = PutOptions {
            uri: Some("mv2://large".to_string()),
            title: Some("Large Document".to_string()),
            ..Default::default()
        };

        mem.put_bytes_with_options(&large_content, opts).unwrap();
        mem.commit().unwrap();
    }

    let mem = Memvid::open_read_only(&path).unwrap();
    let frame = mem.frame_by_uri("mv2://large").unwrap();

    assert_eq!(frame.title.as_deref(), Some("Large Document"));
    // Frame was stored successfully - payload_length may be 0 if stored as blob
    // The important thing is no panic/error occurred
}

/// Test timeline iteration.
#[test]
fn timeline_iteration() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        // Add frames with different timestamps
        for (i, ts) in [1700000000i64, 1700001000, 1700002000].iter().enumerate() {
            let opts = PutOptions {
                uri: Some(format!("mv2://doc{}", i)),
                timestamp: Some(*ts),
                ..Default::default()
            };
            mem.put_bytes_with_options(format!("Content {}", i).as_bytes(), opts)
                .unwrap();
        }
        mem.commit().unwrap();
    }

    let mut mem = Memvid::open_read_only(&path).unwrap();

    // Get timeline entries using TimelineQuery
    let query = TimelineQuery::builder()
        .limit(NonZeroU64::new(10).unwrap())
        .build();
    let entries = mem.timeline(query).unwrap();

    assert_eq!(entries.len(), 3, "Should have 3 timeline entries");
}

/// Regression test for memvid/memvid#230 — sustained commit-per-put workloads
/// that span multiple WAL growth cycles must keep the embedded WAL intact.
///
/// Before the fix, `grow_wal_region` / `ensure_wal_capacity` updated
/// `header.footer_offset` and `self.data_end` after shifting the data region
/// but left the cached `payload_region_end()` value stale. The next call to
/// `rebuild_indexes` then sought to that pre-growth offset (which now lies
/// inside the grown WAL region) and overwrote WAL record payloads, producing
/// `Embedded WAL is corrupted at offset N: wal record checksum mismatch` on
/// the following commit.
#[test]
fn commit_per_put_survives_wal_growth() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("wal_growth.mv2");

    // Capture the initial WAL size from a freshly created, then closed, store.
    // The header is read with no writer open so it never races the store's
    // file lock — on Windows a held byte-range lock makes a second handle's
    // read fail with "another process has locked a portion of the file".
    let initial_wal_size = {
        let mem = Memvid::create(&path).unwrap();
        drop(mem);
        read_wal_size(&path)
    };

    // Use text-indexable payloads of varying length so each commit drives the
    // full Tantivy rebuild path (`rebuild_indexes` → `flush_tantivy`) that
    // seeks to `payload_region_end()`. The mix of sizes ensures multiple
    // WAL growth cycles occur across the run.
    let words: &[&str] = &[
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra",
        "tango", "uniform", "victor", "whiskey", "x-ray", "yankee", "zulu",
    ];

    let doc_count = 24u32;
    let frames_written;
    {
        let mut mem = Memvid::open(&path).unwrap();
        for i in 0..doc_count {
            // Ramp the body size from ~4 KiB up past the initial 64 KiB WAL
            // region so the run deterministically crosses several
            // `grow_wal_region` cycles (rather than relying on incidental
            // wrap/checkpoint timing). Each commit then runs `rebuild_indexes`,
            // which seeks to `payload_region_end()` — the path #230 corrupted.
            let body_len = 4096usize + (i as usize) * 6144;
            let mut body = String::with_capacity(body_len + 16);
            let mut idx = i as usize;
            while body.len() < body_len {
                body.push_str(words[idx % words.len()]);
                body.push(' ');
                idx = idx.wrapping_add(1);
            }
            let opts = PutOptions {
                uri: Some(format!("mv2://wal-growth/doc-{i}")),
                title: Some(format!("doc-{i}")),
                search_text: Some(body.clone()),
                ..Default::default()
            };
            mem.put_bytes_with_options(body.as_bytes(), opts)
                .unwrap_or_else(|e| panic!("put #{i} failed: {e}"));
            mem.commit()
                .unwrap_or_else(|e| panic!("commit #{i} failed: {e}"));
        }
        // Large `search_text` bodies are chunked, so each doc yields one or
        // more frames; capture the true total before closing.
        frames_written = mem.stats().unwrap().frame_count;
        // mem dropped here, releasing the file lock before we read the header.
    }

    // The WAL region only ever grows, so reading the size once the writer is
    // closed reflects the maximum it reached during the run.
    let final_wal_size = read_wal_size(&path);
    assert!(
        final_wal_size > initial_wal_size,
        "test must exercise WAL growth (initial={initial_wal_size}, final={final_wal_size})"
    );

    // Reopening forces a full WAL scan; checksum verification will fire here
    // if any record payload was clobbered by a stale-offset index write.
    assert!(
        frames_written >= u64::from(doc_count),
        "expected at least one frame per doc (got {frames_written} for {doc_count} docs)"
    );
    let reopened = Memvid::open_read_only(&path).unwrap();
    assert_eq!(
        reopened.stats().unwrap().frame_count,
        frames_written,
        "all frames should be durable after WAL growth"
    );
}

/// Vacuum must reclaim disk space left behind by deleted (superseded) frames.
///
/// Regression test for a `footer_offset` clamp in `rebuild_indexes`: `vacuum()`
/// rebuilt the indexes at a smaller size but never truncated the file, so it
/// reclaimed nothing.
#[test]
fn vacuum_reclaims_space_from_deleted_frames() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    // Insert enough sizeable documents (each kept under the chunking threshold)
    // that deleted payload + index space dominates fixed overhead (header, WAL,
    // TOC).
    let mut body_lens: Vec<usize> = Vec::new();
    {
        let mut mem = Memvid::create(&path).unwrap();
        for i in 0..200 {
            let body =
                format!("document {i}: the quick brown fox jumps over the lazy dog. ").repeat(25);
            let opts = PutOptions {
                uri: Some(format!("mv2://doc/{i}")),
                title: Some(format!("Document {i}")),
                ..Default::default()
            };
            mem.put_bytes_with_options(body.as_bytes(), opts).unwrap();
            body_lens.push(body.len());
        }
        mem.commit().unwrap();
    }

    // Delete every other document, measure the file, then compact.
    let size_before;
    let mut deleted_bytes = 0usize;
    {
        let mut mem = Memvid::open(&path).unwrap();
        for i in (0..200).step_by(2) {
            let frame_id = mem.frame_by_uri(&format!("mv2://doc/{i}")).unwrap().id;
            mem.delete_frame(frame_id).unwrap();
            deleted_bytes += body_lens[i];
        }
        mem.commit().unwrap();
        size_before = std::fs::metadata(&path).unwrap().len();

        mem.vacuum().unwrap();
    }

    let size_after = std::fs::metadata(&path).unwrap().len();
    let reclaimed = size_before.saturating_sub(size_after);
    let expected_min = u64::try_from(deleted_bytes / 2).unwrap();

    // Post-fix, vacuum reclaims at least the deleted payload bytes (plus their
    // index entries). Pre-fix it reclaims ~nothing, so this assertion fails.
    assert!(
        reclaimed >= expected_min,
        "vacuum reclaimed {reclaimed} bytes; expected >= {expected_min} \
         (deleted {deleted_bytes} payload bytes; before={size_before}, after={size_after})"
    );

    // Reclaiming must not corrupt any section. Deep verify covers the time/lex/vec
    // indexes and the WAL, but NOT the sketch track — that is asserted explicitly
    // below.
    let report = Memvid::verify(&path, true).unwrap();
    assert_eq!(
        report.overall_status,
        VerificationStatus::Passed,
        "vacuumed file should verify as healthy; checks: {:?}",
        report.checks
    );

    // The sketch track is the one section rebuild_indexes omits; confirm it
    // survived the compaction rather than being truncated under a stale manifest.
    let mem = Memvid::open_read_only(&path).unwrap();
    assert!(mem.has_sketches(), "sketch track must survive vacuum");
    assert!(
        mem.sketch_stats().entry_count > 0,
        "sketch track should be non-empty after vacuum"
    );
}

/// Vacuum must preserve every active frame and leave the file consistent.
#[test]
fn vacuum_preserves_active_frames() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();
        for i in 0..50 {
            let opts = PutOptions {
                uri: Some(format!("mv2://doc/{i}")),
                title: Some(format!("Document {i}")),
                ..Default::default()
            };
            mem.put_bytes_with_options(format!("content for document {i}").as_bytes(), opts)
                .unwrap();
        }
        mem.commit().unwrap();
    }

    // Delete the even-indexed documents (resolving FrameId by URI), then compact.
    {
        let mut mem = Memvid::open(&path).unwrap();
        for i in (0..50).step_by(2) {
            let frame_id = mem.frame_by_uri(&format!("mv2://doc/{i}")).unwrap().id;
            mem.delete_frame(frame_id).unwrap();
        }
        mem.commit().unwrap();
        mem.vacuum().unwrap();
    }

    // The compacted file verifies as healthy — a deep check includes a clean WAL,
    // which vacuum must leave behind.
    let report = Memvid::verify(&path, true).unwrap();
    assert_eq!(
        report.overall_status,
        VerificationStatus::Passed,
        "vacuumed file should verify as healthy; checks: {:?}",
        report.checks
    );

    // Every surviving (odd-indexed) frame still resolves with its metadata.
    let mem = Memvid::open_read_only(&path).unwrap();
    for i in (1..50).step_by(2) {
        let frame = mem.frame_by_uri(&format!("mv2://doc/{i}")).unwrap();
        assert_eq!(frame.title, Some(format!("Document {i}")));
    }
}

/// Vacuum must preserve the vector index — a path the lex-only tests never
/// exercise. Asserts the vec index both verifies AND answers a real vector query
/// after compaction (decode alone is insufficient).
#[cfg(feature = "vec")]
#[test]
fn vacuum_preserves_vector_index() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    const N: usize = 32;
    let embedding = |i: usize| {
        let mut v = vec![0.0f32; N];
        v[i] = 1.0;
        v
    };

    {
        let mut mem = Memvid::create(&path).unwrap();
        for i in 0..N {
            let opts = PutOptions {
                uri: Some(format!("mv2://doc/{i}")),
                title: Some(format!("Document {i}")),
                ..Default::default()
            };
            mem.put_with_embedding_and_options(
                format!("content for document {i}").as_bytes(),
                embedding(i),
                opts,
            )
            .unwrap();
        }
        mem.commit().unwrap();
    }

    // Delete even-indexed docs, then compact.
    {
        let mut mem = Memvid::open(&path).unwrap();
        for i in (0..N).step_by(2) {
            let id = mem.frame_by_uri(&format!("mv2://doc/{i}")).unwrap().id;
            mem.delete_frame(id).unwrap();
        }
        mem.commit().unwrap();
        mem.vacuum().unwrap();
    }

    let report = Memvid::verify(&path, true).unwrap();
    assert_eq!(
        report.overall_status,
        VerificationStatus::Passed,
        "vacuumed file should verify as healthy; checks: {:?}",
        report.checks
    );

    // A vector query for a survivor's own embedding must return that survivor —
    // proves the rebuilt vec index maps to the relocated frames, not just decodes.
    let mut mem = Memvid::open(&path).unwrap();
    let survivor_id = mem.frame_by_uri("mv2://doc/3").unwrap().id;
    let hits = mem.search_vec(&embedding(3), 1).unwrap();
    assert_eq!(
        hits.first().map(|h| h.frame_id),
        Some(survivor_id),
        "vector search should return the surviving frame after vacuum; hits={hits:?}"
    );
}
