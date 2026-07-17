//! Integration tests for Memvid search operations.
//! Tests: search (lex), timeline queries

use memvid_core::{Memvid, PutOptions, SearchRequest, TimelineQuery};
use std::num::NonZeroU64;
use tempfile::TempDir;

/// Helper to create a memory with searchable content.
fn create_searchable_memory(path: &std::path::Path) {
    let mut mem = Memvid::create(path).unwrap();
    mem.enable_lex().unwrap();

    let docs = vec![
        (
            "mv2://physics/quantum",
            "Quantum Physics",
            "Quantum mechanics describes the behavior of particles at the atomic scale",
        ),
        (
            "mv2://physics/classical",
            "Classical Mechanics",
            "Classical mechanics describes motion of macroscopic objects",
        ),
        (
            "mv2://biology/cells",
            "Cell Biology",
            "Cells are the basic building blocks of all living organisms",
        ),
        (
            "mv2://chemistry/atoms",
            "Atomic Chemistry",
            "Atoms combine to form molecules through chemical bonds",
        ),
        (
            "mv2://math/calculus",
            "Calculus",
            "Calculus studies continuous change and rates of change",
        ),
    ];

    for (uri, title, content) in docs {
        let opts = PutOptions {
            uri: Some(uri.to_string()),
            title: Some(title.to_string()),
            search_text: Some(content.to_string()),
            timestamp: Some(1700000000),
            ..Default::default()
        };
        mem.put_bytes_with_options(content.as_bytes(), opts)
            .unwrap();
    }

    mem.commit().unwrap();
}

/// Test basic lexical search.
#[test]
#[cfg(feature = "lex")]
fn search_basic_query() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    create_searchable_memory(&path);

    let mut mem = Memvid::open_read_only(&path).unwrap();
    let results = mem
        .search(SearchRequest {
            query: "quantum".to_string(),
            top_k: 10,
            snippet_chars: 200,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    assert!(!results.hits.is_empty(), "Should find quantum document");
    assert!(
        results.hits[0].uri.contains("quantum"),
        "Top result should be quantum physics"
    );
}

/// Test search with multiple results.
#[test]
#[cfg(feature = "lex")]
fn search_multiple_results() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    create_searchable_memory(&path);

    let mut mem = Memvid::open_read_only(&path).unwrap();

    // Search for "mechanics" should find both quantum and classical
    let results = mem
        .search(SearchRequest {
            query: "mechanics".to_string(),
            top_k: 10,
            snippet_chars: 200,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    assert_eq!(
        results.hits.len(),
        2,
        "Should find both mechanics documents"
    );
}

/// Test quoted search requires the exact phrase, not just all words.
#[test]
#[cfg(feature = "lex")]
fn search_quoted_phrase_requires_exact_match() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();
        mem.enable_lex().unwrap();

        for (uri, content) in [
            ("mv2://exact", "quantum mechanics is weird"),
            ("mv2://separate", "quantum effects differ from classical mechanics"),
        ] {
            let opts = PutOptions {
                uri: Some(uri.to_string()),
                search_text: Some(content.to_string()),
                ..Default::default()
            };
            mem.put_bytes_with_options(content.as_bytes(), opts).unwrap();
        }
        mem.commit().unwrap();
    }

    let mut mem = Memvid::open_read_only(&path).unwrap();
    let results = mem
        .search(SearchRequest {
            query: "\"quantum mechanics\"".to_string(),
            top_k: 10,
            snippet_chars: 200,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].uri, "mv2://exact");
}

/// Test search with top_k limit.
#[test]
#[cfg(feature = "lex")]
fn search_respects_top_k() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    // Create memory with many documents
    {
        let mut mem = Memvid::create(&path).unwrap();
        mem.enable_lex().unwrap();

        for i in 0..20 {
            let opts = PutOptions {
                uri: Some(format!("mv2://doc{}", i)),
                title: Some(format!("Document {}", i)),
                search_text: Some(format!(
                    "This document contains searchable content number {}",
                    i
                )),
                ..Default::default()
            };
            mem.put_bytes_with_options(format!("Content {}", i).as_bytes(), opts)
                .unwrap();
        }
        mem.commit().unwrap();
    }

    let mut mem = Memvid::open_read_only(&path).unwrap();
    let results = mem
        .search(SearchRequest {
            query: "document".to_string(),
            top_k: 5,
            snippet_chars: 200,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    assert_eq!(results.hits.len(), 5, "Should return exactly top_k results");
}

/// Test search with scope filter.
#[test]
#[cfg(feature = "lex")]
fn search_with_scope() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    create_searchable_memory(&path);

    let mut mem = Memvid::open_read_only(&path).unwrap();

    // Search only in physics scope
    let results = mem
        .search(SearchRequest {
            query: "mechanics".to_string(),
            top_k: 10,
            snippet_chars: 200,
            uri: None,
            scope: Some("mv2://physics/".to_string()),
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    // All results should be from physics scope
    for hit in &results.hits {
        assert!(
            hit.uri.starts_with("mv2://physics/"),
            "Results should be from physics scope"
        );
    }
}

/// Test search returns snippets.
#[test]
#[cfg(feature = "lex")]
fn search_returns_snippets() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    create_searchable_memory(&path);

    let mut mem = Memvid::open_read_only(&path).unwrap();
    let results = mem
        .search(SearchRequest {
            query: "quantum".to_string(),
            top_k: 10,
            snippet_chars: 200,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    assert!(!results.hits.is_empty());
    let hit = &results.hits[0];

    // Snippet should contain matched content
    assert!(!hit.text.is_empty(), "Hit should include text snippet");
}

/// Test search with no results.
#[test]
#[cfg(feature = "lex")]
fn search_no_results() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    create_searchable_memory(&path);

    let mut mem = Memvid::open_read_only(&path).unwrap();
    let results = mem
        .search(SearchRequest {
            query: "xyznonexistentterm".to_string(),
            top_k: 10,
            snippet_chars: 200,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    assert_eq!(results.hits.len(), 0, "Should return no results");
}

/// Test search on empty memory.
#[test]
#[cfg(feature = "lex")]
fn search_empty_memory() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();
        mem.enable_lex().unwrap();
        mem.commit().unwrap();
    }

    let mut mem = Memvid::open_read_only(&path).unwrap();
    let results = mem
        .search(SearchRequest {
            query: "anything".to_string(),
            top_k: 10,
            snippet_chars: 200,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    assert_eq!(
        results.hits.len(),
        0,
        "Empty memory should return no results"
    );
}

/// Test timeline query returns ordered results.
#[test]
fn timeline_returns_ordered() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        // Add frames with different timestamps
        let timestamps = [1700000000i64, 1700003000, 1700001000, 1700002000];

        for (i, ts) in timestamps.iter().enumerate() {
            let opts = PutOptions {
                uri: Some(format!("mv2://doc{}", i)),
                title: Some(format!("Document {}", i)),
                timestamp: Some(*ts),
                ..Default::default()
            };
            mem.put_bytes_with_options(format!("Content {}", i).as_bytes(), opts)
                .unwrap();
        }
        mem.commit().unwrap();
    }

    let mut mem = Memvid::open_read_only(&path).unwrap();
    let query = TimelineQuery::builder()
        .limit(NonZeroU64::new(10).unwrap())
        .build();
    let entries = mem.timeline(query).unwrap();

    // Verify timeline is ordered by timestamp (either ascending or descending)
    if entries.len() > 1 {
        let is_descending = entries[0].timestamp >= entries[1].timestamp;
        for i in 1..entries.len() {
            if is_descending {
                assert!(
                    entries[i - 1].timestamp >= entries[i].timestamp,
                    "Timeline should be consistently ordered (descending)"
                );
            } else {
                assert!(
                    entries[i - 1].timestamp <= entries[i].timestamp,
                    "Timeline should be consistently ordered (ascending)"
                );
            }
        }
    }
}

/// Test timeline with since filter.
#[test]
fn timeline_with_since_filter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        let timestamps = [1700000000i64, 1700001000, 1700002000, 1700003000];

        for (i, ts) in timestamps.iter().enumerate() {
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

    // Get entries since 1700001500
    let query = TimelineQuery::builder()
        .limit(NonZeroU64::new(10).unwrap())
        .since(1700001500)
        .build();
    let entries = mem.timeline(query).unwrap();

    for entry in &entries {
        assert!(
            entry.timestamp >= 1700001500,
            "All entries should be >= since timestamp"
        );
    }
}

/// Test timeline with until filter.
#[test]
fn timeline_with_until_filter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        let timestamps = [1700000000i64, 1700001000, 1700002000, 1700003000];

        for (i, ts) in timestamps.iter().enumerate() {
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

    // Get entries until 1700001500
    let query = TimelineQuery::builder()
        .limit(NonZeroU64::new(10).unwrap())
        .until(1700001500)
        .build();
    let entries = mem.timeline(query).unwrap();

    for entry in &entries {
        assert!(
            entry.timestamp <= 1700001500,
            "All entries should be <= until timestamp"
        );
    }
}

/// Test timeline respects limit.
#[test]
fn timeline_respects_limit() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.mv2");

    {
        let mut mem = Memvid::create(&path).unwrap();

        for i in 0..20 {
            let opts = PutOptions {
                uri: Some(format!("mv2://doc{}", i)),
                timestamp: Some(1700000000 + i as i64 * 1000),
                ..Default::default()
            };
            mem.put_bytes_with_options(format!("Content {}", i).as_bytes(), opts)
                .unwrap();
        }
        mem.commit().unwrap();
    }

    let mut mem = Memvid::open_read_only(&path).unwrap();
    let query = TimelineQuery::builder()
        .limit(NonZeroU64::new(5).unwrap())
        .build();
    let entries = mem.timeline(query).unwrap();

    assert_eq!(
        entries.len(),
        5,
        "Timeline should return exactly limit entries"
    );
}
