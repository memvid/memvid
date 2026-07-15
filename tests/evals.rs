//! integration tests for the ndcg eval harness.
//!
//! - ndcg math tests: validate the formula with synthetic hits
//! - fixture tests: make sure eval_fixtures.json loads and validates
//! - end-to-end test: build a real .mv2, run the full suite, check scores

#![cfg(feature = "evals")]
#![cfg(feature = "lex")]

use memvid_core::evals::{EvalSuite, run_eval_suite};
use memvid_core::evals::ndcg::compute_ndcg;
use memvid_core::evals::ground_truth::EvalCase;
use memvid_core::types::SearchHit;
use memvid_core::{Memvid, PutOptions, Result};
use tempfile::tempdir;
use std::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::new(());


fn make_hit(uri: &str) -> SearchHit {
    SearchHit {
        rank: 0,
        frame_id: 0,
        uri: uri.to_string(),
        title: None,
        range: (0, 0),
        text: String::new(),
        matches: 0,
        chunk_range: None,
        chunk_text: None,
        score: None,
        metadata: None,
    }
}


#[test]
fn test_ndcg_math_perfect_score() {
    let hits = vec![make_hit("mv2://a.md"), make_hit("mv2://b.md")];
    let case = EvalCase {
        query: "q".into(),
        relevant_uris: vec!["mv2://a.md".into()],
        grades: vec![1.0],
    };
    let score = compute_ndcg(&hits, &case, 5);
    assert!((score - 1.0).abs() < 1e-5, "expected 1.0 got {score}");
}

#[test]
fn test_ndcg_math_zero_when_miss() {
    let hits = vec![make_hit("mv2://irrelevant.md")];
    let case = EvalCase {
        query: "q".into(),
        relevant_uris: vec!["mv2://target.md".into()],
        grades: vec![1.0],
    };
    let score = compute_ndcg(&hits, &case, 5);
    assert_eq!(score, 0.0);
}

#[test]
fn test_ndcg_math_rank_order_matters() {
    let case = EvalCase {
        query: "q".into(),
        relevant_uris: vec!["mv2://primary.md".into(), "mv2://secondary.md".into()],
        grades: vec![1.0, 0.5],
    };
    // Best doc at rank 1 → should score 1.0
    let ideal = vec![make_hit("mv2://primary.md"), make_hit("mv2://secondary.md")];
    // Best doc buried at rank 2 → should score < 1.0
    let suboptimal = vec![make_hit("mv2://secondary.md"), make_hit("mv2://primary.md")];

    let ideal_score = compute_ndcg(&ideal, &case, 5);
    let sub_score = compute_ndcg(&suboptimal, &case, 5);

    assert!((ideal_score - 1.0).abs() < 1e-5);
    assert!(ideal_score > sub_score);
}


#[test]
fn test_fixture_loads_and_validates() {
    let suite = EvalSuite::from_embedded().expect("fixture must load");
    assert!(!suite.is_empty(), "fixture must not be empty");
    assert!(suite.len() >= 20, "expect at least 20 eval cases");
}

#[test]
fn test_fixture_all_cases_have_at_least_one_relevant_uri() {
    let suite = EvalSuite::from_embedded().unwrap();
    for case in &suite.cases {
        assert!(
            !case.relevant_uris.is_empty(),
            "case '{}' has no relevant URIs",
            case.query
        );
        assert!(
            !case.grades.is_empty(),
            "case '{}' has no grades",
            case.query
        );
    }
}

#[test]
fn test_fixture_grades_are_valid() {
    let suite = EvalSuite::from_embedded().unwrap();
    for case in &suite.cases {
        for &grade in &case.grades {
            assert!(
                (0.0..=1.0).contains(&grade),
                "case '{}' has out-of-range grade {grade}",
                case.query
            );
        }
    }
}

#[test]
fn test_filter_by_keyword() {
    let suite = EvalSuite::from_embedded().unwrap();
    let filtered = suite.filter_by_keyword("crash");
    assert!(!filtered.is_empty(), "expected at least one 'crash' case");
    for case in &filtered.cases {
        assert!(case.query.to_lowercase().contains("crash"));
    }
}

#[test]
fn test_eval_suite_against_real_search() -> Result<()> {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("eval_test.mv2");

    {
        let mut mem = Memvid::create(&path)?;
        mem.enable_lex()?;
        insert_eval_corpus(&mut mem)?;
        mem.commit()?;
    }

    let mut mem = Memvid::open_read_only(&path)?;

    let suite = EvalSuite::from_embedded()?;
    let report = run_eval_suite(&mut mem, &suite)?;

    report.print_summary();
    report.print_per_query();

    assert_eq!(report.total_cases, suite.len());
    assert!(
        report.mean_ndcg_at_5 >= 0.0,
        "NDCG@5 must be non-negative, got {}",
        report.mean_ndcg_at_5
    );
    assert!(
        report.cases_with_hits > 0,
        "at least some eval queries should return hits from the corpus"
    );
    assert!(!report.memvid_version.is_empty());
    for r in &report.per_query {
        assert_eq!(r.query_hash.len(), 64, "blake3 hex should be 64 chars");
    }

    Ok(())
}

#[test]
fn test_eval_wal_cases() -> Result<()> {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("eval_wal.mv2");

    {
        let mut mem = Memvid::create(&path)?;
        mem.enable_lex()?;
        insert_eval_corpus(&mut mem)?;
        mem.commit()?;
    }

    let mut mem = Memvid::open_read_only(&path)?;
    let suite = EvalSuite::from_embedded()?;
    let wal_suite = suite.filter_by_keyword("wal");
    assert!(!wal_suite.is_empty(), "fixture must contain at least one WAL case");

    let report = run_eval_suite(&mut mem, &wal_suite)?;
    println!("WAL-specific NDCG@5: {:.4}", report.mean_ndcg_at_5);

    assert_eq!(report.total_cases, wal_suite.len());
    assert!(report.mean_ndcg_at_5 >= 0.0);
    Ok(())
}

/// synthetic docs with uris matching the eval fixtures so bm25 can find them
fn insert_eval_corpus(mem: &mut Memvid) -> Result<()> {
    let docs: &[(&str, &str, &str)] = &[
        (
            "mv2://docs/wal.md",
            "Write-Ahead Log",
            "The write-ahead log (WAL) is a circular buffer embedded in the mv2 file starting at \
             byte 4096. It ensures crash safety by logging all mutations before they are flushed \
             to data segments. The WAL checkpoint triggers at 75% full or every 1000 appends. \
             During crash recovery the WAL is replayed to reconstruct the last committed state.",
        ),
        (
            "mv2://docs/crash-safety.md",
            "Crash Safety",
            "Memvid is designed for crash safety. All writes go through the write-ahead log before \
             being committed to data segments. In the event of a crash the WAL is replayed on next \
             open to ensure no committed data is lost. The file lock prevents concurrent writers.",
        ),
        (
            "mv2://docs/file-format.md",
            "MV2 File Format",
            "The mv2 single-file format consists of: Header (4KB), Embedded WAL (1-64MB), \
             Data Segments, Lex Index (Tantivy), Vec Index (HNSW), Time Index, TOC (table of \
             contents), and Footer (56 bytes). All sections are appended sequentially; frames are \
             immutable once committed.",
        ),
        (
            "mv2://docs/header.md",
            "File Header",
            "The mv2 file header occupies the first 4096 bytes (4KB) of the file. It stores \
             magic bytes, format version, feature flags bitmask, root frame ID, WAL offset, \
             and the Blake3 checksum of the header itself.",
        ),
        (
            "mv2://docs/lex-index.md",
            "Lexical Search Index",
            "The lexical search index uses Tantivy BM25 full-text search. Indexed fields are: \
             uri, title, content, tags, and dates. The index is stored as a Tantivy segment \
             embedded in the mv2 file. Queries support AND, OR, phrase, and fuzzy matching.",
        ),
        (
            "mv2://docs/vec-index.md",
            "Vector Index",
            "The vector similarity search index uses HNSW (Hierarchical Navigable Small World) \
             for approximate nearest neighbour lookup. Below 1000 vectors a flat linear scan is \
             used instead. The index is built using ONNX embeddings and stored in the mv2 file.",
        ),
        (
            "mv2://docs/search.md",
            "Search Architecture",
            "Memvid supports lexical search (BM25 via Tantivy), vector similarity search (HNSW), \
             and hybrid search combining both via Reciprocal Rank Fusion (RRF) with k=60. \
             The sketch SimHash pre-filter generates candidate frames before full BM25 scoring. \
             Search supports cursor pagination for large result sets.",
        ),
        (
            "mv2://docs/sketch.md",
            "Sketch Pre-filter",
            "The sketch track stores a SimHash (locality sensitive hashing) fingerprint per frame. \
             It is used as a sub-millisecond pre-filter to generate candidate frames before the \
             more expensive BM25 or vector scoring. SimHash groups similar documents into the same \
             bucket, enabling fast approximate candidate generation.",
        ),
        (
            "mv2://docs/ask.md",
            "RAG Ask Interface",
            "The ask() API combines retrieval augmented generation (RAG) with hybrid search. \
             It uses Reciprocal Rank Fusion (RRF) to merge lexical and semantic results. \
             The adaptive retrieval mode dynamically adjusts top_k based on relevance score \
             distribution. Analytical questions retrieve up to 5x more candidates for \
             comprehensive context. Results are returned as AskContextFragment objects.",
        ),
        (
            "mv2://docs/frame.md",
            "Frame Design",
            "Frames are immutable once committed. Each frame has a unique frame_id (u64), \
             a URI, optional title and tags, compressed bytes payload, a Blake3 checksum, \
             and a timestamp. The append-only design ensures deterministic checksums across \
             operations.",
        ),
        (
            "mv2://docs/mutation.md",
            "Write Operations",
            "The primary write operations are put_bytes() and put_bytes_with_options(). \
             PutOptions allows setting uri, title, and tags via a builder pattern. \
             All writes are buffered in the WAL and flushed on commit(). \
             The commit() call flushes WAL frames to permanent data segments.",
        ),
        (
            "mv2://docs/clip.md",
            "CLIP Visual Embeddings",
            "The clip feature flag enables CLIP image embeddings for visual search. \
             CLIP encodes both images and text into the same embedding space, enabling \
             cross-modal search. Embeddings are computed via an ONNX model and stored \
             in the vector index.",
        ),
        (
            "mv2://docs/whisper.md",
            "Whisper Audio Transcription",
            "The whisper feature flag enables audio transcription using OpenAI Whisper \
             running locally via the Candle ML framework. Supported formats include MP3, \
             WAV, FLAC, and AAC. Transcribed text is indexed in the lexical index.",
        ),
        (
            "mv2://docs/encryption.md",
            "Encryption Capsules",
            "The encryption feature flag enables AES-256-GCM encryption for mv2e capsule files. \
             Keys are derived via Argon2 from a user password. Encrypted capsules have the .mv2e \
             extension. The encryption module uses zeroize to clear sensitive key material from memory.",
        ),
        (
            "mv2://docs/timeline.md",
            "Timeline and Time Index",
            "The time index stores frames in chronological order, enabling timeline queries. \
             timeline() returns frames sorted by insertion timestamp. Supports filtering by \
             time range. Used by the ask() API for recency-biased queries.",
        ),
        (
            "mv2://docs/replay.md",
            "Time-Travel Replay",
            "The replay feature enables time-travel views of the memory. SearchRequest supports \
             as_of_frame and as_of_ts fields to restrict results to frames committed before \
             a given frame ID or Unix timestamp. This allows agents to query the memory as it \
             existed at any prior point in time.",
        ),
        (
            "mv2://docs/mesh.md",
            "Logic-Mesh Entity Graph",
            "The logic_mesh feature flag enables a NER (Named Entity Recognition) entity \
             relationship graph. Entities are extracted from frame content using DistilBERT-NER \
             and stored in the TOC. The mesh enables structured relationship queries.",
        ),
        (
            "mv2://docs/acl.md",
            "Access Control Lists",
            "The ACL system restricts search results based on caller identity. SearchRequest \
             includes an optional acl_context field with caller identity. ACL enforcement \
             mode can be audit (log violations) or enforce (filter results). Frame-level \
             ACL tags are set at write time via PutOptions.",
        ),
        (
            "mv2://docs/verify.md",
            "File Verification",
            "Memvid::verify() performs file integrity checks. Shallow verify checks the header \
             and TOC checksums. Deep verify reads every frame and validates its Blake3 checksum. \
             Returns a VerificationReport with overall_status and per-section results.",
        ),
        (
            "mv2://docs/doctor.md",
            "Doctor and Recovery",
            "The doctor module performs health checks and recovery operations. It can detect \
             and repair common corruption patterns, rebuild indices, and recover from partial \
             WAL writes. The doctor report includes a detailed breakdown of all issues found.",
        ),
        (
            "mv2://docs/checksums.md",
            "Blake3 Checksums",
            "Memvid uses Blake3 for all checksums: frame payload checksums, header checksums, \
             and index checksums. Blake3 is chosen for its speed (faster than SHA-256) and \
             security. Checksums are deterministic: the same content always produces the same \
             Blake3 hash.",
        ),
        (
            "mv2://docs/compression.md",
            "Frame Compression",
            "Frame bytes are compressed before storage. Supported codecs are zstd (default, \
             best ratio) and lz4 (fast, lower ratio). The codec is stored per-frame in the \
             frame header. Compression is applied transparently during put_bytes() and \
             decompressed transparently during search and retrieval.",
        ),
        (
            "mv2://docs/features.md",
            "Feature Flags",
            "Memvid uses Rust feature flags for conditional compilation. Default features are \
             lex, pdf_extract, and simd. Optional features include vec, clip, whisper, \
             encryption, temporal_track, logic_mesh, and api_embed. Feature flags are \
             specified in Cargo.toml and guard code with #[cfg(feature = \"...\")].",
        ),
        (
            "mv2://docs/lifecycle.md",
            "Memvid Lifecycle",
            "Memvid::create() creates a new mv2 file with an exclusive file lock. \
             Memvid::open() opens an existing file for read-write access. The file lock \
             ensures crash safety by preventing concurrent writers. Dropping the Memvid \
             instance releases the lock and flushes any pending WAL entries.",
        ),
    ];

    for (uri, title, content) in docs {
        let opts = PutOptions::builder()
            .uri(*uri)
            .title(*title)
            .search_text(*content)
            .build();
        mem.put_bytes_with_options(content.as_bytes(), opts)?;
    }

    Ok(())
}
