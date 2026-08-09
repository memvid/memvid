//! `shrink_wal_to`: reclaim the WAL region a bulk build reserved or grew
//! into. The region otherwise persists at its high-water mark forever —
//! measured 14–45% dead space in production shards.

use memvid_core::{Memvid, PutManyOpts, PutOptions, SearchRequest};

fn batch_opts(pre_size: u64) -> PutManyOpts {
    PutManyOpts {
        compression_level: 3,
        disable_auto_checkpoint: true,
        skip_sync: false,
        enable_embedding: false,
        auto_tag: false,
        extract_dates: false,
        no_raw: true,
        enable_enrichment: false,
        wal_pre_size_bytes: pre_size,
    }
}

fn doc_text(i: usize) -> String {
    let mut s = format!("zebra{i} quarterly report section ");
    for w in 0..600 {
        s.push_str(&format!("word{}x{} ", i, w));
    }
    s
}

fn put_doc(mem: &mut Memvid, i: usize) {
    let opts = PutOptions::builder()
        .uri(format!("mv2://shrink/{i}"))
        .title(format!("doc {i}"))
        .timestamp(1_700_000_000 + i as i64)
        .extract_triplets(false)
        .instant_index(false)
        .build();
    mem.put_bytes_with_options(doc_text(i).as_bytes(), opts)
        .expect("put");
}

#[test]
fn shrink_reclaims_space_and_preserves_content() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("shrink.mv2");

    let mut mem = Memvid::create(&path).expect("create");
    mem.enable_lex().expect("lex");
    // Pre-size generously (16 MiB, within the default capacity tier) —
    // the exact situation a bulk build leaves behind.
    mem.begin_batch(batch_opts(16 * 1024 * 1024))
        .expect("batch");
    for i in 0..200 {
        put_doc(&mut mem, i);
    }
    mem.end_batch().expect("end_batch");
    mem.commit().expect("commit");

    let before = std::fs::metadata(&path).expect("meta").len();
    mem.shrink_wal_to(0).expect("shrink");
    let after = std::fs::metadata(&path).expect("meta").len();
    assert!(
        before - after > 8 * 1024 * 1024,
        "shrink must reclaim most of the 16MiB region (before {before}, after {after})"
    );
    drop(mem);

    let report = Memvid::verify(&path, true).expect("verify runs");
    assert!(
        !matches!(
            report.overall_status,
            memvid_core::VerificationStatus::Failed
        ),
        "deep verify must pass after shrink: {report:?}"
    );

    let mut mem = Memvid::open(&path).expect("reopen");
    let hits = mem
        .search(SearchRequest {
            query: "zebra150".to_owned(),
            top_k: 5,
            snippet_chars: 80,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: true,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .expect("search");
    assert!(
        !hits.hits.is_empty(),
        "post-shrink search must find doc 150"
    );
    for i in [0usize, 99, 199] {
        // Chunking means frame ids != doc order; resolve through the uri.
        let frame = mem
            .frame_by_uri(&format!("mv2://shrink/{i}"))
            .expect("frame by uri");
        let text = mem.frame_text_by_id(frame.id).expect("frame text readable");
        assert!(
            text.contains(&format!("zebra{i} ")),
            "doc {i} text intact after shrink"
        );
    }

    // Post-shrink appends must work (region regrows if needed).
    mem.begin_batch(batch_opts(0)).expect("batch2");
    for i in 200..220 {
        put_doc(&mut mem, i);
    }
    mem.end_batch().expect("end_batch2");
    mem.commit().expect("commit2");
    let frame = mem
        .frame_by_uri("mv2://shrink/219")
        .expect("appended frame by uri");
    let text = mem.frame_text_by_id(frame.id).expect("new frame");
    assert!(text.contains("zebra219 "), "post-shrink append readable");
}

#[test]
fn shrink_refuses_pending_wal_records() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("pending.mv2");
    let mut mem = Memvid::create(&path).expect("create");
    // Batch mode defers checkpointing — the record stays pending in the
    // WAL until commit (a bare put would auto-checkpoint itself away).
    mem.begin_batch(batch_opts(0)).expect("batch");
    put_doc(&mut mem, 0);
    mem.end_batch().expect("end_batch");
    // No commit — a pending record must block the shrink.
    let err = mem.shrink_wal_to(0);
    assert!(err.is_err(), "shrink with pending WAL must refuse");
}

#[test]
fn shrink_is_noop_at_or_below_floor() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("noop.mv2");
    let mut mem = Memvid::create(&path).expect("create");
    put_doc(&mut mem, 0);
    mem.commit().expect("commit");
    let before = std::fs::metadata(&path).expect("meta").len();
    mem.shrink_wal_to(u64::MAX / 4).expect("noop shrink");
    let after = std::fs::metadata(&path).expect("meta").len();
    assert_eq!(before, after, "target above current size must be a no-op");
}

/// The retained head of a shrunk region still holds the old (already
/// checkpointed) record bytes. With more than one region-floor's worth
/// of records, one record straddles the new boundary and the post-shrink
/// reopen scan misparsed it as "wal record length invalid". Shrink must
/// leave the region reading as empty.
#[test]
fn shrink_survives_records_larger_than_target_region() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("straddle.mv2");

    let mut mem = Memvid::create(&path).expect("create");
    mem.enable_lex().expect("lex");
    mem.begin_batch(batch_opts(16 * 1024 * 1024)).expect("batch");
    // ~6.6MB of doc text (chunk entries roughly double it in the WAL) —
    // comfortably past the 4MiB floor the region shrinks back to.
    for i in 0..600 {
        put_doc(&mut mem, i);
    }
    mem.end_batch().expect("end_batch");
    mem.commit().expect("commit");

    mem.shrink_wal_to(0)
        .expect("shrink with >4MiB of stale records must succeed");
    drop(mem);

    let mut mem = Memvid::open(&path).expect("reopen after shrink");
    let frame = mem
        .frame_by_uri("mv2://shrink/599")
        .expect("frame by uri");
    let text = mem.frame_text_by_id(frame.id).expect("frame text");
    assert!(text.contains("zebra599 "), "doc intact after shrink");
}
