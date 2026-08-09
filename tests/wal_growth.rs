//! Mid-batch WAL growth replaces the `EmbeddedWal` instance, which
//! used to reset `skip_sync` to false — every append after the first
//! doubling paid a full fsync, silently defeating the batch's deferred
//! fsync mode. Growth must re-apply the batch options.

use memvid_core::{Memvid, PutManyOpts, PutOptions};

fn batch_opts() -> PutManyOpts {
    PutManyOpts {
        compression_level: 3,
        disable_auto_checkpoint: true,
        skip_sync: true,
        enable_embedding: false,
        auto_tag: false,
        extract_dates: false,
        no_raw: true,
        enable_enrichment: false,
        wal_pre_size_bytes: 0,
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
        .uri(format!("mv2://growth/{i}"))
        .title(format!("doc {i}"))
        .timestamp(1_700_000_000 + i as i64)
        .extract_triplets(false)
        .instant_index(false)
        .build();
    mem.put_bytes_with_options(doc_text(i).as_bytes(), opts)
        .expect("put");
}

#[test]
fn growth_preserves_batch_skip_sync() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("growth.mv2");

    let mut mem = Memvid::create(&path).expect("create");
    mem.enable_lex().expect("lex");
    mem.begin_batch(batch_opts()).expect("batch");
    assert!(mem.wal_stats().skip_sync, "batch mode sets skip_sync");
    let start_region = mem.wal_stats().region_size;

    // ~6.6MB of docs against the default region forces mid-batch
    // doubling — the growth path under test.
    for i in 0..600 {
        put_doc(&mut mem, i);
    }
    let stats = mem.wal_stats();
    assert!(
        stats.region_size > start_region,
        "test requires mid-batch growth (region {} -> {})",
        start_region,
        stats.region_size
    );
    assert!(
        stats.skip_sync,
        "mid-batch growth must preserve the batch's skip_sync mode"
    );

    mem.end_batch().expect("end_batch");
    mem.commit().expect("commit");
}
