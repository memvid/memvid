//! Batch-mode commit replay must be able to read back frames it just
//! wrote. A chunk entry whose text normalizes to nothing (the naive
//! chunker emits a single-space chunk when a boundary lands on a space
//! followed by a long whitespace-free run) carries no search text, so
//! replay reads its payload back from the file for lex indexing.
//! `data_end` used to advance only after the full replay loop, so that
//! read tripped `validate_frame_bounds` with "payload extends past data
//! region".

use memvid_core::{Memvid, PutManyOpts, PutOptions};

fn batch_opts() -> PutManyOpts {
    PutManyOpts {
        compression_level: 3,
        disable_auto_checkpoint: true,
        skip_sync: false,
        enable_embedding: false,
        auto_tag: false,
        extract_dates: false,
        no_raw: true,
        enable_enrichment: false,
        wal_pre_size_bytes: 0,
    }
}

/// Chunk 1 ends after "Z." (sentence terminal just past the start), so
/// chunk 2 begins on the space. The 4,000-char run that follows has no
/// newline, sentence terminal, or whitespace, so the boundary search
/// falls back to the backward whitespace scan and finds only the space
/// at the chunk's own start, emitting a single-space chunk. Documents
/// embedding large machine-generated blobs (escaped JSON, base64)
/// produce this shape.
fn degenerate_chunk_text() -> String {
    let mut text = String::from("Z. ");
    text.push_str(&"q".repeat(4_000));
    text
}

#[test]
fn batch_commit_replays_chunk_without_search_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("readback.mv2");

    let mut mem = Memvid::create(&path).expect("create");
    mem.enable_lex().expect("lex");
    mem.begin_batch(batch_opts()).expect("batch");
    let opts = PutOptions::builder()
        .uri("mv2://test/degenerate")
        .title("degenerate chunk doc")
        .timestamp(1_700_000_000)
        .extract_triplets(false)
        .instant_index(false)
        .build();
    mem.put_bytes_with_options(degenerate_chunk_text().as_bytes(), opts)
        .expect("put");
    mem.end_batch().expect("end_batch");
    mem.commit()
        .expect("commit must replay a chunk whose text normalizes to nothing");
}
