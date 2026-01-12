#[cfg(test)]
mod tests {
    use crate::{Memvid, PutOptions, SearchRequest, run_serial_test};
    use tempfile::NamedTempFile;

    #[test]
    #[cfg(not(target_os = "windows"))] // Windows file locking prevents tempfile cleanup
    fn test_lex_persists_and_search_works() {
        run_serial_test(|| {
            let temp = NamedTempFile::new().unwrap();
            let path = temp.path();

            // Phase 1: create, enable lex, ingest docs with periodic seals
            {
                let mut mem = Memvid::create(path).unwrap();
                mem.enable_lex().unwrap();

                for i in 0..1000 {
                    let content = format!(
                        "Document {i} with searchable content about technology and artificial intelligence systems"
                    );
                    let opts = PutOptions::builder()
                        .uri(format!("mv2://doc/{i}"))
                        .search_text(content.clone())
                        .build();
                    mem.put_bytes_with_options(content.as_bytes(), opts)
                        .unwrap();
                    if (i + 1) % 100 == 0 {
                        mem.commit().unwrap();
                    }
                }
                mem.commit().unwrap();

                // Index is present in TOC
                assert!(
                    mem.toc.segment_catalog.lex_enabled,
                    "lex_enabled should be set in catalog"
                );
                assert!(
                    !mem.toc.segment_catalog.tantivy_segments.is_empty(),
                    "tantivy_segments should not be empty"
                );
            }

            // Phase 2: reopen RO and search
            {
                let mut mem = Memvid::open_read_only(path).unwrap();
                assert!(mem.lex_enabled, "lex_enabled should persist after reopen");
                assert!(
                    mem.toc.segment_catalog.lex_enabled,
                    "catalog.lex_enabled should persist after reopen"
                );
                assert!(
                    !mem.toc.segment_catalog.tantivy_segments.is_empty(),
                    "tantivy_segments should persist after reopen"
                );

                let resp = mem
                    .search(SearchRequest {
                        query: "artificial intelligence".into(),
                        top_k: 10,
                        snippet_chars: 200,
                        uri: None,
                        frames: Vec::new(),
                        scope: None,
                        cursor: None,
                        #[cfg(feature = "temporal_track")]
                        temporal: None,
                        as_of_frame: None,
                        as_of_ts: None,
                        no_sketch: false,
                    })
                    .expect("search must succeed");

                assert!(
                    !resp.hits.is_empty(),
                    "expected some hits for 'artificial intelligence'"
                );
                let first_hit = &resp.hits[0];
                let text_lower = first_hit.text.to_lowercase();
                assert!(
                    text_lower.contains("artificial") || text_lower.contains("intelligence"),
                    "first hit should contain search terms, got: {}",
                    first_hit.text
                );
            }
        });
    }
}
