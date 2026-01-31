//! Example: HTTP Streaming Access to .mv2 Files
//!
//! This example demonstrates how to read .mv2 files hosted on CDNs using HTTP
//! range requests. Only the header, TOC, and needed data segments are fetched,
//! minimizing network transfer.
//!
//! # Usage
//!
//! ```bash
//! # With a local file (for testing)
//! cargo run --example http_streaming --features streaming -- local path/to/file.mv2
//!
//! # With an HTTP URL
//! cargo run --example http_streaming --features streaming -- http https://cdn.example.com/knowledge.mv2
//! ```
//!
//! # Features
//!
//! - **CDN deployment**: Store .mv2 files on S3, CloudFront, Cloudflare R2, etc.
//! - **20x+ network reduction**: Only fetch header, TOC, and needed data segments
//! - **Mobile/serverless enablement**: Low memory footprint, no full file download
//! - **Zero local storage**: Read directly from HTTP without caching to disk

use memvid_core::streaming::{HttpStreamingSource, LocalStreamingSource, StreamingMemvid};
use memvid_core::{SearchRequest, TimelineQuery};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: http_streaming <local|http> <path_or_url>");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  http_streaming local ./data/knowledge.mv2");
        eprintln!("  http_streaming http https://cdn.example.com/knowledge.mv2");
        std::process::exit(1);
    }

    let mode = &args[1];
    let source_path = &args[2];

    match mode.as_str() {
        "local" => run_local_example(source_path),
        "http" => run_http_example(source_path),
        _ => {
            eprintln!("Unknown mode: {mode}. Use 'local' or 'http'.");
            std::process::exit(1);
        }
    }
}

fn run_local_example(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Opening local file via streaming interface: {path}");
    println!();

    let source = LocalStreamingSource::open(path)?;
    let streaming = StreamingMemvid::open(source)?;

    print_stats(&streaming);
    print_timeline(&streaming)?;
    search_example(&streaming)?;

    Ok(())
}

fn run_http_example(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Opening HTTP source: {url}");
    println!();

    let source = HttpStreamingSource::from_url(url)?;
    let streaming = StreamingMemvid::open(source)?;

    print_stats(&streaming);
    print_timeline(&streaming)?;
    search_example(&streaming)?;

    Ok(())
}

fn print_stats<S: memvid_core::streaming::StreamingSource>(streaming: &StreamingMemvid<S>) {
    let stats = streaming.stats();

    println!("=== File Statistics ===");
    println!("  Source: {}", streaming.source_id());
    println!("  File size: {} bytes", streaming.file_size());
    println!("  Frame count: {}", stats.frame_count);
    println!("  Active frames: {}", stats.active_frame_count);
    println!("  Has lex index: {}", stats.has_lex_index);
    println!("  Has vec index: {}", stats.has_vec_index);
    println!("  WAL size: {} bytes", stats.wal_bytes);
    println!("  Lex index size: {} bytes", stats.lex_index_bytes);
    println!();
}

fn print_timeline<S: memvid_core::streaming::StreamingSource>(
    streaming: &StreamingMemvid<S>,
) -> Result<(), Box<dyn std::error::Error>> {
    let timeline = streaming.timeline(
        TimelineQuery::builder()
            .limit(std::num::NonZeroU64::new(5).unwrap())
            .build(),
    )?;

    println!("=== Timeline (first 5 entries) ===");
    for entry in timeline {
        let preview: String = entry.preview.chars().take(60).collect();
        println!(
            "  [{:3}] {} - {}{}",
            entry.frame_id,
            entry.uri.as_deref().unwrap_or("<no uri>"),
            preview,
            if entry.preview.len() > 60 { "..." } else { "" }
        );
    }
    println!();

    Ok(())
}

fn search_example<S: memvid_core::streaming::StreamingSource>(
    streaming: &StreamingMemvid<S>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Search Example ===");

    // Prompt for search query
    println!("Enter a search query (or press Enter for 'test'):");
    let mut query = String::new();
    std::io::stdin().read_line(&mut query)?;
    let query = query.trim();
    let query = if query.is_empty() { "test" } else { query };

    println!("Searching for: \"{query}\"");

    let response = streaming.search(SearchRequest {
        query: query.to_string(),
        top_k: 5,
        snippet_chars: 100,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal_track")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
    })?;

    println!(
        "  Found {} total hits (showing top {})",
        response.total_hits,
        response.hits.len()
    );
    println!("  Search took {}ms", response.elapsed_ms);
    println!();

    for hit in &response.hits {
        println!("  [Rank {}] Frame {} - {}", hit.rank, hit.frame_id, hit.uri);
        if let Some(title) = &hit.title {
            println!("    Title: {title}");
        }
        let snippet: String = hit.text.chars().take(80).collect();
        println!(
            "    Snippet: {}{}",
            snippet,
            if hit.text.len() > 80 { "..." } else { "" }
        );
        if let Some(score) = hit.score {
            println!("    Score: {score:.3}");
        }
        println!();
    }

    Ok(())
}
