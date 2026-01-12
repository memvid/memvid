//! Basic usage example demonstrating create, put, find, and timeline operations.
//!
//! Run with: cargo run --example basic_usage

use std::path::PathBuf;
use tempfile::tempdir;

use memvid_core::{Memvid, PutOptions, Result, SearchRequest, TimelineQuery};

fn main() -> Result<()> {
    // Create a temporary directory for our example
    let dir = tempdir().expect("failed to create temp dir");
    let path: PathBuf = dir.path().join("example.mv2");

    println!("=== Memvid Core Basic Usage Example ===\n");

    // ========================================
    // 1. CREATE a new memory file
    // ========================================
    println!("1. Creating memory file at {:?}", path);
    let mut mem = Memvid::create(&path)?;
    println!("   Memory created successfully!\n");

    // ========================================
    // 2. PUT documents into the memory
    // ========================================
    println!("2. Adding documents to memory...");

    // Simple put with just bytes
    let seq1 = mem.put_bytes(b"Hello, Memvid! This is a simple text document.")?;
    println!("   Added document 1, sequence: {}", seq1);

    // Put with options (title, URI, tags)
    let options = PutOptions::builder()
        .title("Getting Started Guide")
        .uri("mv2://docs/getting-started.md")
        .tag("category", "documentation")
        .tag("version", "2.0")
        .build();
    let seq2 = mem.put_bytes_with_options(
        b"This guide covers the basics of using Memvid for AI memory storage.",
        options,
    )?;
    println!("   Added document 2 (with metadata), sequence: {}", seq2);

    // Add more documents
    let options = PutOptions::builder()
        .title("API Reference")
        .uri("mv2://docs/api-reference.md")
        .tag("category", "documentation")
        .build();
    mem.put_bytes_with_options(
        b"The Memvid API provides methods for create, put, find, and timeline operations.",
        options,
    )?;

    let options = PutOptions::builder()
        .title("FAQ")
        .uri("mv2://docs/faq.md")
        .tag("category", "support")
        .build();
    mem.put_bytes_with_options(
        b"Frequently asked questions about Memvid memory files and search.",
        options,
    )?;

    // Commit changes to persist them
    mem.commit()?;
    println!("   Committed all changes\n");

    // ========================================
    // 3. STATS - Check memory statistics
    // ========================================
    println!("3. Memory statistics:");
    let stats = mem.stats()?;
    println!("   Frame count: {}", stats.frame_count);
    println!("   Has lexical index: {}", stats.has_lex_index);
    println!("   Has vector index: {}", stats.has_vec_index);
    println!("   Has time index: {}", stats.has_time_index);
    println!();

    // ========================================
    // 4. FIND - Search for documents
    // ========================================
    println!("4. Searching for documents...");

    // Search for "memvid"
    let request = SearchRequest {
        query: "memvid".to_string(),
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
    };
    let response = mem.search(request)?;
    println!("   Query: 'memvid'");
    println!("   Total hits: {}", response.total_hits);
    println!("   Elapsed: {}ms", response.elapsed_ms);
    for hit in &response.hits {
        let title = hit.title.as_deref().unwrap_or("Untitled");
        let score = hit.score.unwrap_or(0.0);
        println!("   - [{}] {} (score: {:.3})", hit.frame_id, title, score);
        println!(
            "     Snippet: {}...",
            &hit.text.chars().take(60).collect::<String>()
        );
    }
    println!();

    // Search within a scope
    let request = SearchRequest {
        query: "documentation".to_string(),
        top_k: 10,
        snippet_chars: 100,
        uri: None,
        frames: Vec::new(),
        scope: Some("mv2://docs/".to_string()),
        cursor: None,
        #[cfg(feature = "temporal_track")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
    };
    let response = mem.search(request)?;
    println!("   Query: 'documentation' (scope: mv2://docs/)");
    println!("   Total hits: {}", response.total_hits);
    println!();

    // ========================================
    // 5. TIMELINE - Browse documents chronologically
    // ========================================
    println!("5. Timeline (chronological view):");
    let timeline = mem.timeline(TimelineQuery::default())?;
    for entry in &timeline {
        let uri = entry.uri.as_deref().unwrap_or("(no uri)");
        println!(
            "   [{}] {} - {}",
            entry.frame_id,
            uri,
            entry.preview.chars().take(40).collect::<String>()
        );
    }
    println!();

    // ========================================
    // 6. REOPEN - Close and reopen the memory
    // ========================================
    println!("6. Closing and reopening memory...");
    drop(mem);

    let reopened = Memvid::open(&path)?;
    let stats = reopened.stats()?;
    println!("   Reopened successfully!");
    println!("   Frame count after reopen: {}", stats.frame_count);
    println!();

    // ========================================
    // 7. VERIFY - Check file integrity
    // ========================================
    println!("7. Verifying file integrity...");
    drop(reopened); // Close the memory before verifying
    let report = Memvid::verify(&path, false)?;
    println!("   Verification status: {:?}", report.overall_status);
    println!();

    println!("=== Example completed successfully! ===");

    Ok(())
}
