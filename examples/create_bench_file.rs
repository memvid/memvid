//! Creates a test .mv2 file for benchmarking

use memvid_core::{Memvid, PutOptions};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let frame_count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("/tmp/memvid_bench/test.mv2");

    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap())?;

    // Remove old file if exists
    let _ = std::fs::remove_file(path);

    let mut mem = Memvid::create(path)?;
    mem.enable_lex()?;

    let base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Machine learning neural networks deep learning artificial intelligence. Database systems query optimization indexing performance tuning. ";

    for i in 0..frame_count {
        let content = base.repeat(10) + &format!(" Frame {} unique identifier content here.", i);
        let opts = PutOptions::builder()
            .uri(format!("mv2://documents/doc_{i}.txt"))
            .title(format!("Document {i}"))
            .search_text(&content)
            .build();
        mem.put_bytes_with_options(content.as_bytes(), opts)?;
    }

    mem.commit()?;

    let size = std::fs::metadata(path)?.len();
    eprintln!(
        "Created {} with {} frames, {} bytes ({:.1} KB)",
        path,
        frame_count,
        size,
        size as f64 / 1024.0
    );

    Ok(())
}
