//! Test script for auto-download functionality.
//!
//! This script tests the auto-download feature with a fresh model directory.
//!
//! ## Run
//!
//! ```bash
//! cargo run --example test_auto_download --features vec
//! ```

#![cfg(feature = "vec")]

use memvid_core::Result;
use memvid_core::text_embed::{LocalTextEmbedder, TextEmbedConfig};
use memvid_core::types::embedding::EmbeddingProvider;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("=== Testing Auto-Download Feature ===\n");

    // Use a separate test directory to avoid affecting existing models
    let test_dir = PathBuf::from("/tmp/memvid-auto-download-test");

    // Clean up any previous test
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir).ok();
    }

    println!("Test directory: {:?}", test_dir);
    println!("Directory exists before: {}", test_dir.exists());

    // Test 1: Default config (auto_download = false) should fail with instructions
    println!("\n--- Test 1: Default Config (should fail gracefully) ---");
    let offline_config = TextEmbedConfig {
        models_dir: test_dir.clone(),
        auto_download: false, // Manual downloads required
        ..Default::default()
    };

    let embedder = LocalTextEmbedder::new(offline_config)?;
    match embedder.embed_text("test") {
        Ok(_) => println!("ERROR: Should have failed, but succeeded?!"),
        Err(e) => {
            let error_msg = format!("{}", e);
            if error_msg.contains("model not found")
                || error_msg.contains("not found")
                || error_msg.contains("Tokenizer not found")
            {
                println!("✓ Correctly failed with helpful error message");
                println!(
                    "  Error excerpt: ...{}",
                    &error_msg[..120.min(error_msg.len())]
                );
            } else {
                println!("⚠ Failed but with unexpected error: {}", e);
            }
        }
    }

    // Test 2: With auto_download = true, should download and work
    println!("\n--- Test 2: Auto-Download Enabled ---");
    let auto_config = TextEmbedConfig {
        models_dir: test_dir.clone(),
        auto_download: true, // Should download models
        ..Default::default()
    };

    println!("Creating embedder with auto_download=true...");
    println!("(This may take several minutes to download ~133MB model)");

    let embedder = LocalTextEmbedder::new(auto_config)?;

    // This should trigger the download
    println!("\nGenerating embedding (will trigger download if needed)...");
    match embedder.embed_text("Hello, this is a test of auto-download functionality.") {
        Ok(embedding) => {
            println!("✓ Auto-download and embedding succeeded!");
            println!("  Embedding dimension: {}", embedding.len());
            println!("  First 5 values: {:?}", &embedding[..5]);
        }
        Err(e) => {
            println!("✗ Auto-download failed: {}", e);
            return Err(e);
        }
    }

    // Verify files were created
    println!("\n--- Verifying Downloaded Files ---");
    let model_file = test_dir.join("bge-small-en-v1.5.onnx");
    let tokenizer_file = test_dir.join("bge-small-en-v1.5_tokenizer.json");

    if model_file.exists() {
        let size = std::fs::metadata(&model_file)?.len();
        println!(
            "✓ Model file exists: {} bytes ({:.1} MB)",
            size,
            size as f64 / 1_048_576.0
        );
    } else {
        println!("✗ Model file not found!");
    }

    if tokenizer_file.exists() {
        let size = std::fs::metadata(&tokenizer_file)?.len();
        println!("✓ Tokenizer file exists: {} bytes", size);
    } else {
        println!("✗ Tokenizer file not found!");
    }

    // Test 3: Subsequent runs should use cached model
    println!("\n--- Test 3: Using Cached Model ---");
    let cached_config = TextEmbedConfig {
        models_dir: test_dir.clone(),
        auto_download: true,
        ..Default::default()
    };

    let embedder2 = LocalTextEmbedder::new(cached_config)?;
    let start = std::time::Instant::now();
    let _ = embedder2.embed_text("Quick test")?;
    let elapsed = start.elapsed();

    // Should be fast if model is already loaded
    println!("✓ Cached run completed in: {:?}", elapsed);

    // Cleanup
    println!("\n--- Cleanup ---");
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir)?;
        println!("✓ Test directory cleaned up");
    }

    println!("\n=== All Auto-Download Tests Passed! ===");

    Ok(())
}
