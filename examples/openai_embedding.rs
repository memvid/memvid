//! Example demonstrating OpenAI API embedding usage.
//!
//! This example shows how to:
//! - Create an OpenAI embedder with default configuration
//! - Generate embeddings using the API
//! - Compute cosine similarity between embeddings
//! - Use different models (small, large, ada)
//!
//! ## Prerequisites
//!
//! Set your OpenAI API key:
//! ```bash
//! export OPENAI_API_KEY="sk-..."
//! ```
//!
//! ## Run
//!
//! ```bash
//! cargo run --example openai_embedding --features api_embed
//! ```

use memvid_core::Result;

#[cfg(feature = "api_embed")]
use memvid_core::api_embed::{OpenAIConfig, OpenAIEmbedder};
#[cfg(feature = "api_embed")]
use memvid_core::types::embedding::EmbeddingProvider;

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(feature = "api_embed")]
fn main() -> Result<()> {
    println!("=== OpenAI Embedding Example ===\n");

    // Check if API key is set
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("Error: OPENAI_API_KEY environment variable not set.");
        eprintln!("Please set it with: export OPENAI_API_KEY=\"sk-...\"");
        std::process::exit(1);
    }

    // Create embedder with default config (text-embedding-3-small, 1536 dimensions)
    println!("Creating OpenAI embedder (text-embedding-3-small)...");
    let config = OpenAIConfig::default();
    let embedder = OpenAIEmbedder::new(config)?;

    println!("Model: {}", embedder.model());
    println!("Kind: {}", embedder.kind());
    println!("Dimensions: {}", embedder.dimension());
    println!("Ready: {}\n", embedder.is_ready());

    // Example 1: Single text embedding
    println!("--- Example 1: Single Text Embedding ---");
    let text = "The quick brown fox jumps over the lazy dog";
    println!("Embedding text: \"{}\"", text);

    let embedding = embedder.embed_text(text)?;
    println!("Generated embedding of dimension {}\n", embedding.len());

    // Example 2: Semantic similarity
    println!("--- Example 2: Semantic Similarity ---");
    let text1 = "Machine learning and artificial intelligence";
    let text2 = "Deep neural networks for AI applications";
    let text3 = "The history of ancient Rome";

    let emb1 = embedder.embed_text(text1)?;
    let emb2 = embedder.embed_text(text2)?;
    let emb3 = embedder.embed_text(text3)?;

    println!("Text 1: \"{}\"", text1);
    println!("Text 2: \"{}\"", text2);
    println!("Text 3: \"{}\"", text3);
    println!();

    let sim_1_2 = cosine_similarity(&emb1, &emb2);
    let sim_1_3 = cosine_similarity(&emb1, &emb3);
    let sim_2_3 = cosine_similarity(&emb2, &emb3);

    println!("Similarity (1 ↔ 2): {:.4}", sim_1_2);
    println!("Similarity (1 ↔ 3): {:.4}", sim_1_3);
    println!("Similarity (2 ↔ 3): {:.4}", sim_2_3);

    if sim_1_2 > sim_1_3 {
        println!("\n✓ Related texts (1 & 2) have higher similarity than unrelated (1 & 3)!");
    }
    println!();

    // Example 3: Batch processing
    println!("--- Example 3: Batch Processing ---");
    let documents = vec![
        "Python programming language",
        "JavaScript web development",
        "Rust systems programming",
        "Italian cooking recipes",
    ];

    println!("Processing {} documents in batch...", documents.len());
    let batch_embeddings = embedder.embed_batch(&documents)?;
    println!(
        "✓ Generated {} embeddings of dimension {}\n",
        batch_embeddings.len(),
        batch_embeddings.first().map(|e| e.len()).unwrap_or(0)
    );

    // Find most similar pair
    let mut max_sim = 0.0;
    let mut max_pair = (0, 0);

    for i in 0..batch_embeddings.len() {
        for j in (i + 1)..batch_embeddings.len() {
            let sim = cosine_similarity(&batch_embeddings[i], &batch_embeddings[j]);
            if sim > max_sim {
                max_sim = sim;
                max_pair = (i, j);
            }
        }
    }

    println!("Most similar pair (similarity: {:.4}):", max_sim);
    println!("  [{}] \"{}\"", max_pair.0, documents[max_pair.0]);
    println!("  [{}] \"{}\"\n", max_pair.1, documents[max_pair.1]);

    // Example 4: Available models
    println!("--- Example 4: Available Models ---");
    println!("OpenAI embedding models:");
    println!("  - text-embedding-3-small (1536d) - Default, fastest, cheapest");
    println!("  - text-embedding-3-large (3072d) - Highest quality");
    println!("  - text-embedding-ada-002 (1536d) - Legacy model");
    println!();
    println!("To use a different model:");
    println!("  let config = OpenAIConfig::large();");
    println!("  let embedder = OpenAIEmbedder::new(config)?;");
    println!();

    println!("=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("✓ API embeddings require OPENAI_API_KEY environment variable");
    println!("✓ text-embedding-3-small is fast and cost-effective");
    println!("✓ Batch processing reduces API calls for multiple texts");
    println!("✓ Similar texts have higher cosine similarity scores");

    Ok(())
}

#[cfg(not(feature = "api_embed"))]
fn main() {
    eprintln!("This example requires the 'api_embed' feature.");
    eprintln!("Run with: cargo run --example openai_embedding --features api_embed");
    std::process::exit(1);
}
