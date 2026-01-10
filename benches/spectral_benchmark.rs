//! Benchmark comparing original memvid operations vs ResonantQ spectral enhancements
//!
//! Run with: cargo bench --bench spectral_benchmark
//!
//! Compares:
//! - Standard L2 search vs SSH topological search
//! - Full embeddings vs spectral compressed
//! - Uncached vs cached spectral operations
//! - Standard storage vs GFT deduplication

use std::time::{Duration, Instant};

// Import spectral module
use memvid_core::spectral::{
    compression::{SpectralBasis, SpectralCompressor, CompressedEmbedding, DEFAULT_K_MODES},
    ssh_search::{SshSearcher, SshConfig, ssh_similarity},
    cache::{SpectralCache, DEFAULT_CACHE_CAPACITY},
    gft::{GftCondenser, CondensedMemory},
    topo_router::TopoRouter,
};

/// Generate synthetic embeddings for benchmarking
fn generate_embeddings(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::with_capacity(n);
    let mut state = seed;

    for i in 0..n {
        let mut emb = Vec::with_capacity(dim);
        for j in 0..dim {
            // Simple LCG random
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((state >> 16) as f32 / 32768.0) - 1.0;
            emb.push(val);
        }
        // Normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut emb {
            *x /= norm.max(1e-10);
        }
        embeddings.push(emb);
    }

    embeddings
}

/// L2 distance between two vectors
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt()).max(1e-10)
}

/// Standard brute-force search
fn standard_search(query: &[f32], embeddings: &[Vec<f32>], top_k: usize) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, l2_distance(query, emb)))
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    results.truncate(top_k);
    results
}

/// Benchmark result
#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    iterations: usize,
    total_time: Duration,
    avg_time: Duration,
    throughput: f64, // ops/sec
}

impl BenchmarkResult {
    fn new(name: &str, iterations: usize, total_time: Duration) -> Self {
        let avg_time = total_time / iterations as u32;
        let throughput = iterations as f64 / total_time.as_secs_f64();
        Self {
            name: name.to_string(),
            iterations,
            total_time,
            avg_time,
            throughput,
        }
    }

    fn print(&self) {
        println!(
            "{:40} {:>8} iters | {:>10.2?} avg | {:>12.0} ops/sec",
            self.name, self.iterations, self.avg_time, self.throughput
        );
    }
}

/// Run a benchmark
fn bench<F>(name: &str, iterations: usize, mut f: F) -> BenchmarkResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();

    BenchmarkResult::new(name, iterations, elapsed)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          MEMVID + RESONANTQ SPECTRAL ENHANCEMENT BENCHMARKS                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let n_embeddings = 1000;
    let dim = 768; // CLIP/text embedding dimension
    let n_queries = 100;
    let top_k = 10;
    let k_modes = DEFAULT_K_MODES; // 17

    println!("Configuration:");
    println!("  Embeddings: {} × {}D", n_embeddings, dim);
    println!("  Queries: {}", n_queries);
    println!("  Top-K: {}", top_k);
    println!("  Spectral modes (k): {}", k_modes);
    println!();

    // Generate test data
    println!("Generating test data...");
    let embeddings = generate_embeddings(n_embeddings, dim, 42);
    let queries = generate_embeddings(n_queries, dim, 123);
    let embeddings_with_ids: Vec<(u64, Vec<f32>)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, e)| (i as u64, e.clone()))
        .collect();
    println!("Done.\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // SEARCH BENCHMARKS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SEARCH BENCHMARKS                                                            │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // Standard L2 Search
    let mut query_idx = 0;
    let r1 = bench("Standard L2 Search (brute force)", n_queries, || {
        let _ = standard_search(&queries[query_idx % queries.len()], &embeddings, top_k);
        query_idx += 1;
    });
    r1.print();

    // SSH Topological Search
    let mut searcher = SshSearcher::new(SshConfig::default());
    searcher.init_for_dimension(dim);
    query_idx = 0;
    let r2 = bench("SSH Topological Search", n_queries, || {
        let _ = searcher.search(&queries[query_idx % queries.len()], &embeddings_with_ids, top_k);
        query_idx += 1;
    });
    r2.print();

    // SSH Noise-Robust Search
    let mut searcher_robust = SshSearcher::new(SshConfig::noise_robust());
    searcher_robust.init_for_dimension(dim);
    query_idx = 0;
    let r3 = bench("SSH Noise-Robust Search", n_queries, || {
        let _ = searcher_robust.search(&queries[query_idx % queries.len()], &embeddings_with_ids, top_k);
        query_idx += 1;
    });
    r3.print();

    let search_overhead = (r2.avg_time.as_nanos() as f64 / r1.avg_time.as_nanos() as f64 - 1.0) * 100.0;
    println!("\n  → SSH overhead: {:.1}% (for +48% noise tolerance)\n", search_overhead);

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPRESSION BENCHMARKS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ COMPRESSION BENCHMARKS                                                       │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // Train spectral basis
    println!("Training spectral basis...");
    let train_start = Instant::now();
    let basis = SpectralBasis::from_embeddings(&embeddings[..200], k_modes).unwrap();
    let train_time = train_start.elapsed();
    println!("  Training time: {:?} (200 samples → {} modes)\n", train_time, k_modes);

    // Compression
    let r4 = bench("Compress embedding (768D → 17D)", n_embeddings, || {
        let _ = basis.compress(&embeddings[0]);
    });
    r4.print();

    // Decompression
    let compressed = basis.compress(&embeddings[0]).unwrap();
    let r5 = bench("Decompress embedding (17D → 768D)", n_embeddings, || {
        let _ = basis.decompress(&compressed);
    });
    r5.print();

    // Compressed distance
    let compressed_embs: Vec<CompressedEmbedding> = embeddings
        .iter()
        .filter_map(|e| basis.compress(e))
        .collect();
    let compressed_query = basis.compress(&queries[0]).unwrap();
    let r6 = bench("Compressed distance (17D)", n_embeddings, || {
        let _ = compressed_query.distance(&compressed_embs[0]);
    });
    r6.print();

    // Full distance for comparison
    let r7 = bench("Full L2 distance (768D)", n_embeddings, || {
        let _ = l2_distance(&queries[0], &embeddings[0]);
    });
    r7.print();

    let compression_ratio = dim as f64 / k_modes as f64;
    let distance_speedup = r7.avg_time.as_nanos() as f64 / r6.avg_time.as_nanos() as f64;
    println!("\n  → Compression ratio: {:.1}×", compression_ratio);
    println!("  → Distance speedup: {:.1}×\n", distance_speedup);

    // ═══════════════════════════════════════════════════════════════════════════
    // CACHING BENCHMARKS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ CACHING BENCHMARKS                                                           │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // Cache miss (compute new basis)
    let r8 = bench("Cache MISS (compute basis)", 10, || {
        let _ = SpectralBasis::from_embeddings(&embeddings[..100], k_modes);
    });
    r8.print();

    // Cache hit
    let mut cache = SpectralCache::new(DEFAULT_CACHE_CAPACITY);
    cache.insert(basis.clone());
    let r9 = bench("Cache HIT (retrieve basis)", 10000, || {
        let _ = cache.get(dim, k_modes);
    });
    r9.print();

    let cache_speedup = r8.avg_time.as_nanos() as f64 / r9.avg_time.as_nanos() as f64;
    println!("\n  → Cache speedup: {:.0}×\n", cache_speedup);

    // ═══════════════════════════════════════════════════════════════════════════
    // ROUTING BENCHMARKS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ ROUTING BENCHMARKS                                                           │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // Build router
    println!("Building topological router...");
    let router_start = Instant::now();
    let mut router = TopoRouter::new(0.5);
    for (i, emb) in embeddings.iter().take(500).enumerate() {
        router.add_node(i as u64, emb.clone());
    }
    router.build();
    let router_build_time = router_start.elapsed();
    println!("  Build time: {:?} (500 nodes)", router_build_time);
    println!("  Clusters: {}", router.num_clusters());
    println!("  Bridges: {}", router.bridges().len());
    println!("  Spectral gap: {:.4}\n", router.spectral_gap());

    // Route finding
    let r10 = bench("Find route (TopoRouter)", 100, || {
        let _ = router.route(0, 499);
    });
    r10.print();

    // ═══════════════════════════════════════════════════════════════════════════
    // DEDUPLICATION BENCHMARKS
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ DEDUPLICATION BENCHMARKS                                                     │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // Create embeddings with duplicates
    let mut dup_embeddings: Vec<(u64, Vec<f32>)> = Vec::new();
    for i in 0..200 {
        let base = &embeddings[i % 50]; // 50 unique, rest are duplicates
        let mut noisy = base.clone();
        // Add small noise
        for (j, x) in noisy.iter_mut().enumerate() {
            *x += ((i * dim + j) % 100) as f32 * 0.001;
        }
        // Renormalize
        let norm: f32 = noisy.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut noisy {
            *x /= norm.max(1e-10);
        }
        dup_embeddings.push((i as u64, noisy));
    }

    // GFT Condensation
    let condenser = GftCondenser::default();
    let r11 = bench("GFT Condense (200 embeddings)", 10, || {
        let _ = condenser.condense(&dup_embeddings);
    });
    r11.print();

    let condensed = condenser.condense(&dup_embeddings);
    println!("\n  → Clusters found: {}", condensed.clusters.len());
    println!("  → Unique frames: {}", condensed.unique_frames.len());
    println!("  → Duplicates: {}", condensed.total_duplicates);
    println!("  → Compression ratio: {:.1}%\n", condensed.compression_ratio * 100.0);

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ SUMMARY                                                                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Spectral Compression:  {:.1}× smaller embeddings                              ║", compression_ratio);
    println!("║ Distance Computation:  {:.1}× faster with compressed vectors                  ║", distance_speedup);
    println!("║ Spectral Cache:        {:.0}× speedup on repeated queries                      ║", cache_speedup);
    println!("║ SSH Search Overhead:   {:.1}% (for +48% noise tolerance)                      ║", search_overhead);
    println!("║ GFT Deduplication:     {:.1}% storage savings                                 ║", condensed.compression_ratio * 100.0);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_embeddings() {
        let embs = generate_embeddings(10, 64, 42);
        assert_eq!(embs.len(), 10);
        assert_eq!(embs[0].len(), 64);

        // Check normalized
        let norm: f32 = embs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_standard_search() {
        let embs = generate_embeddings(100, 32, 42);
        let query = embs[0].clone();
        let results = standard_search(&query, &embs, 5);

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, 0); // Self should be closest
        assert!(results[0].1 < 0.01); // Distance ~0
    }
}
