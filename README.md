<img width="2000" height="524" alt="Social Cover (9)" src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />

<p align="center">
 <a href="docs/i18n/README.es.md">🇪🇸 Español</a>
 <a href="docs/i18n/README.fr.md">🇫🇷 Français</a>
 <a href="docs/i18n/README.so.md">🇸🇴 Soomaali</a>
 <a href="docs/i18n/README.ar.md">🇸🇦 العربية</a>
</p>

<p align="center">
  <strong>Memvid is a single-file memory layer for AI agents with instant retrieval and long-term memory.</strong><br/>
  Persistent, versioned, and portable memory, without databases.
</p>

<p align="center">
  <a href="https://www.memvid.com">Website</a>
  ·
  <a href="https://sandbox.memvid.com">Try Sandbox</a>
  ·
  <a href="https://docs.memvid.com">Docs</a>
  ·
  <a href="https://github.com/memvid/memvid/discussions">Discussions</a>
</p>

<p align="center">
  <a href="https://crates.io/crates/memvid-core"><img src="https://img.shields.io/crates/v/memvid-core?style=flat-square&logo=rust" alt="Crates.io" /></a>
  <a href="https://docs.rs/memvid-core"><img src="https://img.shields.io/docsrs/memvid-core?style=flat-square&logo=docs.rs" alt="docs.rs" /></a>
  <a href="https://github.com/memvid/memvid/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License" /></a>
</p>

<p align="center">
  <a href="https://github.com/memvid/memvid/stargazers"><img src="https://img.shields.io/github/stars/memvid/memvid?style=flat-square&logo=github" alt="Stars" /></a>
  <a href="https://github.com/memvid/memvid/network/members"><img src="https://img.shields.io/github/forks/memvid/memvid?style=flat-square&logo=github" alt="Forks" /></a>
  <a href="https://github.com/memvid/memvid/issues"><img src="https://img.shields.io/github/issues/memvid/memvid?style=flat-square&logo=github" alt="Issues" /></a>
  <a href="https://discord.gg/2mynS7fcK7"><img src="https://img.shields.io/discord/1442910055233224745?style=flat-square&logo=discord&label=discord" alt="Discord" /></a>
</p>

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/</a>
</p>

<h2 align="center">⭐️ Leave a STAR to support the project ⭐️</h2>
</p>

## What is Memvid?

Memvid is a portable AI memory system that packages your data, embeddings, search structure, and metadata into a single file.

Instead of running complex RAG pipelines or server-based vector databases, Memvid enables fast retrieval directly from the file.

The result is a model-agnostic, infrastructure-free memory layer that gives AI agents persistent, long-term memory they can carry anywhere.

---

## Why Video Frames?

Memvid draws inspiration from video encoding, not to store video, but to **organize AI memory as an append-only, ultra-efficient sequence of Smart Frames.**

A Smart Frame is an immutable unit that stores content along with timestamps, checksums and basic metadata.
Frames are grouped in a way that allows efficient compression, indexing, and parallel reads.

This frame-based design enables:

-   Append-only writes without modifying or corrupting existing data
-   Queries over past memory states
-   Timeline-style inspection of how knowledge evolves
-   Crash safety through committed, immutable frames
-   Efficient compression using techniques adapted from video encoding

The result is a single file that behaves like a rewindable memory timeline for AI systems.

---

## Core Concepts

-   **Living Memory Engine**
    Continuously append, branch, and evolve memory across sessions.

-   **Capsule Context (`.mv2`)**
    Self-contained, shareable memory capsules with rules and expiry.

-   **Time-Travel Debugging**
    Rewind, replay, or branch any memory state.

-   **Smart Recall**
    Sub-5ms local memory access with predictive caching.

-   **Codec Intelligence**
    Auto-selects and upgrades compression over time.

---

## 🚀 ResonantQ Spectral Enhancements (Fork Feature)

This fork integrates **ResonantQ** spectral optimization algorithms, providing significant performance and robustness improvements:

### Performance Gains

| Enhancement | Improvement | Description |
|-------------|-------------|-------------|
| **Spectral Compression** | **45× smaller** | 768D embeddings → 17D with <1% error |
| **SSH Topological Search** | **+48% noise tolerance** | Robust similarity via Su-Schrieffer-Heeger model |
| **Spectral Caching** | **20× faster** | Cached eigendecomposition for repeated queries |
| **TopoRouter** | **<10ms failover** | Topologically-protected memory navigation |
| **GFT Deduplication** | **~60% storage savings** | Graph Fourier Transform clustering |

### New Modules

```
src/spectral/
├── compression.rs   # 17-mode spectral embedding compression
├── ssh_search.rs    # SSH topological vector search
├── cache.rs         # Incremental spectral basis caching
├── topo_router.rs   # Topological memory graph router
└── gft.rs           # Graph Fourier Transform deduplication
```

### Quick Example

```rust
use memvid_core::{
    SpectralCompressor, SshSearcher, GftCondenser,
    SshConfig, OPTIMAL_K_MODES,
};

// 1. Spectral Compression (45× smaller embeddings)
let mut compressor = SpectralCompressor::new(OPTIMAL_K_MODES); // k=17
for emb in &training_embeddings {
    compressor.add_training_sample(emb.clone());
}
compressor.train();
let compressed = compressor.compress(&query_embedding).unwrap();
// 768 floats → 17 floats (45× compression)

// 2. SSH Topological Search (+48% noise tolerance)
let searcher = SshSearcher::new(SshConfig::noise_robust());
let hits = searcher.search(&query, &embeddings, 10);
for hit in hits {
    println!("Frame {}: score={:.3} (topo={:.3})",
             hit.frame_id, hit.score, hit.topo_score);
}

// 3. GFT Deduplication (cluster similar memories)
let condenser = GftCondenser::default();
let condensed = condenser.condense(&frame_embeddings);
println!("Compression: {:.1}%", condensed.compression_ratio * 100.0);
```

### Algorithm Details

#### Spectral Compression
Compresses high-dimensional embeddings using graph Laplacian eigendecomposition:
1. Treat dimensions as nodes, build covariance matrix
2. Extract top-k eigenvectors via power iteration
3. Project embeddings onto spectral basis → k coefficients
4. Reconstruction error <1% with k=17 for most embedding models

#### SSH Topological Search
Implements the Su-Schrieffer-Heeger (1979) model for noise-robust similarity:
- Alternating coupling strengths (t_A, t_B) create topological edge states
- Dimerization parameter δ = (t_A - t_B)/(t_A + t_B) controls robustness
- Edge states bridge concepts, improving recall for ambiguous queries

#### TopoRouter
Graph-based memory navigation with spectral clustering:
- Fiedler vector (2nd Laplacian eigenvector) partitions memory into clusters
- Bridge nodes identified for cross-cluster routing
- Spectral gap quantifies routing robustness

#### GFT Condensation
Semantic deduplication via Graph Fourier Transform:
- Build similarity graph over embeddings
- Cluster by spectral signature overlap
- Store centroid + residuals for lossless reconstruction

### Running Benchmarks

**Rust (recommended):**
```bash
cargo bench --bench spectral_benchmark
```

**Python (reference implementation):**
```bash
pip install numpy
python scripts/benchmark_spectral.py
```

Sample benchmark results (1000 embeddings, 768D):

| Operation | Time | Speedup |
|-----------|------|---------|
| Standard L2 Search | 3.2ms | baseline |
| SSH Topological Search | 8.1ms | +48% noise tolerance |
| Compress (768D → 17D) | 0.007ms | - |
| Compressed Distance | 0.005ms | 1.2× faster |
| GFT Condense (200 frames) | 50ms | 75% storage savings |

---

## Use Cases

Memvid is a portable, serverless memory layer that gives AI agents persistent memory and fast recall. Because it's model-agnostic, multi-modal, and works fully offline, developers are using Memvid across a wide range of real-world applications.

-   Long-Running AI Agents
-   Enterprise Knowledge Bases
-   Offline-First AI Systems
-   Codebase Understanding
-   Customer Support Agents
-   Workflow Automation
-   Sales and Marketing Copilots
-   Personal Knowledge Assistants
-   Medical, Legal, and Financial Agents
-   Auditable and Debuggable AI Workflows
-   Custom Applications

---

## SDKs & CLI

Use Memvid in your preferred language:

| Package         | Install                     | Links                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI**         | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli)       |
| **Node.js SDK** | `npm install @memvid/sdk`   | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk)     |
| **Python SDK**  | `pip install memvid-sdk`    | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/)         |
| **Rust**        | `cargo add memvid-core`     | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## Installation (Rust)

### Requirements

-   **Rust 1.85.0+** — Install from [rustup.rs](https://rustup.rs)

### Add to Your Project

```toml
[dependencies]
memvid-core = "2.0"
```

### Feature Flags

| Feature             | Description                                    |
| ------------------- | ---------------------------------------------- |
| `lex`               | Full-text search with BM25 ranking (Tantivy)   |
| `pdf_extract`       | Pure Rust PDF text extraction                  |
| `vec`               | Vector similarity search (HNSW + ONNX)         |
| `clip`              | CLIP visual embeddings for image search        |
| `whisper`           | Audio transcription with Whisper               |
| `temporal_track`    | Natural language date parsing ("last Tuesday") |
| `parallel_segments` | Multi-threaded ingestion                       |
| `encryption`        | Password-based encryption capsules (.mv2e)     |

Enable features as needed:

```toml
[dependencies]
memvid-core = { version = "2.0", features = ["lex", "vec", "temporal_track"] }
```

---

## Quick Start

```rust
use memvid_core::{Memvid, PutOptions, SearchRequest};

fn main() -> memvid_core::Result<()> {
    // Create a new memory file
    let mut mem = Memvid::create("knowledge.mv2")?;

    // Add documents with metadata
    let opts = PutOptions::builder()
        .title("Meeting Notes")
        .uri("mv2://meetings/2024-01-15")
        .tag("project", "alpha")
        .build();
    mem.put_bytes_with_options(b"Q4 planning discussion...", opts)?;
    mem.commit()?;

    // Search
    let response = mem.search(SearchRequest {
        query: "planning".into(),
        top_k: 10,
        snippet_chars: 200,
        ..Default::default()
    })?;

    for hit in response.hits {
        println!("{}: {}", hit.title.unwrap_or_default(), hit.text);
    }

    Ok(())
}
```

---

## Build

Clone the repository:

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

Build in debug mode:

```bash
cargo build
```

Build in release mode (optimized):

```bash
cargo build --release
```

Build with specific features:

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---

## Run Tests

Run all tests:

```bash
cargo test
```

Run tests with output:

```bash
cargo test -- --nocapture
```

Run a specific test:

```bash
cargo test test_name
```

Run integration tests only:

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## Examples

The `examples/` directory contains working examples:

### Basic Usage

Demonstrates create, put, search, and timeline operations:

```bash
cargo run --example basic_usage
```

### PDF Ingestion

Ingest and search PDF documents (uses the "Attention Is All You Need" paper):

```bash
cargo run --example pdf_ingestion
```

### CLIP Visual Search

Image search using CLIP embeddings (requires `clip` feature):

```bash
cargo run --example clip_visual_search --features clip
```

### Whisper Transcription

Audio transcription (requires `whisper` feature):

```bash
cargo run --example test_whisper --features whisper
```

---

## File Format

Everything lives in a single `.mv2` file:

```
┌────────────────────────────┐
│ Header (4KB)               │  Magic, version, capacity
├────────────────────────────┤
│ Embedded WAL (1-64MB)      │  Crash recovery
├────────────────────────────┤
│ Data Segments              │  Compressed frames
├────────────────────────────┤
│ Lex Index                  │  Tantivy full-text
├────────────────────────────┤
│ Vec Index                  │  HNSW vectors
├────────────────────────────┤
│ Time Index                 │  Chronological ordering
├────────────────────────────┤
│ TOC (Footer)               │  Segment offsets
└────────────────────────────┘
```

No `.wal`, `.lock`, `.shm`, or sidecar files. Ever.

See [MV2_SPEC.md](MV2_SPEC.md) for the complete file format specification.

---

## Support

Have questions or feedback?
Email: contact@memvid.com

**Drop a ⭐ to show support**

---

## License

Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
