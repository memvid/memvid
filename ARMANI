<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)"
     src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<div style="height: 16px;"></div>

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>
<!-- BADGES:END -->

<p align="center">
  <strong>Memvid is a single-file memory layer for AI agents with instant retrieval and long-term memory.</strong><br/>
  Persistent, versioned, and portable memory, without databases.
</p>

<!-- NAV:START -->
<p align="center">
  <a href="https://www.memvid.com">Website</a>
  ·
  <a href="https://sandbox.memvid.com">Try Sandbox</a>
  ·
  <a href="https://docs.memvid.com">Docs</a>
  ·
  <a href="https://github.com/memvid/memvid/discussions">Discussions</a>
</p>
<!-- NAV:END -->

<!-- BADGES:START -->
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




## Benchmark Highlights

**🚀 Higher accuracy than any other memory system :** +35% SOTA on LoCoMo, best-in-class long-horizon conversational recall & reasoning

**🧠 Superior multi-hop & temporal reasoning:**  +76% multi-hop, +56% temporal vs. the industry average

**⚡ Ultra-low latency at scale** 0.025ms P50 and 0.075ms P99, with 1,372× higher throughput than standard

**🔬 Fully reproducible benchmarks:** LoCoMo (10 × ~26K-token conversations), open-source eval, LLM-as-Judge


## What is Memvid?

Memvid is a portable AI memory system that packages your data, embeddings, search structure, and metadata into a single file.

Instead of running complex RAG pipelines or server-based vector databases, Memvid enables fast retrieval directly from the file.

The result is a model-agnostic, infrastructure-free memory layer that gives AI agents persistent, long-term memory they can carry anywhere.

    
## What are Smart Frames?

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

| Feature             | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `lex`               | Full-text search with BM25 ranking (Tantivy)                     |
| `pdf_extract`       | Pure Rust PDF text extraction                                    |
| `vec`               | Vector similarity search (HNSW + local text embeddings via ONNX) |
| `clip`              | CLIP visual embeddings for image search                          |
| `whisper`           | Audio transcription with Whisper                                 |
| `api_embed`         | Cloud API embeddings (OpenAI)                                    |
| `temporal_track`    | Natural language date parsing ("last Tuesday")                   |
| `parallel_segments` | Multi-threaded ingestion                                         |
| `encryption`        | Password-based encryption capsules (.mv2e)                       |
| `symspell_cleanup`  | Robust PDF text repair (fixes "emp lo yee" -> "employee")        |

Enable features as needed:

```toml
[dependencies]
memvid-core = { version = "2.0", features = ["lex", "vec", "temporal_track"] }
```


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
cargo run --example test_whisper --features whisper -- /path/to/audio.mp3
```

**Available Models:**

| Model                 | Size   | Speed   | Use Case                            |
| --------------------- | ------ | ------- | ----------------------------------- |
| `whisper-small-en`    | 244 MB | Slowest | Best accuracy (default)             |
| `whisper-tiny-en`     | 75 MB  | Fast    | Balanced                            |
| `whisper-tiny-en-q8k` | 19 MB  | Fastest | Quick testing, resource-constrained |

**Model Selection:**

```bash
# Default (FP32 small, highest accuracy)
cargo run --example test_whisper --features whisper -- audio.mp3

# Quantized tiny (75% smaller, faster)
MEMVID_WHISPER_MODEL=whisper-tiny-en-q8k cargo run --example test_whisper --features whisper -- audio.mp3
```

**Programmatic Configuration:**

```rust
use memvid_core::{WhisperConfig, WhisperTranscriber};

// Default FP32 small model
let config = WhisperConfig::default();

// Quantized tiny model (faster, smaller)
let config = WhisperConfig::with_quantization();

// Specific model
let config = WhisperConfig::with_model("whisper-tiny-en-q8k");

let transcriber = WhisperTranscriber::new(&config)?;
let result = transcriber.transcribe_file("audio.mp3")?;
println!("{}", result.text);
```


## Text Embedding Models

The `vec` feature includes local text embedding support using ONNX models. Before using local text embeddings, you need to download the model files manually.

### Quick Start: BGE-small (Recommended)

Download the default BGE-small model (384 dimensions, fast and efficient):

```bash
mkdir -p ~/.cache/memvid/text-models

# Download ONNX model
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5.onnx

# Download tokenizer
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5_tokenizer.json
```

### Available Models

| Model                   | Dimensions | Size   | Best For        |
| ----------------------- | ---------- | ------ | --------------- |
| `bge-small-en-v1.5`     | 384        | ~120MB | Default, fast   |
| `bge-base-en-v1.5`      | 768        | ~420MB | Better quality  |
| `nomic-embed-text-v1.5` | 768        | ~530MB | Versatile tasks |
| `gte-large`             | 1024       | ~1.3GB | Highest quality |

### Other Models

**BGE-base** (768 dimensions):
```bash
curl -L 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-base-en-v1.5.onnx
curl -L 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-base-en-v1.5_tokenizer.json
```

**Nomic** (768 dimensions):
```bash
curl -L 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/nomic-embed-text-v1.5.onnx
curl -L 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/nomic-embed-text-v1.5_tokenizer.json
```

**GTE-large** (1024 dimensions):
```bash
curl -L 'https://huggingface.co/thenlper/gte-large/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/gte-large.onnx
curl -L 'https://huggingface.co/thenlper/gte-large/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/gte-large_tokenizer.json
```

### Usage in Code

```rust
use memvid_core::text_embed::{LocalTextEmbedder, TextEmbedConfig};
use memvid_core::types::embedding::EmbeddingProvider;

// Use default model (BGE-small)
let config = TextEmbedConfig::default();
let embedder = LocalTextEmbedder::new(config)?;

let embedding = embedder.embed_text("hello world")?;
assert_eq!(embedding.len(), 384);

// Use different model
let config = TextEmbedConfig::bge_base();
let embedder = LocalTextEmbedder::new(config)?;
```

See `examples/text_embedding.rs` for a complete example with similarity computation and search ranking.

### Model Consistency

To prevent accidental model mixing (e.g., querying a BGE-small index with OpenAI embeddings), you can explicitly bind your Memvid instance to a specific model name:

```rust
// Bind the index to a specific model.
// If the index was previously created with a different model, this will return an error.
mem.set_vec_model("bge-small-en-v1.5")?;
```

This binding is persistent. Once set, future attempts to use a different model name will fail fast with a `ModelMismatch` error.



## API Embeddings (OpenAI)

The `api_embed` feature enables cloud-based embedding generation using OpenAI's API.

### Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

### Usage

```rust
use memvid_core::api_embed::{OpenAIConfig, OpenAIEmbedder};
use memvid_core::types::embedding::EmbeddingProvider;

// Use default model (text-embedding-3-small)
let config = OpenAIConfig::default();
let embedder = OpenAIEmbedder::new(config)?;

let embedding = embedder.embed_text("hello world")?;
assert_eq!(embedding.len(), 1536);

// Use higher quality model
let config = OpenAIConfig::large();  // text-embedding-3-large (3072 dims)
let embedder = OpenAIEmbedder::new(config)?;
```

### Available Models

| Model                    | Dimensions | Best For                   |
| ------------------------ | ---------- | -------------------------- |
| `text-embedding-3-small` | 1536       | Default, fastest, cheapest |
| `text-embedding-3-large` | 3072       | Highest quality            |
| `text-embedding-ada-002` | 1536       | Legacy model               |

See `examples/openai_embedding.rs` for a complete example.



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



## Support

Have questions or feedback?
Email: contact@memvid.com

**Drop a ⭐ to show support**

---

> **Memvid v1 (QR-based memory) is deprecated**
>
> If you are referencing QR codes, you are using outdated information.
>
> See: https://docs.memvid.com/memvid-v1-deprecation

---

## License

Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
