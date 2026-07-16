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
  <strong>Memvid oferă un sistem de memorie într-un singur fișier pentru agenți AI, cu regăsire instantanee și memorie pe termen lung.</strong><br/>
  Memorie persistentă, versionată și portabilă, fără baze de date.
</p>

<!-- NAV:START -->
<p align="center">
  <a href="https://www.memvid.com">Website</a>
  ·
  <a href="https://sandbox.memvid.com">Deschide sandbox-ul</a>
  ·
  <a href="https://docs.memvid.com">Documentație</a>
  ·
  <a href="https://github.com/memvid/memvid/discussions">Discuții</a>
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

## Rezultate din benchmark-uri

**🚀 Acuratețe mai mare decât orice alt sistem de memorie:** +35% SOTA pe LoCoMo, cu regăsire conversațională și raționament pe termen lung la cel mai înalt nivel.

**🧠 Raționament multi-hop și temporal superior:** +76% la multi-hop și +56% la temporal față de media industriei.

**⚡ Latență ultra-redusă la scară:** 0.025 ms P50 și 0.075 ms P99, cu throughput de 1,372× mai mare decât abordările standard.

**🔬 Benchmark-uri complet reproductibile:** LoCoMo (10 conversații de ~26K tokeni), evaluare open source și LLM-as-Judge.


## Ce este Memvid?

Memvid este un sistem portabil de memorie AI care împachetează datele, embedding-urile, structura de căutare și metadatele într-un singur fișier.

În loc să rulezi pipeline-uri RAG complexe sau baze de date vectoriale pe server, Memvid permite regăsirea rapidă direct din fișier.

Rezultatul este o memorie fără infrastructură, agnostică față de model, care le oferă agenților AI memorie persistentă, pe termen lung, pe care o pot lua oriunde.

    

## Ce sunt Smart Frames?

Memvid se inspiră din codarea video, nu pentru a stoca video, ci pentru a **organiza memoria AI ca o secvență append-only, ultra-eficientă, de Smart Frames.**

Un Smart Frame este o unitate imuabilă care stochează conținut împreună cu timestamp-uri, checksum-uri și metadate de bază.
Frame-urile sunt grupate într-un mod care permite compresie, indexare și citiri paralele eficiente.

Acest design bazat pe frame-uri permite:

-   Scrieri append-only fără modificarea sau coruperea datelor existente
-   Interogări peste stări anterioare ale memoriei
-   Inspectarea evoluției cunoștințelor pe axa temporală
-   Siguranță la crash prin frame-uri confirmate și imutabile
-   Compresie eficientă folosind tehnici adaptate din codarea video

Rezultatul este un singur fișier care se comportă ca un timeline de memorie derulabil pentru sisteme AI.


## Concepte principale

-   **Living Memory Engine**
    Adaugă, ramifică și evoluează memoria continuu între sesiuni.

-   **Capsule Context (`.mv2`)**
    Capsule de memorie autonome și partajabile, cu reguli și expirare.

-   **Time-Travel Debugging**
    Derulează, redă sau ramifică orice stare a memoriei.

-   **Smart Recall**
    Acces local la memorie în sub 5 ms, cu cache predictiv.

-   **Codec Intelligence**
    Selectează și actualizează automat compresia în timp.


## Cazuri de utilizare

Memvid este un strat de memorie portabilă, serverless, care oferă agenților AI memorie persistentă și regăsire rapidă. Pentru că este agnostică față de model, multimodală și funcționează complet offline, dezvoltatorii folosesc Memvid într-o gamă largă de aplicații.

-   Agenți AI de lungă durată
-   Baze de cunoștințe enterprise
-   Sisteme AI offline-first
-   Înțelegerea codebase-urilor
-   Agenți de suport pentru clienți
-   Automatizarea workflow-urilor
-   Copiloți pentru vânzări și marketing
-   Asistenți personali de cunoștințe
-   Agenți medicali, juridici și financiari
-   Workflow-uri AI auditabile și depanabile
-   Aplicații personalizate


## SDK-uri & CLI

Folosește Memvid în limbajul preferat:

| Package         | Install                     | Links                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI**         | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli)       |
| **Node.js SDK** | `npm install @memvid/sdk`   | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk)     |
| **Python SDK**  | `pip install memvid-sdk`    | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/)         |
| **Rust**        | `cargo add memvid-core`     | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## Instalare (Rust)

### Cerințe

-   **Rust 1.85.0+** — Instalează de la [rustup.rs](https://rustup.rs)

### Adaugă în proiectul tău

```toml
[dependencies]
memvid-core = "2.0"
```

### Feature Flags

| Feature             | Descriere                                                                 |
| ------------------- | ------------------------------------------------------------------------- |
| `lex`               | Căutare full-text cu ranking BM25 (Tantivy)                               |
| `pdf_extract`       | Extragere de text PDF în Rust pur                                         |
| `vec`               | Căutare prin similitudine vectorială (HNSW + embedding-uri text locale prin ONNX) |
| `clip`              | Embedding-uri vizuale CLIP pentru căutare de imagini                      |
| `whisper`           | Transcriere audio cu Whisper                                              |
| `api_embed`         | Embedding-uri cloud prin API (OpenAI)                                     |
| `temporal_track`    | Parsarea datelor calendaristice în limbaj natural ("last Tuesday")        |
| `parallel_segments` | Ingestie multi-threaded                                                   |
| `encryption`        | Capsule criptate pe bază de parolă (.mv2e)                                |
| `symspell_cleanup`  | Reparare robustă a textului PDF (corectează "emp lo yee" -> "employee")  |

Activează feature-urile după nevoie:

```toml
[dependencies]
memvid-core = { version = "2.0", features = ["lex", "vec", "temporal_track"] }
```


## Pornire

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

Clonează repository-ul:

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

Build în modul debug:

```bash
cargo build
```

Build în modul release (optimizat):

```bash
cargo build --release
```

Build cu feature-uri specifice:

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---

## Rulează testele

Rulează toate testele:

```bash
cargo test
```

Rulează testele cu output:

```bash
cargo test -- --nocapture
```

Rulează un test specific:

```bash
cargo test test_name
```

Rulează doar testele de integrare:

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## Exemple

Directorul `examples/` conține exemple funcționale:

### Utilizare de bază

Demonstrează operațiile create, put, search și timeline:

```bash
cargo run --example basic_usage
```

### Ingestie PDF

Importă și caută documente PDF (folosește articolul "Attention Is All You Need"):

```bash
cargo run --example pdf_ingestion
```

### Căutare vizuală CLIP

Căutare de imagini folosind embedding-uri CLIP (necesită feature-ul `clip`):

```bash
cargo run --example clip_visual_search --features clip
```

### Transcriere Whisper

Transcriere audio (necesită feature-ul `whisper`):

```bash
cargo run --example test_whisper --features whisper -- /path/to/audio.mp3
```

**Modele disponibile:**

| Model                 | Dimensiune | Viteză       | Caz de utilizare                         |
| --------------------- | ---------- | ------------ | ---------------------------------------- |
| `whisper-small-en`    | 244 MB     | Cel mai lent | Cea mai bună acuratețe (implicit)        |
| `whisper-tiny-en`     | 75 MB      | Rapid        | Echilibrat                               |
| `whisper-tiny-en-q8k` | 19 MB      | Cel mai rapid | Testare rapidă, resurse limitate        |

**Selectarea modelului:**

```bash
# Default (FP32 small, highest accuracy)
cargo run --example test_whisper --features whisper -- audio.mp3

# Quantized tiny (75% smaller, faster)
MEMVID_WHISPER_MODEL=whisper-tiny-en-q8k cargo run --example test_whisper --features whisper -- audio.mp3
```

**Configurare programatică:**

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


## Modele de embedding-uri text

Feature-ul `vec` include suport pentru embedding-uri text locale folosind modele ONNX. Înainte să folosești embedding-uri text locale, trebuie să descarci manual fișierele modelului.

### Pornire rapidă: BGE-small (recomandat)

Descarcă modelul BGE-small implicit (384 dimensiuni, rapid și eficient):

```bash
mkdir -p ~/.cache/memvid/text-models

# Download ONNX model
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5.onnx

# Download tokenizer
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5_tokenizer.json
```

### Modele disponibile

| Model                   | Dimensiuni | Dimensiune | Potrivit pentru        |
| ----------------------- | ---------- | ---------- | ---------------------- |
| `bge-small-en-v1.5`     | 384        | ~120MB     | Implicit, rapid        |
| `bge-base-en-v1.5`      | 768        | ~420MB     | Calitate mai bună      |
| `nomic-embed-text-v1.5` | 768        | ~530MB     | Sarcini versatile      |
| `gte-large`             | 1024       | ~1.3GB     | Cea mai bună calitate  |

### Alte modele

**BGE-base** (768 dimensiuni):
```bash
curl -L 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-base-en-v1.5.onnx
curl -L 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-base-en-v1.5_tokenizer.json
```

**Nomic** (768 dimensiuni):
```bash
curl -L 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/nomic-embed-text-v1.5.onnx
curl -L 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/nomic-embed-text-v1.5_tokenizer.json
```

**GTE-large** (1024 dimensiuni):
```bash
curl -L 'https://huggingface.co/thenlper/gte-large/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/gte-large.onnx
curl -L 'https://huggingface.co/thenlper/gte-large/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/gte-large_tokenizer.json
```

### Utilizare în cod

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

Vezi `examples/text_embedding.rs` pentru un exemplu complet cu calculul similarității și ranking de căutare.

### Consistența modelelor

Pentru a preveni amestecarea accidentală a modelelor (de exemplu, interogarea unui index BGE-small cu embedding-uri OpenAI), poți lega explicit instanța Memvid de un anumit nume de model:

```rust
// Bind the index to a specific model.
// If the index was previously created with a different model, this will return an error.
mem.set_vec_model("bge-small-en-v1.5")?;
```

Această relație este persistentă. Odată setată, încercările viitoare de a folosi un alt nume de model vor eșua rapid cu o eroare `ModelMismatch`.


## API Embeddings (OpenAI)

Feature-ul `api_embed` permite generarea de embedding-uri în cloud folosind API-ul OpenAI.

### Configurare

Setează cheia ta API OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

### Utilizare

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

### Modele disponibile

| Model                    | Dimensiuni | Potrivit pentru              |
| ------------------------ | ---------- | ---------------------------- |
| `text-embedding-3-small` | 1536       | Implicit, cel mai rapid, cel mai ieftin |
| `text-embedding-3-large` | 3072       | Cea mai bună calitate        |
| `text-embedding-ada-002` | 1536       | Model legacy                 |

Vezi `examples/openai_embedding.rs` pentru un exemplu complet.


## Formatul fișierului

Totul se află într-un singur fișier `.mv2`:

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

Fără fișiere `.wal`, `.lock`, `.shm` sau sidecar. Niciodată.

Vezi [MV2_SPEC.md](MV2_SPEC.md) pentru specificația completă a formatului de fișier.


## Suport

Ai întrebări sau feedback?
Email: contact@memvid.com

**Lasă o ⭐ pentru a arăta susținere**

---

> **Memvid v1 (memorie bazată pe QR) nu mai este suportat**
>
> Dacă vezi referințe la coduri QR, folosești informații învechite.
>
> Vezi: https://docs.memvid.com/memvid-v1-deprecation

---

## Licență

Apache License 2.0 — vezi fișierul [LICENSE](LICENSE) pentru detalii.
