<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)" src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<div style="height: 16px;"></div>

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <strong>Memvid is een geheugenlaag van één bestand voor AI-agenten met directe toegang en langetermijnsgeheugen.</strong><br/>
  Volhardend en draagbaar geheugen met versiebeheer en zonder databases.
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
  ·
  <a href="docs/i18n/translation_hub.md">Translations</a>
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
<!-- BADGES:END -->

<h2 align="center">⭐️ Laat een ster achter om het project te steunen ⭐️</h2>
</p>

## Wat is Memvid?

Memvid is een draagbaar AI-geheugensysteem dat uw data, embedding, zoekstructuur en metadata in één bestand opslaat.

In plaats van complexe RAG pijplijnen of servergebaseerde vectordatabases te gebruiken, zal Memvid snelle toegang recht vanuit het bestand toestaan.

Het resultaat is een model-agnostische, infrastructuurvrije geheugenlaag die AI-agenten een volhardende langetermijnsgeheugen geeft, die ze overal kunnen meenemen.

---

## Waarom videoframes?

Memvid neemt inspiratie uit videos encoderen, niet om de video op te slaan, maar om **het organiseren van AI-geheugen als een ultra-efficiënte sequentie van Smart Frames waarbij je enkel kan toevoegen.**

Een Smart Frame is een immutabele eenheid die content opslaat samen met zijn tijdstempels, controlesommen en basismetadata.
Frames worden gegroupeerd in een manier die voor efficiënte compressie, indexing en parallele lezingen zorgt.

Dit frame-gebaseerde design maakt het volgende mogelijk:

- Append-only bijschrijven van data zonder het aanpassen of corrumperen van bestaande data
- Zoekopdrachten over vorige geheugenstaten
- Tijdlijn-stijl inspectie van hoe kennis evolueert
- Crashveiligheid door de vastgelegde immutabele frames
- Efficiënte compressie gebruikmakend van technieken aangepast uit video encoderen

Het resultaat is één bestand dat werkt als een terugspoelbare geheugentijdslijn van AI-systemen.

---

## Basisconcepten

-   **Living Memory Engine**
    Append, vertakt en evolueert geheugen continu over sessies.

-   **Capsule Context (`.mv2`)**
    Autonome, deelbaar geheugencapsules met regels en vervalling.

-   **Time-Travel Debugging**
    Spoel terug, herspeel, of vertak elke geheugenstatus.

-   **Smart Recall**
    Sub-5ms lokale geheugentoegang met voorspelbare caching.

-   **Codec Intelligence**
    Selecteert en verbetert automatisch de compressie doorheen de tijd.

---

## Gebruiksgevallen

Memvid is een draagbare, serverloze geheugenlaag dat AI-agenten een volhardend geheugen en snelle herroepingen geeft. Door zijn model-agnostische, multi-modale en het feit dat het volledig offline werkt, gebruiken ontwikkelaars het over een wijd scala aan real-world applicaties.

-   Lang werkende AI-agenten
-   Kennisbanken voor ondernemingen
-   Offline-First AI-systemen
-   Codebase-begrip
-   Klantenondersteuningsagenten
-   Automatisering van de workflow
-   Verkoop- en marketingcopiloten
-   Persoonlijke Kennisassistenten
-   Medische, juridische en financiële adviseurs
-   Controleerbare en debugbare AI-workflows
-   Aangepaste toepassingen

---

## SDKs & CLI

Gebruik Memvid in je lievelingstaal:

| Pakket          | Installatie                 | Links                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI**         | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli)       |
| **Node.js SDK** | `npm install @memvid/sdk`   | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk)     |
| **Python SDK**  | `pip install memvid-sdk`    | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/)         |
| **Rust**        | `cargo add memvid-core`     | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## Installatie (Rust)

### Benodigdheden

-   **Rust 1.85.0+** — Installeer vanuit [rustup.rs](https://rustup.rs)

### Voeg dit aan je project toe

```toml
[dependencies]
memvid-core = "2.0"
```

### Feature Flags

| Feature             | Beschrijving                                   |
| ------------------- | ---------------------------------------------- |
| `lex`               | Full-text search with BM25 ranking (Tantivy)   |
| `pdf_extract`       | Pure Rust PDF text extraction                  |
| `vec`               | Vector similarity search (HNSW + ONNX)         |
| `clip`              | CLIP visual embeddings for image search        |
| `whisper`           | Audio transcription with Whisper               |
| `temporal_track`    | Natural language date parsing ("last Tuesday") |
| `parallel_segments` | Multi-threaded ingestion                       |
| `encryption`        | Password-based encryption capsules (.mv2e)     |

Schakel functies in indien nodig:

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

Clone de repository:

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

Build in debug modus:

```bash
cargo build
```

Build in release modus (geoptimaliseerd):

```bash
cargo build --release
```

Build with specific features:

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---

## Tests uitvoeren

Voer alle tests uit:

```bash
cargo test
```

Voer tests uit met uitvoer:

```bash
cargo test -- --nocapture
```

Voer een specifieke test uit:

```bash
cargo test test_name
```

Voer enkel integratie tests uit:

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## Voorbeelden

De `examples/` map bedraagd werkende Voorbeelden:

### Basisgebruik

Beeldt create, put, search, and timeline operaties uit:

```bash
cargo run --example basic_usage
```

### PDF Ingestion

PDF-documenten importeren en doorzoeken (gebruikt de "Attention Is All You Need" paper):

```bash
cargo run --example pdf_ingestion
```

### CLIP Visual Search

Afbeeldingen zoeken met behulp van CLIP-integraties (gebruikt `clip` feature):

```bash
cargo run --example clip_visual_search --features clip
```

### Whisper Transcription

Audio transcripties (gebruikt `whisper` feature):

```bash
cargo run --example test_whisper --features whisper
```

---

## Bestandsformaat

Alles leeft in één `.mv2` bestand:

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

Geen `.wal`, `.lock`, `.shm`, of sidecar-bestanden. Ooit.

Zie [MV2_SPEC.md](MV2_SPEC.md) voor de complete bestandsformaat specificaties.

---

## Ondersteuning

Heb je vragen of feedback?
Email: contact@memvid.com

**Laat een ⭐ om je ondersteuning te tonen**

---

## Licentie

Apache License 2.0 — zie het [LICENSE](LICENSE) bestandvoor details.
