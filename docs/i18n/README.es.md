<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)" src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<!-- FLAGS:START -->
<p align="center">
 <a href="../../README.md">🇺🇸 English</a>
 <a href="README.es.md">🇪🇸 Español</a>
 <a href="README.fr.md">🇫🇷 Français</a>
 <a href="README.so.md">🇸🇴 Soomaali</a>
 <a href="README.ar.md">🇸🇦 العربية</a>
 <a href="README.nl.md">🇧🇪/🇳🇱 Nederlands</a>
 <a href="README.hi.md">🇮🇳 हिन्दी</a>
 <a href="README.bn.md">🇧🇩 বাংলা</a>
 <a href="README.cs.md">🇨🇿 Čeština</a>
 <a href="README.ko.md">🇰🇷 한국어</a>
 <a href="README.ja.md">🇯🇵 日本語</a>
 <!-- Next Flag -->
</p>
<!-- FLAGS:END -->

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

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>
<!-- BADGES:END -->

<p align="center">
  <strong>Memvid es una capa de memoria de un solo archivo para agentes de IA, con recuperación instantánea y memoria a largo plazo.</strong><br/>
  Memoria persistente, versionada y portable, sin bases de datos.
</p>

<h2 align="center">⭐️ Deja una STAR para apoyar el proyecto ⭐️</h2>
</p>

## Lo más destacado de los benchmarks

**🚀 Mayor precisión que cualquier otro sistema de memoria:** +35% SOTA en LoCoMo, con recall y razonamiento conversacional de largo horizonte de primer nivel.

**🧠 Mejor razonamiento multi-hop y temporal:** +76% en multi-hop y +56% en temporal frente al promedio de la industria.

**⚡ Latencia ultra baja a escala:** 0.025 ms P50 y 0.075 ms P99, con 1,372× más throughput que los enfoques estándar.

**🔬 Benchmarks totalmente reproducibles:** LoCoMo (10 conversaciones de ~26K tokens), evaluación open source y LLM-as-Judge.

---

## ¿Qué es Memvid?

Memvid es un sistema de memoria portable para IA que empaqueta tus datos, embeddings, estructura de búsqueda y metadatos en un solo archivo.

En lugar de ejecutar pipelines RAG complejos o bases de datos vectoriales basadas en servidor, Memvid permite una recuperación rápida directamente desde el archivo.

El resultado es una capa de memoria agnóstica al modelo, sin infraestructura, que da a los agentes de IA una memoria persistente y a largo plazo que pueden llevar a cualquier parte.

---

## ¿Por qué fotogramas de vídeo?

Memvid se inspira en la codificación de vídeo, no para almacenar vídeo, sino para **organizar la memoria de IA como una secuencia de Smart Frames ultrarrápida y append-only.**

Un Smart Frame es una unidad inmutable que almacena contenido junto con marcas de tiempo (timestamps), checksums y metadatos básicos.
Los frames se agrupan de una forma que permite una compresión, indexación y lecturas paralelas eficientes.

Este diseño basado en frames permite:

-   Escrituras append-only sin modificar ni corromper los datos existentes
-   Consultas sobre estados pasados de la memoria
-   Inspección estilo línea temporal (timeline) de cómo evoluciona el conocimiento
-   Seguridad ante fallos (crash safety) mediante frames confirmados e inmutables
-   Compresión eficiente usando técnicas adaptadas de la codificación de vídeo

El resultado es un único archivo que se comporta como una línea temporal de memoria “rebobinable” para sistemas de IA.

---

## Conceptos principales

-   **Living Memory Engine**
    Añade, ramifica (branch) y evoluciona la memoria de forma continua entre sesiones.

-   **Capsule Context (`.mv2`)**
    Cápsulas de memoria autocontenidas y compartibles, con reglas y caducidad.

-   **Time-Travel Debugging**
    Rebobina, reproduce (replay) o ramifica cualquier estado de memoria.

-   **Smart Recall**
    Acceso local a memoria en menos de 5ms con caché predictiva.

-   **Codec Intelligence**
    Selecciona y actualiza la compresión automáticamente con el tiempo.

---

## Casos de uso

Memvid es una capa de memoria portable y serverless que da a los agentes de IA memoria persistente y recuerdo rápido. Como es agnóstica al modelo, multi-modal y funciona totalmente offline, los desarrolladores están usando Memvid en una amplia gama de aplicaciones reales.

-   Agentes de IA de larga duración
-   Bases de conocimiento empresariales
-   Sistemas de IA offline-first
-   Comprensión de codebases
-   Agentes de soporte al cliente
-   Automatización de flujos de trabajo
-   Copilotos de ventas y marketing
-   Asistentes de conocimiento personal
-   Agentes médicos, legales y financieros
-   Flujos de trabajo de IA auditables y depurables
-   Aplicaciones personalizadas

---

## SDKs & CLI

Usa Memvid en tu lenguaje preferido:

| Package         | Install                     | Links                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI**         | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli)       |
| **Node.js SDK** | `npm install @memvid/sdk`   | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk)     |
| **Python SDK**  | `pip install memvid-sdk`    | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/)         |
| **Rust**        | `cargo add memvid-core`     | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## Instalación (Rust)

### Requisitos

-   **Rust 1.85.0+** — Instálalo desde [rustup.rs](https://rustup.rs)

### Añadir a tu proyecto

```toml
[dependencies]
memvid-core = "2.0"
```

### Feature Flags

| Feature             | Descripción                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `lex`               | Búsqueda full-text con ranking BM25 (Tantivy)                      |
| `pdf_extract`       | Extracción de texto PDF 100% en Rust                               |
| `vec`               | Búsqueda por similitud vectorial (HNSW + embeddings locales vía ONNX) |
| `clip`              | Embeddings visuales CLIP para búsqueda de imágenes                 |
| `whisper`           | Transcripción de audio con Whisper                                 |
| `api_embed`         | Embeddings en la nube mediante API (OpenAI)                        |
| `temporal_track`    | Interpretación de fechas en lenguaje natural ("el martes pasado")  |
| `parallel_segments` | Ingesta multi-hilo                                                 |
| `encryption`        | Cápsulas cifradas con contraseña (.mv2e)                           |
| `symspell_cleanup`  | Reparación robusta de texto PDF (corrige "emp lo yee" -> "employee") |

Activa las features según lo necesites:

```toml
[dependencies]
memvid-core = { version = "2.0", features = ["lex", "vec", "temporal_track"] }
```

---

## Inicio rápido

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

Clona el repositorio:

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

Compila en modo debug:

```bash
cargo build
```

Compila en modo release (optimizado):

```bash
cargo build --release
```

Compila con features específicas:

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---

## Ejecutar tests

Ejecuta todos los tests:

```bash
cargo test
```

Ejecuta los tests con salida:

```bash
cargo test -- --nocapture
```

Ejecuta un test específico:

```bash
cargo test test_name
```

Ejecuta solo tests de integración:

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## Ejemplos

El directorio `examples/` contiene ejemplos funcionales:

### Uso básico

Demuestra operaciones de create, put, search y timeline:

```bash
cargo run --example basic_usage
```

### Ingesta de PDF

Ingiere y busca documentos PDF (usa el paper “Attention Is All You Need”):

```bash
cargo run --example pdf_ingestion
```

### Búsqueda visual con CLIP

Búsqueda de imágenes usando embeddings de CLIP (requiere la feature `clip`):

```bash
cargo run --example clip_visual_search --features clip
```

### Transcripción con Whisper

Transcripción de audio (requiere la feature `whisper`):

```bash
cargo run --example test_whisper --features whisper
```

---

## Modelos de embeddings de texto

La feature `vec` incluye soporte para embeddings de texto locales usando modelos ONNX. Antes de usar embeddings locales, necesitas descargar manualmente los archivos del modelo.

### Inicio rápido: BGE-small (recomendado)

Descarga el modelo BGE-small por defecto (384 dimensiones, rápido y eficiente):

```bash
mkdir -p ~/.cache/memvid/text-models

# Descargar modelo ONNX
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5.onnx

# Descargar tokenizer
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5_tokenizer.json
```

### Modelos disponibles

| Modelo                  | Dimensiones | Tamaño | Mejor para            |
| ----------------------- | ----------- | ------ | --------------------- |
| `bge-small-en-v1.5`     | 384         | ~120MB | Opción por defecto, rápido |
| `bge-base-en-v1.5`      | 768         | ~420MB | Mejor calidad         |
| `nomic-embed-text-v1.5` | 768         | ~530MB | Tareas versátiles     |
| `gte-large`             | 1024        | ~1.3GB | Máxima calidad        |

### Otros modelos

**BGE-base** (768 dimensiones):
```bash
curl -L 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-base-en-v1.5.onnx
curl -L 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-base-en-v1.5_tokenizer.json
```

**Nomic** (768 dimensiones):
```bash
curl -L 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/nomic-embed-text-v1.5.onnx
curl -L 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/nomic-embed-text-v1.5_tokenizer.json
```

**GTE-large** (1024 dimensiones):
```bash
curl -L 'https://huggingface.co/thenlper/gte-large/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/gte-large.onnx
curl -L 'https://huggingface.co/thenlper/gte-large/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/gte-large_tokenizer.json
```

### Uso en código

```rust
use memvid_core::text_embed::{LocalTextEmbedder, TextEmbedConfig};
use memvid_core::types::embedding::EmbeddingProvider;

// Usar el modelo por defecto (BGE-small)
let config = TextEmbedConfig::default();
let embedder = LocalTextEmbedder::new(config)?;

let embedding = embedder.embed_text("hello world")?;
assert_eq!(embedding.len(), 384);

// Usar un modelo distinto
let config = TextEmbedConfig::bge_base();
let embedder = LocalTextEmbedder::new(config)?;
```

Consulta `examples/text_embedding.rs` para ver un ejemplo completo con cálculo de similitud y ranking de búsqueda.

### Consistencia del modelo

Para evitar mezclar modelos por accidente, por ejemplo consultar un índice BGE-small con embeddings de OpenAI, puedes asociar explícitamente tu instancia de Memvid a un nombre de modelo:

```rust
// Vincula el índice a un modelo concreto.
// Si el índice ya fue creado con otro modelo, devolverá un error.
mem.set_vec_model("bge-small-en-v1.5")?;
```

Esta vinculación es persistente. Una vez definida, cualquier intento futuro de usar otro nombre de modelo fallará de inmediato con un error `ModelMismatch`.

---

## Embeddings por API (OpenAI)

La feature `api_embed` habilita la generación de embeddings en la nube usando la API de OpenAI.

### Configuración

Define tu clave de API de OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

### Uso

```rust
use memvid_core::api_embed::{OpenAIConfig, OpenAIEmbedder};
use memvid_core::types::embedding::EmbeddingProvider;

// Usar el modelo por defecto (text-embedding-3-small)
let config = OpenAIConfig::default();
let embedder = OpenAIEmbedder::new(config)?;

let embedding = embedder.embed_text("hello world")?;
assert_eq!(embedding.len(), 1536);

// Usar un modelo de mayor calidad
let config = OpenAIConfig::large();  // text-embedding-3-large (3072 dims)
let embedder = OpenAIEmbedder::new(config)?;
```

### Modelos disponibles

| Modelo                   | Dimensiones | Mejor para                       |
| ------------------------ | ----------- | -------------------------------- |
| `text-embedding-3-small` | 1536        | Por defecto, más rápido y económico |
| `text-embedding-3-large` | 3072        | Máxima calidad                   |
| `text-embedding-ada-002` | 1536        | Modelo heredado                  |

Consulta `examples/openai_embedding.rs` para ver un ejemplo completo.

---

## Formato de archivo

Todo vive en un único archivo `.mv2`:

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

Sin archivos `.wal`, `.lock`, `.shm` ni sidecars. Nunca.

Consulta [MV2_SPEC.md](MV2_SPEC.md) para la especificación completa del formato de archivo.

---

## Soporte

¿Tienes preguntas o feedback?
Email: contact@memvid.com

**Deja una ⭐ para mostrar apoyo**

---

> **Memvid v1 (memoria basada en QR) está obsoleto**
>
> Si estás viendo referencias a códigos QR, estás usando información desactualizada.
>
> Consulta: https://docs.memvid.com/memvid-v1-deprecation

---

## Licencia

Apache License 2.0 — consulta el archivo [LICENSE](LICENSE) para más detalles.
