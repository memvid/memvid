<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)" src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<div style="height: 16px;"></div>

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <strong>Memvid는 AI 에이전트를 위한 단일 파일 메모리 레이어로, 인스턴스 검색 및 장기 메모리 기능을 제공합니다.</strong><br/>
  데이터 베이스 없이 지속적이고, 버전 관리가 용이하며 여러 어플리케이션에 자유로운 적용이 가능합니다.
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

<h2 align="center">⭐️ STAR로 이 프로젝트를 지원해주세요 ⭐️</h2>
</p>

## Memvid란?

Memvid는 데이터, 임베딩, 검색 구조 및 메타데이터를 단일 파일로 패키징하는 이식 가능한 AI 메모리 시스템입니다.
​
복잡한 RAG 파이프라인이나 서버 기반 벡터 데이터베이스를 실행하는 대신, Memvid는 파일에서 직접 빠른 검색을 가능하게 합니다.

결과적으로 모델에 독립적이며 인프라 구조와는 독립적인 메모리 레이어로, AI 에이전트가 어디서나 휴대할 수 있는 지속적 장기 메모리를 제공합니다

---

## Smart Frames란?

Memvid는 **AI 메모리를 추가 전용(append-only)의 초고효율 Smart Frame 시퀀스로 구성하기 위해** 비디오 인코딩에서 영감을 받았습니다.

Smart Frame은 타임스탬프, 체크섬 및 기본 메타데이터와 함께 콘텐츠를 저장하는 불변 단위입니다.
프레임은 효율적인 압축, 인덱싱 및 병렬 읽기를 허용하는 방식으로 그룹화됩니다.

이러한 프레임 기반 설계는 다음을 가능하게 합니다:

-   기존 데이터를 수정하거나 손상시키지 않는 추가 전용(append-only) 쓰기
-   과거 메모리 상태에 대한 쿼리
-   지식이 어떻게 변화하는지에 대한 타임라인 스타일 검사
-   불변 프레임워크를 통한 크래시 안전성
-   비디오 인코딩에서 차용한 기술을 사용한 효율적인 압축

이를 위한 결과물은 AI 시스템을 위한 되감기 가능한 메모리 타임라인처럼 동작하는 단일 파일입니다.

---

## 주요 개념

-   **실시간 변화하는 메모리 엔진**
    세션 간에 메모리를 지속적으로 추가, 분기 및 변화시킵니다.

-   **문맥 캡슐화 (`.mv2`)**
    규칙과 만료 시간이 포함된 자립형 공유 가능 형대의 메모리 캡슐입니다.

-   **시간 기반 디버깅**
    임의의 메모리 상태로 되감기, 재생 또는 분기합니다.

-   **예측 기반 호출**
    예측 캐싱을 사용한 5ms 미만 로컬 메모리 액세스를 제공합니다.

-   **코덱 인텔리전스**
    시간 경과에 따라 압축을 자동 선택 및 업그레이드합니다.

---

## 이용 사례

Memvid 이동 가능한 서버리스 메모리 레이어로 AI 에이전트에 지속적인 메모리와 빠른 호출을 제공합니다. 이는 모델과 독립적이고, 멀티모달을 지원하며, 인터넷을 사용하지 않으므로, 개발자들은 다양한 실제 어플리케이션에서 Memvid를 활용하고 있습니다.

- 장기 실행 AI 에이전트
- 기업 내의 지식 베이스
- 오프라인 우선의 AI 시스템
- 코드베이스 이해
- 고객 지원 에이전트
- 워크플로 자동화
- 판매 및 마케팅 코파일럿
- 개인 지식 어시스턴트
- 의료, 법률 및 금융 에이전트
- 모니터링 및 디버깅 가능한 AI 워크플로
- 그 외의 여러 애플리케이션

---

## SDKs & CLI

원하는 언어로 Memvid를 사용하세요:

| 패키지         | 설치 커맨드                     | 링크                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI**         | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli)       |
| **Node.js SDK** | `npm install @memvid/sdk`   | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk)     |
| **Python SDK**  | `pip install memvid-sdk`    | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/)         |
| **Rust**        | `cargo add memvid-core`     | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## 설치 (Rust)

### 요구 사항

-   **Rust 1.85.0+** — [rustup.rs](https://rustup.rs)에서 설치 가능합니다.

### 프로젝트에 추가

```toml
[dependencies]
memvid-core = "2.0"
```

### Feature Flags

| Feature             | Description                                         |
| ------------------- | --------------------------------------------------- |
| `lex`               | BM25 랭킹 기반 전체 텍스트 검색 (Tantivy)              |
| `pdf_extract`       | Rust 기반 PDF 텍스트 추출                       |
| `vec`               | 벡터 유사도 검색 (HNSW + ONNX) |
| `clip`              | 이미지 검색을 위한 CLIP 임베딩             |
| `whisper`           | Whisper 기반 오디오 전사                    |
| `temporal_track`    | 자연어 날짜 추출 ("지난 화요일")      |
| `parallel_segments` | 멀티-스레딩 처리                            |
| `encryption`        | Password 기반 암호화 (.mv2e)          |

필요한 기능을 아래 방식으로 활성화하세요:

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

## 빌드

이 레포지토리 클론:

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

디버그 모드로 빌드:

```bash
cargo build
```

배포 모드로 빌드 (optimized):

```bash
cargo build --release
```

특수 기능을 포함하도록 빌드:

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---

## 테스트

전체 테스트 실행:

```bash
cargo test
```

테스트 실행 및 결과 출력:

```bash
cargo test -- --nocapture
```

특정 테스트 실행:

```bash
cargo test test_name
```

인테그레이션 테스트만 실행:

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## 예시

`examples/` 디렉토리에 예제가 있습니다:

### 기본 사용법

생성, 추가, 검색 및 타임라인 작업을 보여줍니다:

```bash
cargo run --example basic_usage
```

### PDF 수집

PDF 문서 수집 및 검색 ("Attention Is All You Need" 논문 사용):

```bash
cargo run --example pdf_ingestion
```

### CLIP 이미지 검색

CLIP 임베딩을 사용한 이미지 검색 (`clip` 기능 필요):

```bash
cargo run --example clip_visual_search --features clip
```

### Whisper 전사

오디오 전사 (`whisper` 기능 필요):

```bash
cargo run --example test_whisper --features whisper
```

---

## Text Embedding 모델

`vec` 기능은 ONNX 모델을 사용한 로컬 텍스트 임베딩을 포함합니다. 로컬 텍스트 임베딩을 사용하기 전에 모델 파일을 수동으로 다운로드해야 합니다.

### Quick Start: BGE-small (추천함)

기본 BGE-small 모델(384 차원, 빠르고 효율적) 다운로드:

```bash
mkdir -p ~/.cache/memvid/text-models

# Download ONNX model
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5.onnx

# Download tokenizer
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5_tokenizer.json
```

### 지원 모델

| 모델명                   | 차원 수     | 크기   | 권장 용도               |
| ----------------------- | ---------- | -----  | --------------------- |
| `bge-small-en-v1.5`     | 384        | ~120MB | 기본 설정, 가장 빠름     |
| `bge-base-en-v1.5`      | 768        | ~420MB | 꽤 좋은 성능            |
| `nomic-embed-text-v1.5` | 768        | ~530MB | 다양한 업무 가능       |
| `gte-large`             | 1024       | ~1.3GB | 가장 좋은 성능      |

### 타 모델

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

### 코드 내 사용법

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

유사도 계산 및 검색 랭킹이 포함된 전체 예제는 `examples/text_embedding.rs`를 참조하세요.

---

## 파일 구조

모든 구성 요소는 단일 `.mv2` 파일 내에 구성됩니다:

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

`.wal`, `.lock`, `.shm`, 혹은 그 외의 별도 구성 요소는 없습니다.

[MV2_SPEC.md](MV2_SPEC.md)에서 파일 세부 형식을 확인할 수 있습니다.

---

## Support

문의 사항은 아래 이메일로 부탁드립니다.
Email: contact@memvid.com

**⭐를 눌러 이 프로젝트를 지원해주세요**

---

## License

Apache License 2.0 — [LICENSE](LICENSE) 파일 참고.
