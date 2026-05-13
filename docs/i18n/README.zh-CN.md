<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)"
     src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<div style="height: 16px;"></div>

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <strong>Memvid 是专为 AI 智能体设计的单文件记忆层，具备即时检索和长期记忆能力。</strong><br/>
  持久化、版本化、可移植的记忆，无需数据库。
</p>

<!-- NAV:START -->
<p align="center">
  <a href="https://www.memvid.com">官方网站</a>
  ·
  <a href="https://sandbox.memvid.com">尝试一下沙箱</a>
  ·
  <a href="https://docs.memvid.com">文档</a>
  ·
  <a href="https://github.com/memvid/memvid/discussions">讨论区</a>
  ·
  <a href="docs/i18n/translation_hub.md">Translations</a>
</p>
<!-- NAV:END -->

<!-- BADGES:START -->
<p align="center">
  <a href="https://crates.io/crates/memvid-core"><img src="https://img.shields.io/crates/v/memvid-core?style=flat-square&logo=rust" alt="Crates.io" /></a>
  <a href="https://docs.rs/memvid-core"><img src="https://img.shields.io/docsrs/memvid-core?style=flat-square&logo=docs.rs" alt="docs.rs" /></a>
  <a href="https://github.com/memvid/memvid/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="许可证" /></a>
</p>

<p align="center">
  <a href="https://github.com/memvid/memvid/stargazers"><img src="https://img.shields.io/github/stars/memvid/memvid?style=flat-square&logo=github" alt="Stars" /></a>
  <a href="https://github.com/memvid/memvid/network/members"><img src="https://img.shields.io/github/forks/memvid/memvid?style=flat-square&logo=github" alt="Forks" /></a>
  <a href="https://github.com/memvid/memvid/issues"><img src="https://img.shields.io/github/issues/memvid/memvid?style=flat-square&logo=github" alt="Issues" /></a>
  <a href="https://discord.gg/2mynS7fcK7"><img src="https://img.shields.io/discord/1442910055233224745?style=flat-square&logo=discord&label=discord" alt="Discord" /></a>
</p>
<!-- BADGES:END -->

<h2 align="center">⭐️ 给项目点个星标支持我们 ⭐️</h2>

## 基准测试亮点

**🚀 准确率超越其他记忆系统：** 在 LoCoMo 上领先 SOTA 35%，长时对话回忆与推理能力最佳

**🧠 卓越的多跳与时序推理：** 比行业平均水平高出 76% 多跳推理，56% 时序推理

**⚡ 超低延迟高吞吐：** P50 仅 0.025ms，P99 仅 0.075ms，吞吐量是标准的 1,372 倍

**🔬 完全可复现的基准测试：** LoCoMo（10 次约 26K token 的对话）、开源评估、LLM-as-Judge


## 什么是 Memvid？

Memvid 是可移植的 AI 记忆系统，将数据、嵌入向量、搜索结构和元数据打包成单个文件。

无需运行复杂的 RAG 管道或基于服务器的向量数据库，Memvid 支持直接从文件进行快速检索。

结果是模型无关、无基础设施的记忆层，让 AI 智能体拥有可随身携带的持久化长期记忆。

     
## 什么是 Smart Frames？

Memvid 借鉴视频编码的理念，不是为了存储视频，而是**将 AI 记忆组织为仅追加、超高效序列的 Smart Frames。**

Smart Frame 是存储内容以及时间戳、校验和和基本元数据的不可变单元。
帧以允许高效压缩、索引和并行读取的方式分组。

这种基于帧的设计支持：

-   仅追加写入，不修改或破坏现有数据
-   对过去记忆状态的查询
-   知识演化的时间轴式检查
-   通过基于提交的不可变帧应对崩溃
-   使用基于视频编码技术的高效压缩

结果是一个表现为 AI 系统可追溯记忆时间线的单文件。


## 核心概念

-   **Living Memory Engine**
    持续追加、分支和跨会话演进记忆。

-   **Capsule Context (`.mv2`)**
    自包含、可共享的记忆胶囊，带规则和过期时间。

-   **Time-Travel Debugging**
    回溯、重放或分支化任何记忆状态。

-   **Smart Recall**
    小于 5ms 本地记忆访问，具备预测性缓存。

-   **Codec Intelligence**
    随时间自动选择和升级压缩。


## 使用场景

Memvid 是无服务器便携记忆层，为 AI 智能体提供持久记忆和快速召回。由于它是模型无关、多模态且完全离线工作，开发者正在各种现实应用中广泛的使用 Memvid。

-   长期运行的 AI 智能体
-   企业知识库
-   离线优先 AI 系统
-   代码库理解
-   客户支持智能体
-   工作流自动化
-   销售与营销助手
-   个人知识助理
-   医疗、法律和金融智能体
-   可审计和可调试的 AI 工作流
-   自定义应用


## SDKs 与 CLI

在您喜欢的语言中使用 Memvid：

| 包 | 安装 | 链接 |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI** | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli) |
| **Node.js SDK** | `npm install @memvid/sdk` | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk) |
| **Python SDK** | `pip install memvid-sdk` | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/) |
| **Rust** | `cargo add memvid-core` | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## 安装（Rust）

### 要求

-   **Rust 1.85.0+** — 从 [rustup.rs](https://rustup.rs) 安装

### 添加到项目

```toml
[dependencies]
memvid-core = "2.0"
```

### 功能标志

| 功能 | 描述 |
| -------------------- | ---------------------------------------------------------------- |
| `lex` | 使用 BM25 排序的全文搜索（Tantivy） |
| `pdf_extract` | 纯 Rust PDF 文本提取 |
| `vec` | 向量相似搜索（HNSW + 通过 ONNX 的本地文本嵌入） |
| `clip` | CLIP 视觉嵌入用于图像搜索 |
| `whisper` | 使用 Whisper 进行音频转录 |
| `api_embed` | 云 API 嵌入（OpenAI） |
| `temporal_track` | 自然语言日期解析（"last Tuesday"） |
| `parallel_segments` | 多线程摄取 |
| `encryption` | 基于密码的加密胶囊（.mv2e） |
| `symspell_cleanup` | 强大的 PDF 文本修复（修复 "emp lo yee" -> "employee"） |

按需启用功能：

```toml
[dependencies]
memvid-core = { version = "2.0", features = ["lex", "vec", "temporal_track"] }
```


## 快速开始

```rust
use memvid_core::{Memvid, PutOptions, SearchRequest};

fn main() -> memvid_core::Result<()> {
    // 创建新记忆文件
    let mut mem = Memvid::create("knowledge.mv2")?;

    // 添加带元数据的文档
    let opts = PutOptions::builder()
        .title("Meeting Notes")
        .uri("mv2://meetings/2024-01-15")
        .tag("project", "alpha")
        .build();
    mem.put_bytes_with_options(b"Q4 planning discussion...", opts)?;
    mem.commit()?;

    // 搜索
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

## 构建

克隆仓库：

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

以调试模式构建：

```bash
cargo build
```

以发布模式构建（优化）：

```bash
cargo build --release
```

使用特定功能构建：

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---

## 运行测试

运行所有测试：

```bash
cargo test
```

带输出运行测试：

```bash
cargo test -- --nocapture
```

运行特定测试：

```bash
cargo test test_name
```

仅运行集成测试：

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## 示例

`examples/` 目录包含可运行示例：

### 基本用法

演示创建、添加、搜索和时间线操作：

```bash
cargo run --example basic_usage
```

### PDF 提取

提取和搜索 PDF 文档（使用 "Attention Is All You Need" 论文）：

```bash
cargo run --example pdf_ingestion
```

### CLIP 可视化搜索

使用 CLIP 嵌入进行图像搜索（需要 `clip` 功能）：

```bash
cargo run --example clip_visual_search --features clip
```

### Whisper 转录

音频转录（需要 `whisper` 功能）：

```bash
cargo run --example test_whisper --features whisper -- /path/to/audio.mp3
```

**可用模型：**

| 模型 | 大小 | 速度 | 用例 |
| ---------------------- | ------ | ------- | ----------------------------------- |
| `whisper-small-en` | 244 MB | 最慢 | 最佳准确度（默认） |
| `whisper-tiny-en` | 75 MB | 快 | 平衡 |
| `whisper-tiny-en-q8k` | 19 MB | 最快 | 快速测试，资源受限 |

**模型选择：**

```bash
# 默认（FP32 small，最高准确度）
cargo run --example test_whisper --features whisper -- audio.mp3

# 小型量化（小 75%，更快）
MEMVID_WHISPER_MODEL=whisper-tiny-en-q8k cargo run --example test_whisper --features whisper -- audio.mp3
```

**可编程配置：**

```rust
use memvid_core::{WhisperConfig, WhisperTranscriber};

// 默认 FP32 small 模型
let config = WhisperConfig::default();

// 小型量化模型（更快，更小）
let config = WhisperConfig::with_quantization();

// 特定模型
let config = WhisperConfig::with_model("whisper-tiny-en-q8k");

let transcriber = WhisperTranscriber::new(&config)?;
let result = transcriber.transcribe_file("audio.mp3")?;
println!("{}", result.text);
```


## 文本嵌入模型

`vec` 功能包括使用 ONNX 模型的本地文本嵌入支持。在使用本地文本嵌入之前，您需要手动下载模型文件。

### 快速开始：BGE-small（推荐）

下载默认 BGE-small 模型（384 维，快速高效）：

```bash
mkdir -p ~/.cache/memvid/text-models

# 下载 ONNX 模型
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5.onnx

# 下载分词器
curl -L 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-small-en-v1.5_tokenizer.json
```

### 可用模型

| 模型 | 维度 | 大小 | 最适合 |
| ------------------------ | ---------- | ------ | --------------- |
| `bge-small-en-v1.5` | 384 | ~120MB | 默认，快速 |
| `bge-base-en-v1.5` | 768 | ~420MB | 更好的质量 |
| `nomic-embed-text-v1.5` | 768 | ~530MB | 多用途任务 |
| `gte-large` | 1024 | ~1.3GB | 最高质量 |

### 其他模型

**BGE-base**（768 维）：
```bash
curl -L 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/bge-base-en-v1.5.onnx
curl -L 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/bge-base-en-v1.5_tokenizer.json
```

**Nomic**（768 维）：
```bash
curl -L 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/nomic-embed-text-v1.5.onnx
curl -L 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/nomic-embed-text-v1.5_tokenizer.json
```

**GTE-large**（1024 维）：
```bash
curl -L 'https://huggingface.co/thenlper/gte-large/resolve/main/onnx/model.onnx' \
  -o ~/.cache/memvid/text-models/gte-large.onnx
curl -L 'https://huggingface.co/thenlper/gte-large/resolve/main/tokenizer.json' \
  -o ~/.cache/memvid/text-models/gte-large_tokenizer.json
```

### 在代码中使用

```rust
use memvid_core::text_embed::{LocalTextEmbedder, TextEmbedConfig};
use memvid_core::types::embedding::EmbeddingProvider;

// 使用默认模型（BGE-small）
let config = TextEmbedConfig::default();
let embedder = LocalTextEmbedder::new(config)?;

let embedding = embedder.embed_text("hello world")?;
assert_eq!(embedding.len(), 384);

// 使用不同模型
let config = TextEmbedConfig::bge_base();
let embedder = LocalTextEmbedder::new(config)?;
```

有关相似度计算和搜索排名的完整示例，请参见 `examples/text_embedding.rs`。

### 模型一致性

为防止意外地模型混合（例如，使用 OpenAI 嵌入查询 BGE-small 索引），您可以将 Memvid 实例显式绑定到特定模型名称：

```rust
// 将索引绑定到特定模型。
// 如果之前使用不同模型创建索引将返回错误。
mem.set_vec_model("bge-small-en-v1.5")?;
```

绑定是持久化的。一旦设置，将来尝试使用不同模型名称将快速失败并返回 `ModelMismatch` 错误。



## API 嵌入（OpenAI）

`api_embed` 功能使用 OpenAI 的 API 启用基于云的嵌入生成。

### 设置

设置您的 OpenAI API 密钥：

```bash
export OPENAI_API_KEY="sk-..."
```

### 用法

```rust
use memvid_core::api_embed::{OpenAIConfig, OpenAIEmbedder};
use memvid_core::types::embedding::EmbeddingProvider;

// 使用默认模型（text-embedding-3-small）
let config = OpenAIConfig::default();
let embedder = OpenAIEmbedder::new(config)?;

let embedding = embedder.embed_text("hello world")?;
assert_eq!(embedding.len(), 1536);

// 使用更高质量模型
let config = OpenAIConfig::large();  // text-embedding-3-large (3072 维)
let embedder = OpenAIEmbedder::new(config)?;
```

### 可用模型

| 模型 | 维度 | 最适合 |
| ------------------------ | ---------- | -------------------------- |
| `text-embedding-3-small` | 1536 | 默认，最快，最便宜 |
| `text-embedding-3-large` | 3072 | 最高质量 |
| `text-embedding-ada-002` | 1536 | 传统模型 |

有关完整示例，请参见 `examples/openai_embedding.rs`。



## 文件格式

所有内容都存储在单个 `.mv2` 文件中：

```
┌────────────────────────────┐
│ Header (4KB)               │  魔数，版本，容量
├────────────────────────────┤
│ Embedded WAL (1-64MB)      │  崩溃恢复
├────────────────────────────┤
│ Data Segments              │  压缩帧
├────────────────────────────┤
│ Lex Index                  │  Tantivy 全文
├────────────────────────────┤
│ Vec Index                  │  HNSW 向量
├────────────────────────────┤
│ Time Index                 │  时序排序
├────────────────────────────┤
│ TOC (Footer)               │  段偏移
└────────────────────────────┘
```

不会有 `.wal`、`.lock`、`.shm` 或附带文件。永远不会。

有关完整文件格式规范，请参见 [MV2_SPEC.md](MV2_SPEC.md)。



## 支持

有问题或反馈？
邮箱：contact@memvid.com

**点个 ⭐ 支持我们**

---

> **Memvid v1（基于 QR 码的记忆）已弃用**
>
> 如果您参考的是 QR 码，那么您正在使用过时信息。
>
> 参见：https://docs.memvid.com/memvid-v1-deprecation

---

## 许可证

Apache License 2.0 — 详细信息请参见 [LICENSE](LICENSE) 文件。
