<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)"
     src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<div style="height: 16px;"></div>

# মেমভিড-এ অবদান (বাংলা অনুবাদ)

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"></a>
</p>

<p align="center">
  <strong>মেমভিড হল এআই এজেন্টদের জন্য একটি একক-ফাইল মেমরি স্তর যার তাৎক্ষণিক পুনরুদ্ধার এবং দীর্ঘমেয়াদী মেমরি রয়েছে।</strong><br/>
  ডাটাবেস ছাড়াই স্থায়ী, সংস্করণযুক্ত এবং পোর্টেবল মেমরি।
</p>

<!-- NAV:START -->
<p align="center">
  <a href="https://www.memvid.com">ওয়েবসাইট</a>
  ·
  <a href="https://sandbox.memvid.com">স্যান্ডবক্সি চেষ্টা করে দেখুন</a>
  ·
  <a href="https://docs.memvid.com">ডক্স</a>
  ·
  <a href="https://github.com/memvid/memvid/discussions">আলোচনা</a>
  ·
  <a href="docs/i18n/translation_hub.md">Translations</a>
</p>
<!-- NAV:END -->

<p align="center">
  <a href="https://crates.io/crates/memvid-core"><img src="https://img.shields.io/crates/v/memvid-core?style=flat-square&logo=rust" alt="Crates.io" /></a>
  <a href="https://docs.rs/memvid-core"><img src="https://img.shields.io/docsrs/memvid-core?style=flat-square&logo=docs.rs" alt="docs.rs" /></a>
  <a href="https://github.com/memvid/memvid/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License" /></a>
</p>

<!-- BADGES:START -->
<p align="center">
  <a href="https://github.com/memvid/memvid/stargazers"><img src="https://img.shields.io/github/stars/memvid/memvid?style=flat-square&logo=github" alt="Stars" /></a>
  <a href="https://github.com/memvid/memvid/network/members"><img src="https://img.shields.io/github/forks/memvid/memvid?style=flat-square&logo=github" alt="Forks" /></a>
  <a href="https://github.com/memvid/memvid/issues"><img src="https://img.shields.io/github/issues/memvid/memvid?style=flat-square&logo=github" alt="Issues" /></a>
  <a href="https://discord.gg/2mynS7fcK7"><img src="https://img.shields.io/discord/1442910055233224745?style=flat-square&logo=discord&label=discord" alt="Discord" /></a>
</p>
<!-- BADGES:END -->

<h2 align="center">⭐️ প্রকল্পটি সমর্থন করার জন্য একটি তারকা দিন। ⭐️</h2>
</p>

## মেমভিড কী?

মেমভিড হল একটি পোর্টেবল এআই মেমরি সিস্টেম যা আপনার ডেটা, এম্বেডিং, অনুসন্ধান কাঠামো এবং মেটাডেটা একটি একক ফাইলে প্যাক করে।

জটিল RAG পাইপলাইন বা সার্ভার-ভিত্তিক ভেক্টর ডাটাবেস চালানোর পরিবর্তে, Memvid ফাইল থেকে সরাসরি দ্রুত ডেটা পুনরুদ্ধারে সহায়তা করে।

ফলাফল হল একটি মডেল-অজ্ঞেয়বাদী, অবকাঠামো-মুক্ত মেমোরি স্তর যা AI এজেন্টদের স্থায়ী, দীর্ঘমেয়াদী মেমোরি দেয় যা তারা যেকোনো জায়গায় নিতে পারে।
---

## ভিডিও ফ্রেম কেন?

মেমভিড ভিডিও এনকোডিং থেকে অনুপ্রেরণা নেয়, ভিডিও সংরক্ষণের জন্য নয়, বরং এআই মেমোরিকে অ্যাপেন্ড-ওনলি, স্মার্ট ফ্রেমের অতি-দক্ষ সিকোয়েন্স হিসেবে সংগঠিত করার জন্য।

একটি স্মার্ট ফ্রেম হল একটি অ-পরিবর্তনযোগ্য ইউনিট যা টাইমস্ট্যাম্প, চেকসাম এবং মৌলিক মেটাডেটা সহ বিষয়বস্তু সংরক্ষণ করে।
ফ্রেমগুলিকে এমনভাবে গোষ্ঠীভুক্ত করা হয় যা দক্ষ কম্প্রেশন, ইনডেক্সিং এবং সমান্তরাল পঠনের সুযোগ করে দেয়।

এই ফ্রেম-ভিত্তিক নকশাটি সক্ষম করে:

- বিদ্যমান ডেটা পরিবর্তন বা দূষিত না করে কেবল ডেটা যোগ করা
- পূর্ববর্তী মেমরি অবস্থা অনুসন্ধান করা
- জ্ঞান কীভাবে বিকশিত হয় তার টাইমলাইন-স্টাইল তদন্ত
- প্রতিশ্রুতিবদ্ধ, অপরিবর্তনীয় ফ্রেমের মাধ্যমে ক্র্যাশ সুরক্ষা
- ভিডিও এনকোডিং থেকে গৃহীত কৌশল ব্যবহার করে দক্ষ কম্প্রেশন।

ফলাফল হল একটি একক ফাইল যা AI সিস্টেমের জন্য একটি রিওয়াইন্ডেবল মেমরি টাইমলাইন হিসাবে কাজ করে।

---

## মূল ধারণা

- **লিভিং মেমোরি ইঞ্জিন**

একটি সেশনের সময় স্থায়ী মেমোরি যোগ করুন, শাখা করুন এবং বিকশিত করুন।

- **ক্যাপসুল রেফারেন্স (`.mv2`)**

নিয়ম এবং মেয়াদোত্তীর্ণতা সহ স্বয়ংসম্পূর্ণ, শেয়ারযোগ্য মেমোরি ক্যাপসুল।

- **টাইম-ট্রাভেল ডিবাগিং**

যেকোন মেমোরি স্টেট রিওয়াইন্ড, রিপ্লে বা শাখা করুন।

- **স্মার্ট রিকল**

প্রেডিক্টিভ ক্যাশিং সহ সাব-5ms স্থানীয় মেমোরি অ্যাক্সেস।

- **কোডেক ইন্টেলিজেন্স**

সময়ের সাথে সাথে স্বয়ংক্রিয়ভাবে কম্প্রেশন নির্বাচন এবং আপগ্রেড করে।

---

## ব্যবহারের ক্ষেত্রে

মেমভিড হল একটি পোর্টেবল, সার্ভারলেস মেমরি স্তর যা এআই এজেন্টদের স্থায়ী মেমরি এবং দ্রুত রিকল দেয়। যেহেতু এটি মডেল-অ্যাগনস্টিক, মাল্টি-মডেল এবং সম্পূর্ণ অফলাইনে কাজ করে, তাই ডেভেলপাররা বিভিন্ন বাস্তব-বিশ্ব অ্যাপ্লিকেশনে মেমভিড ব্যবহার করছে।

- দীর্ঘমেয়াদী এআই এজেন্ট
- এন্টারপ্রাইজ জ্ঞান ভিত্তি
- অফলাইন-প্রথম এআই সিস্টেম
- কোডবেস বোঝা
- গ্রাহক সহায়তা এজেন্ট
- ওয়ার্কফ্লো অটোমেশন
- বিক্রয় এবং বিপণন সহ-পাইলট
- ব্যক্তিগত জ্ঞান সহকারী
- চিকিৎসা, আইনি এবং আর্থিক এজেন্ট
- নিরীক্ষণযোগ্য এবং ডিবাগযোগ্য এআই কর্মপ্রবাহ
- ​​কাস্টম অ্যাপ্লিকেশন

---

## SDKs & CLI

আপনার পছন্দের ভাষায় Memvid ব্যবহার করুন:

| Package         | Install                     | Links                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI**         | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli)       |
| **Node.js SDK** | `npm install @memvid/sdk`   | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk)     |
| **Python SDK**  | `pip install memvid-sdk`    | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/)         |
| **Rust**        | `cargo add memvid-core`     | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## Installation (Rust)

### আবশ্যকতা

-   **Rust 1.85.0+** — Install from [rustup.rs](https://rustup.rs)

### আপনার প্রকল্পে যোগ করুন

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

## দ্রুত শুরু

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

## তৈরি করুন

রিপোজিটরিটি ক্লোন করুন:

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

ডিবাগ মোডে তৈরি করুন:

```bash
cargo build
```

রিলিজ মোডে তৈরি করুন (অপ্টিমাইজ করা):

```bash
cargo build --release
```

স্বতন্ত্র বৈশিষ্ট্য সহ তৈরি করুন:

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---
## পরীক্ষাগুলি চালান

সকল পরীক্ষা চালান:

```bash
cargo test
```

নিম্নলিখিত আউটপুট দিয়ে পরীক্ষাটি চালান:

```bash
cargo test -- --nocapture
```

একটি নির্দিষ্ট পরীক্ষা চালান:
```bash
cargo test test_name
```
শুধুমাত্র ইন্টিগ্রেশন পরীক্ষা চালান:

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## উদাহরণ

The `examples/` কার্যকরী ডিরেক্টরিগুলির উদাহরণ হল:

### মৌলিক ব্যবহার

এটি তৈরি, পুট, অনুসন্ধান এবং টাইমলাইন ক্রিয়াকলাপগুলি দেখায়:

```bash
cargo run --example basic_usage
```

### পিডিএফ ইনজেকশন

পিডিএফ ডকুমেন্টগুলি গ্রহণ করুন এবং অনুসন্ধান করুন ("Attention Is All You Need" পেপার ব্যবহার করে):

```bash
cargo run --example pdf_ingestion
```

### CLIP ভিজ্যুয়াল সার্চ

CLIP এম্বেডিং ব্যবহার করে ছবি সার্চ (`ক্লিপ` বৈশিষ্ট্য প্রয়োজন):

```bash
cargo run --example clip_visual_search --features clip
```

### হুইস্পার ট্রান্সক্রিপশন

অডিও ট্রান্সক্রিপশন (`হুইস্পার` বৈশিষ্ট্য প্রয়োজন):

```bash
cargo run --example test_whisper --features whisper
```

---

## ফাইল ফরম্যাট

সবকিছু একটি একক `.mv2` ফাইলের মধ্যে রয়েছে:

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

কোন `.wal`, `.lock`, `.shm`, অথবা সাইডকার ফাইল নেই। কখনও না।

সম্পূর্ণ ফাইল ফর্ম্যাট স্পেসিফিকেশনের জন্য [MV2_SPEC.md](MV2_SPEC.md) দেখুন।

---

## সহায়তা

আপনার কি কোন প্রশ্ন বা প্রতিক্রিয়া আছে?

ইমেল: contact@memvid.com

**সমর্থন দেখানোর জন্য ⭐ দিন**

---

## লাইসেন্স

Apache License 2.0 — আরও তথ্যের জন্য [LICENSE](LICENSE) ফাইলটি দেখুন।