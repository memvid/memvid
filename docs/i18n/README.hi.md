<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)" src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<div style="height: 16px;"></div>

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <strong>मेमविड AI एजेंट के लिए एक सिंगल-फाइल मेमोरी लेयर है जिसमें तुरंत रिट्रीवल और लॉन्ग-टर्म मेमोरी होती है।</strong><br/>
  बिना डेटाबेस के, स्थायी, वर्शन वाली और पोर्टेबल मेमोरी।
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

<h2 align="center">⭐️ प्रोजेक्ट को सपोर्ट करने के लिए एक स्टार दें। ⭐️</h2>
</p>

## What is Memvid?

मेमविड एक पोर्टेबल AI मेमोरी सिस्टम है जो आपके डेटा, एम्बेडिंग, सर्च स्ट्रक्चर और मेटाडेटा को एक ही फ़ाइल में पैक करता है।

जटिल RAG पाइपलाइन या सर्वर-आधारित वेक्टर डेटाबेस चलाने के बजाय, Memvid सीधे फ़ाइल से तेज़ी से डेटा रिट्रीव करने में मदद करता है।

इसका नतीजा एक मॉडल-एग्नोस्टिक, इंफ्रास्ट्रक्चर-फ्री मेमोरी लेयर है जो AI एजेंट को परमानेंट, लॉन्ग-टर्म मेमोरी देती है जिसे वे कहीं भी ले जा सकते हैं।

---

## वीडियो फ़्रेम क्यों?

मेमविड वीडियो एन्कोडिंग से प्रेरणा लेता है, वीडियो स्टोर करने के लिए नहीं, बल्कि**AI मेमोरी को स्मार्ट फ्रेम्स के अपेंड-ओनली, अल्ट्रा-एफ़िशिएंट सीक्वेंस के तौर पर ऑर्गनाइज़ करें।**

स्मार्ट फ्रेम एक ऐसा यूनिट है जिसे बदला नहीं जा सकता, जो टाइमस्टैम्प, चेकसम और बेसिक मेटाडेटा के साथ कंटेंट स्टोर करता है।
फ्रेम को इस तरह से ग्रुप किया जाता है जिससे कुशल कम्प्रेशन, इंडेक्सिंग और पैरेलल रीड संभव हो सके।

यह फ्रेम-आधारित डिज़ाइन इन चीज़ों को संभव बनाता है:

-   मौजूदा डेटा को संशोधित या दूषित किए बिना केवल डेटा जोड़ना
-   पिछली मेमोरी स्थितियों पर प्रश्न
-   ज्ञान कैसे विकसित होता है, इसकी टाइमलाइन-शैली में जांच
-   प्रतिबद्ध, अपरिवर्तनीय फ्रेम के माध्यम से क्रैश सुरक्षा
-   वीडियो एन्कोडिंग से अपनाई गई तकनीकों का उपयोग करके कुशल कम्प्रेशन।

इसका नतीजा एक सिंगल फ़ाइल होती है जो AI सिस्टम के लिए रिवाइंड करने लायक मेमोरी टाइमलाइन की तरह काम करती है।

---

## मुख्य अवधारणाएँ

-   **लिविंग मेमोरी इंजन**
    सेशन के दौरान लगातार मेमोरी को जोड़ें, ब्रांच करें और विकसित करें।

-   **कैप्सूल संदर्भ (`.mv2`)**
    नियमों और एक्सपायरी के साथ सेल्फ-कंटेन्ड, शेयर करने लायक मेमोरी कैप्सूल।

-   **टाइम-ट्रैवल डिबगिंग**
    किसी भी मेमोरी स्टेट को रिवाइंड, रिप्ले या ब्रांच करें।

-   **स्मार्ट रिकॉल**
    प्रेडिक्टिव कैशिंग के साथ सब-5ms लोकल मेमोरी एक्सेस।

-   **कोडेक इंटेलिजेंस**
    यह समय के साथ कम्प्रेशन को ऑटो-सेलेक्ट और अपग्रेड करता है।

---

## उपयोग के मामले

मेमविड एक पोर्टेबल, सर्वरलेस मेमोरी लेयर है जो AI एजेंट को परमानेंट मेमोरी और तेज़ रिकॉल देता है। क्योंकि यह मॉडल-एग्नोस्टिक, मल्टी-मॉडल है, और पूरी तरह से ऑफ़लाइन काम करता है, इसलिए डेवलपर्स मेमविड का इस्तेमाल कई तरह के रियल-वर्ल्ड एप्लीकेशन में कर रहे हैं।

- लंबे समय तक चलने वाले AI एजेंट
- एंटरप्राइज़ नॉलेज बेस
- ऑफ़लाइन-फ़र्स्ट AI सिस्टम
- कोडबेस को समझना
- कस्टमर सपोर्ट एजेंट
- वर्कफ़्लो ऑटोमेशन
- सेल्स और मार्केटिंग कोपायलट
- पर्सनल नॉलेज असिस्टेंट
- मेडिकल, लीगल और फाइनेंशियल एजेंट
- ऑडिट करने योग्य और डीबग करने योग्य AI वर्कफ़्लो
- कस्टम एप्लीकेशन

---

## SDKs & CLI

मेमविड को अपनी पसंदीदा भाषा में इस्तेमाल करें:

| Package         | Install                     | Links                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI**         | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli)       |
| **Node.js SDK** | `npm install @memvid/sdk`   | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk)     |
| **Python SDK**  | `pip install memvid-sdk`    | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/)         |
| **Rust**        | `cargo add memvid-core`     | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## Installation (Rust)

### आवश्यकताएं

-   **Rust 1.85.0+** — Install from [rustup.rs](https://rustup.rs)

### अपने प्रोजेक्ट में जोड़ें

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

## त्वरित प्रारंभ

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

## निर्माण

रिपॉजिटरी को क्लोन करें:

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

डीबग मोड में बिल्ड करें:

```bash
cargo build
```

रिलीज़ मोड में बिल्ड करें (ऑप्टिमाइज़्ड):

```bash
cargo build --release
```

विशिष्ट विशेषताओं के साथ बनाएं:

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---

## टेस्ट चलाएँ

सभी टेस्ट चलाएँ:

```bash
cargo test
```

आउटपुट के साथ टेस्ट चलाएँ:

```bash
cargo test -- --nocapture
```

एक विशिष्ट टेस्ट चलाएँ:

```bash
cargo test test_name
```

केवल इंटीग्रेशन टेस्ट चलाएँ:

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## उदाहरण

The `examples/` डायरेक्टरी में काम करने वाले उदाहरण हैं:

### बेसिक उपयोग

यह क्रिएट, पुट, सर्च और टाइमलाइन ऑपरेशन दिखाता है:

```bash
cargo run --example basic_usage
```

### PDF इन्जेक्शन

PDF डॉक्यूमेंट्स को इन्जेस्ट करें और सर्च करें ("अटेंशन इज़ ऑल यू नीड" पेपर का इस्तेमाल करता है):

```bash
cargo run --example pdf_ingestion
```

### CLIP विज़ुअल सर्च

CLIP एम्बेडिंग का इस्तेमाल करके इमेज सर्च (इसके लिए `clip` फ़ीचर ज़रूरी है):

```bash
cargo run --example clip_visual_search --features clip
```

### व्हिस्पर ट्रांसक्रिप्शन

ऑडियो ट्रांसक्रिप्शन (`whisper` फीचर ज़रूरी है):

```bash
cargo run --example test_whisper --features whisper
```

---

## फ़ाइल फ़ॉर्मेट

सब कुछ एक ही `.mv2` फ़ाइल में होता है:

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

कोई `.wal`, `.lock`, `.shm`, या साइडकार फ़ाइल नहीं। कभी नहीं।

पूरे फ़ाइल फ़ॉर्मेट स्पेसिफ़िकेशन के लिए [MV2_SPEC.md](MV2_SPEC.md) देखें।

---

## सपोर्ट

क्या आपके कोई सवाल या फीडबैक हैं?
Email: contact@memvid.com

**सपोर्ट दिखाने के लिए ⭐ दें**

---

## लाइसेंस

Apache License 2.0 — ज़्यादा जानकारी के लिए [LICENSE](LICENSE) फ़ाइल देखें।