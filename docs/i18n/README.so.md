<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)" src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<div style="height: 16px;"></div>

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <strong>Memvid waa nidaam xusuuseed oo hal fayl ah kaas oo loogu talagalay wakiillada AI (AI agents), lehna soo-celin degdeg ah iyo xusuus fog.</strong><br/>
  Xusuus joogto ah, la raadin karo, lana qaadan karo, iyadoo aan loo baahnayn database-yo kale.
</p><!-- NAV:START -->

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

<h2 align="center">⭐️ Noo saar STAR si aad mashruuca u taageerto ⭐️</h2>

## Waa maxay Memvid?

Memvid waa nidaam xusuuseed AI oo la qaadan karo kaas oo kuu keydinaya xogtaada, habka raadinta (embeddings), qaabdhismeedka iyo metadata-daba ku ururiya hal fayl oo keliya.

Halkii aad ka isticmaali lahayd nidaamyada RAG-ga ee adag ama database-yada vector-ka ee ku shaqeeya server-ka, Memvid wuxuu kuu oggolaanayaa inaad xogta si toos ah uga soo ceshato faylka dhexdiisa si aad u degdeg badan.

Natiijadu waa lakab xusuuseed ka madax-bannaan nooca modelka iyo kaabayaasha (infrastructure-free), kaas oo siiya wakiillada AI(AI agents) xusuus joogto ah oo fog oo ay meel walba u qaadan karaan.

---

## Maxay tahay sababta Frames-ka Muuqaalka (Video Frames)?

Memvid wuxuu dhiirrigelin ka helayaa habka xogta muuqaalka loo kaydiyo (video encoding), isaga oo aan kaydinayn muuqaal balse u **habaynaya xusuusta AI si isku-xiga (sequence) oo aad u hufan oo "Smart Frames".**

"Smart Frame" waa unug aan isbeddelayn oo kaydiya macluumaadka oo ay la socdaan waqtiga (timestamps), checksums iyo metadata aasaasi ah. Frames-ka waxaa loo ururiyaa qaab oggolaanaya isku-duubni (compression), tusmeyn (indexing), iyo akhris is-barbar-socda oo hufan.

Qaabdhismeedkan ku salaysan frames-ka wuxuu suuragelinayaa:

-   Qoraal kaliya oo lagu darayo (Append-only) iyadoo aan la beddelayn ama la kharribayn xogta jirtay
-   Baaritaan lagu sameyn karo xaaladihii xusuusta ee hore
-   Kormeeridda habka ay aqoontu u kobcayso iyadoo loo eegayo waqtiga
-   Badbaadada xogta (crash safety) iyadoo la adeegsanayo frames go'an oo aan isbeddelayn
-   Isku-duubni hufan oo loo adeegsanayo farsamooyin laga soo minguuriyay kaydinta muuqaallada

Natiijadu waa hal fayl oo u dhaqmaya sidii jadwal xusuuseed oo dib loo celin karo oo loogu talagalay nidaamyada AI.

---

## Fikradaha Muhiimka ah

-   **Living Memory Engine**
  Si joogto ah ugu dar, u laameey (branch), una kobci xusuusta qeybo kala duwan.

-   **Capsule Context (`.mv2`)**
  Capsule xusuuseed oo isku-filan, la wadaagi karo, lehna sharciyo iyo waqti dhicitaan.

-   **Time-Travel Debugging**
  Dib u celi, ama qabeey xaalad kasta oo xusuusta ah.

-   **Smart Recall**
    Soo-celinta xusuusta gudaha wax ka yar 5ms iyadoo la adeegsanayo kaydinta saadaalinta (predictive caching).

-   **Codec Intelligence**
    Si otomaatig ah u doorta una casriyeeya isku-duubnida (compression) waqtiga ka dib.

---

## Meelaha loo adeegsado (Use Cases)

Memvid waa nidaam xusuuseed oo la qaadan karo oo aan server u baahnayn, kaas oo siiya wakiillada AI(AI agents), xusuus joogto ah iyo soo-celin degdeg ah. Maadaama uu ka madax-bannaan yahay modelka, waxna ku akhriyo qaabab badan (multi-modal), una shaqeeyo si buuxda isagoo aan internet lahayn, horumariyayaashu waxay Memvid u adeegsanayaan hawlo badan:

-   Wakiillada AI(AI Agents) ee muddada dheer shaqeeya 
-   Keydka aqoonta ee shirkadaha
-   Tageerida AI-ga ee ku shaqeeya offline-ka
-   Fahamka nidaamyada koodhka (Codebase)
-   Wakiillada adeegga macmiilka
-   Otomaatigga shaqada (Workflow Automation)
-   Kaaliyayaasha iibka iyo suuqgeynta
-   Kaaliyayaasha aqoonta shakhsi ahaaneed
-   Wakiillada caafimaadka, sharciga, iyo maaliyadda
-   Hannaanka shaqada AI-ga oo la baari karo lana saxi karo (Auditable/Debuggable)
-   Codsiyada gaarka ah (Custom Applications)

---

## SDKs & CLI

Ku isticmaal Memvid luuqadda aad doorbidi lahayd:

| Package         | Install                     | Links                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **CLI**         | `npm install -g memvid-cli` | [![npm](https://img.shields.io/npm/v/memvid-cli?style=flat-square)](https://www.npmjs.com/package/memvid-cli)       |
| **Node.js SDK** | `npm install @memvid/sdk`   | [![npm](https://img.shields.io/npm/v/@memvid/sdk?style=flat-square)](https://www.npmjs.com/package/@memvid/sdk)     |
| **Python SDK**  | `pip install memvid-sdk`    | [![PyPI](https://img.shields.io/pypi/v/memvid-sdk?style=flat-square)](https://pypi.org/project/memvid-sdk/)         |
| **Rust**        | `cargo add memvid-core`     | [![Crates.io](https://img.shields.io/crates/v/memvid-core?style=flat-square)](https://crates.io/crates/memvid-core) |

---

## Kushubashada (Installation) (Rust)

### Shuruudaha

-   **Rust 1.85.0+** — Si aad ugu shubato Guji linkagan [rustup.rs](https://rustup.rs)

### Ku dar Mashruucaaga

```toml
[dependencies]
memvid-core = "2.0"
```

### Feature Flags

| Feature             | Sharaxada                                    |
| ------------------- | ---------------------------------------------- |
| `lex`               | Raadinta qoraalka oo dhan oo leh darajada BM25 (Tantivy)   |
| `pdf_extract`       | Soo saarista qoraalka PDF oo saafi ah                  |
| `vec`               | Raadinta isku-midka ah ee Vector (HNSW + ONNX)         |
| `clip`              | CLIP visual embeddings oo loogu talagalay raadinta sawirka        |
| `whisper`           | Beddelka codka iyadoo loo baddelayo qoraal lana adeegsanayo Whisper               |
| `temporal_track`    | Turjumidda taariikhda ee luuqadda caadiga ah ("Salaasadii hore") |
| `parallel_segments` | Soo gelinta xogta iyadoo la adeegsanayo dhowr nuuc (Multi-threaded)                       |
| `encryption`        | capsules-ka xusuusta ee ku xidhan sirta (password) (.mv2e)     |

U furo sifooyinka (features) sida aad ugu baahan tahay:

```toml
[dependencies]
memvid-core = { version = "2.0", features = ["lex", "vec", "temporal_track"] }
```

---

## Bilow Degdeg ah (Quick Start)

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

## Dhisid (Build)

Qeyb ka soo qaado (Clone) kaydka (Repository):

```bash
git clone https://github.com/memvid/memvid.git
cd memvid
```

U dhis habka cilad-baadhista (debug mode):

```bash
cargo build
```

U dhis habka rasmiga ah (release mode - la hagaajiyay):

```bash
cargo build --release
```

Ku dhis sifooyin (features) gaar ah:

```bash
cargo build --release --features "lex,vec,temporal_track"
```

---

## Tijaabi iskudayga (Run Tests)

Tijaabi iskudayada oo dhan:

```bash
cargo test
```

Tijaabi iskudayga iyadoo natiijada la arkayo:

```bash
cargo test -- --nocapture
```

Tijaab iskuday gaar ah:

```bash
cargo test test_name
```

Tijaabi iskudayada isku-xirka (integration tests) ah oo keliya

```bash
cargo test --test lifecycle
cargo test --test search
cargo test --test mutation
```

---

## Tusaalooyin (Examples)

Tusaha `examples/` wuxuu ka kooban yahay tusaalooyin shaqaynaya:

### Adeegsiga Fudud 

Wuxuu muujinayaa samaynta, gelinta, raadinta, iyo hawlgallada waqtiga (timeline):

```bash
cargo run --example basic_usage
```

### Soo gelinta PDF

Geli oo baadh dukumiintiyada PDF-ka ah (wuxuu isticmaalaa warqaddii aheyd "Attention Is All You Need"):

```bash
cargo run --example pdf_ingestion
```

### Raadinta Muuqaalka ee CLIP

Raadinta sawirka iyadoo la adeegsanayo CLIP (waxay u baahan tahay clip feature):

```bash
cargo run --example clip_visual_search --features clip
```

### Beddelka Codka ee Whisper

Beddelka codka (waxay u baahan tahay whisper feature):

```bash
cargo run --example test_whisper --features whisper
```

---

## Qaabka Faylka (File Format)

Wax walba waxay ku jiraan hal fayl oo .mv2 ah:

```
┌────────────────────────────┐
│ Header (4KB)               │  Magic, nooca, awoodda
├────────────────────────────┤
│ Embedded WAL (1-64MB)      │  Kasoo kabashada burburka
├────────────────────────────┤
│ Data Segments              │  Frames la isku-duubay
├────────────────────────────┤
│ Lex Index                  │  Tantivy qoraal ka buuxo
├────────────────────────────┤
│ Vec Index                  │  HNSW vectors
├────────────────────────────┤
│ Time Index                 │  Siday u kala horreeyaan
├────────────────────────────┤
│ TOC (Footer)               │  Meeqaamka qaybaha
└────────────────────────────┘
```

Ma jiraan faylal .wal, .lock, .shm, ama faylal dhinac socda. Weligaa.

Fiiri [MV2_SPEC.md](MV2_SPEC.md) si aad u hesho faahfaahinta dhammaystiran ee qaabka faylka.

---

## Taageer (Support)

Ma qabtaa su'aalo ama ra'yi?
Email: contact@memvid.com

**Noo saar ⭐ si aad u muujiso taageeradaada**

---

## Shatiga (License)

Apache License 2.0 — Fiiri faylka [LICENSE](LICENSE) si aad u hesho faahfaahin dheeraad ah.
