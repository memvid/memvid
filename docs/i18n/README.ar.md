<!-- HEADER:START -->
<img width="2000" height="524" alt="Social Cover (9)"
     src="https://github.com/user-attachments/assets/cf66f045-c8be-494b-b696-b8d7e4fb709c" />
<!-- HEADER:END -->

<div style="height: 16px;"></div>

# المساهمة في Memvid (الترجمة العربية)

<p align="center">
    <a href="https://trendshift.io/repositories/17293" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17293" alt="memvid%2Fmemvid | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
<strong>Memvid هي طبقة ذاكرة مكونة من ملف واحد لوكلاء الذكاء الاصطناعي (AI Agents)، توفر استرجاعاً فورياً وذاكرة طويلة المدى.</strong>

ذاكرة دائمة، مؤرشفة، وقابلة للنقل، دون الحاجة إلى قواعد بيانات.

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

<h2 align="center">⭐️ اترك نجمة (STAR) لدعم المشروع ⭐️</h2>

---

## ما هو Memvid؟

Memvid هو نظام ذاكرة محمول للذكاء الاصطناعي يقوم بتغليف بياناتك، والمتجهات (Embeddings)، وهيكل البحث، والبيانات الوصفية في **ملف واحد فقط**.

بدلاً من تشغيل خطوط أنابيب RAG معقدة أو قواعد بيانات متجهة تعتمد على الخادم، يتيح Memvid استرجاعاً سريعاً للبيانات مباشرة من الملف.

النتيجة هي طبقة ذاكرة مستقلة عن النموذج (Model-agnostic) ولا تحتاج إلى بنية تحتية، مما يمنح وكلاء الذكاء الاصطناعي ذاكرة دائمة وطويلة المدى يمكنهم حملها في أي مكان.

---

## لماذا "إطارات الفيديو" (Video Frames)؟

يستلهم Memvid فكرته من ترميز الفيديو، ليس لتخزين الفيديو، بل **لتنظيم ذاكرة الذكاء الاصطناعي كمتسلسلة فائقة الكفاءة من "الإطارات الذكية" (Smart Frames) التي تُضاف باستمرار.**

"الإطار الذكي" هو وحدة غير قابلة للتغيير تخزن المحتوى مع الطوابع الزمنية، والتحقق من البيانات (Checksums)، والبيانات الوصفية الأساسية. يتم تجميع هذه الإطارات بطريقة تسمح بضغط البيانات وفهرستها والقراءة المتوازية بكفاءة عالية.

يسمح هذا التصميم بـ:

-   **إضافة البيانات فقط:** الكتابة دون تعديل أو إفساد البيانات الموجودة.
-   **الاستعلام عبر الزمن:** البحث في حالات الذاكرة السابقة.
-   **جدول زمني للمعرفة:** فحص كيفية تطور المعرفة بمرور الوقت.
-   **سلامة البيانات:** ضمان عدم فقدان البيانات عند التعطل بفضل الإطارات الثابتة.

---

## المفاهيم الأساسية

-   **محرك الذاكرة الحية:** إضافة وتطوير الذاكرة باستمرار عبر الجلسات.
-   **كبسولة السياق (`.mv2`):** كبسولات ذاكرة ذاتية الاحتواء وقابلة للمشاركة مع قواعد وصلاحية محددة.
-   **تصحيح السفر عبر الزمن:** إرجاع أو إعادة تشغيل أو تفريع أي حالة من حالات الذاكرة.
-   **الاستدعاء الذكي:** وصول محلي للذاكرة في أقل من 5 ملي ثانية مع ذاكرة تخزين مؤقت تنبؤية.
-   **ذكاء الترميز:** يختار ويحدث تقنيات الضغط تلقائياً بمرور الوقت.

---

## حالات الاستخدام

نظرًا لأن Memvid يعمل دون اتصال بالإنترنت ومستقل عن النماذج، فإنه يُستخدم في:

-   وكلاء الذكاء الاصطناعي طويلي الأمد.
-   قواعد المعرفة للمؤسسات.
-   أنظمة الذكاء الاصطناعي التي تعمل "بدون إنترنت أولاً".
-   فهم الأكواد البرمجية (Codebases).
-   المساعدين الشخصيين والأنظمة الطبية والقانونية والمالية.

---

## أدوات المطورين (SDKs)

| الحزمة                        | طريقة التثبيت               |
| ----------------------------- | --------------------------- |
| **واجهة السطر البرمجي (CLI)** | `npm install -g memvid-cli` |
| **Node.js SDK**               | `npm install @memvid/sdk`   |
| **Python SDK**                | `pip install memvid-sdk`    |
| **Rust**                      | `cargo add memvid-core`     |

---

## هيكل الملف

كل شيء يعيش داخل ملف واحد بصيغة `.mv2`:

```
┌────────────────────────────┐
│ Header (4KB)               │  العنوان: النسخة والقدرة الاستيعابية
├────────────────────────────┤
│ Embedded WAL (1-64MB)      │  سجل العمليات للتعافي من الأعطال
├────────────────────────────┤
│ Data Segments              │  أجزاء البيانات: الإطارات المضغوطة
├────────────────────────────┤
│ Lex Index                  │  الفهرس اللغوي: البحث النصي الكامل
├────────────────────────────┤
│ Vec Index                  │  الفهرس المتجهي: البحث بالمتجهات (HNSW)
├────────────────────────────┤
│ Time Index                 │  الفهرس الزمني: الترتيب الزمني
├────────────────────────────┤
│ TOC (Footer)               │  جدول المحتويات: مواقع الأجزاء
└────────────────────────────┘

```

---

## الدعم

هل لديك أسئلة؟
البريد الإلكتروني: contact@memvid.com

**لا تنسَ ترك ⭐ لدعم المشروع!**

---

## الترخيص

رخصة Apache 2.0 — راجع ملف [LICENSE](../../LICENSE) لمزيد من التفاصيل.
