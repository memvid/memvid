## What to expect in v2

> **Early-access notice**  
> Memvid v1 is still experimental. The file format and API may change until we lock in a stable release.
> 
> **Memvid v2 – what's next**  
> - **Living-Memory Engine** – keep adding new data and let LLMs remember it across sessions.  
> - **Capsule Context** – shareable `.mv2` capsules, each with its own rules and expiry.  
> - **Time-Travel Debugging** – rewind or branch any chat to review or test.  
> - **Smart Recall** – local cache guesses what you’ll need and loads it in under 5 ms.  
> - **Codec Intelligence** – auto-tunes AV1 now and future codecs later, so files keep shrinking.  
> - **CLI & Dashboard** – simple tools for branching, analytics, and one-command cloud publish.  

Sneak peek of Memvid v2 - a living memory engine that can be used to chat with your knowledge base.
![Memvid v2 Preview](assets/mv2.png)


---

## Memvid v1



[![PyPI](https://img.shields.io/pypi/v/memvid)](https://pypi.org/project/memvid/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/olow304/memvid)](https://github.com/olow304/memvid)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Memvid - Turn millions of text chunks into a single, searchable video file

Memvid compresses an entire knowledge base into **MP4** files while keeping millisecond-level semantic search. Think of it as *SQLite for AI memory* portable, efficient, and self-contained. By encoding text as **QR codes in video frames**, we deliver **50-100×** smaller storage than vector databases with **zero infrastructure**.

---

## Why Video Compression Changes Everything 🚀

| What it enables | How video codecs make it possible |
|---------|-------------------|
| **50-100× smaller storage** | Modern video codecs compress repetitive visual patterns (QR codes) far better than raw embeddings |
| **Sub-100ms retrieval** | Direct frame seek via index → QR decode → your text. No server round-trips |
| **Zero infrastructure** | Just Python and MP4 files-no DB clusters, no Docker, no ops |
| **True portability** | Copy or stream `memory.mp4`-it works anywhere video plays |
| **Offline-first design** | After encoding, everything runs without internet |

---

## Under the Hood - Memvid v1 🔍

1. **Text → QR → Frame**  
   Each text chunk becomes a QR code, packed into video frames. Modern codecs excel at compressing these repetitive patterns.

2. **Smart indexing**  
   Embeddings map queries → frame numbers. One seek, one decode, millisecond results.

3. **Codec leverage**  
   30 years of video R&D means your text gets compressed better than any custom algorithm could achieve.

4. **Future-proof**  
   Next-gen codecs (AV1, H.266) automatically make your memories smaller and faster-no code changes needed.

---

## Installation
```bash
pip install memvid
# For PDF support
pip install memvid PyPDF2
```

## Quick Start
```python
from memvid import MemvidEncoder, MemvidChat

# Create video memory from text
chunks = ["NASA founded 1958", "Apollo 11 landed 1969", "ISS launched 1998"]
encoder = MemvidEncoder()
encoder.add_chunks(chunks)
encoder.build_video("space.mp4", "space_index.json")

# Chat with your memory (uses Google Gemini by default)
chat = MemvidChat("space.mp4", "space_index.json")
response = chat.chat("When did humans land on the moon?")
print(response)  # References Apollo 11 in 1969

# Or use local models with Ollama (no API key needed!)
chat = MemvidChat("space.mp4", "space_index.json", 
                  llm_provider='ollama', 
                  llm_model='llama3.2:3b')
```

## Real-World Examples

### Documentation Assistant
```python
from memvid import MemvidEncoder
import os

encoder = MemvidEncoder(chunk_size=512)

# Index all markdown files
for file in os.listdir("docs"):
    if file.endswith(".md"):
        with open(f"docs/{file}") as f:
            encoder.add_text(f.read(), metadata={"file": file})

encoder.build_video("docs.mp4", "docs_index.json")
```

### PDF Library Search
```python
# Index multiple PDFs
encoder = MemvidEncoder()
encoder.add_pdf("deep_learning.pdf")
encoder.add_pdf("machine_learning.pdf") 
encoder.build_video("ml_library.mp4", "ml_index.json")

# Semantic search across all books
from memvid import MemvidRetriever
retriever = MemvidRetriever("ml_library.mp4", "ml_index.json")
results = retriever.search("backpropagation", top_k=5)
```

### Interactive Web UI
```python
from memvid import MemvidInteractive

# Launch at http://localhost:7860
interactive = MemvidInteractive("knowledge.mp4", "index.json")
interactive.run()
```

## LLM Provider Support

Memvid supports multiple LLM providers for chat functionality:

### Supported Providers

| Provider | API Key Required | Best For | Setup |
|----------|-----------------|----------|-------|
| **Google (Gemini)** | ✅ Yes | Fast, cost-effective | Set `GOOGLE_API_KEY` |
| **OpenAI (GPT)** | ✅ Yes | High quality | Set `OPENAI_API_KEY` |
| **Anthropic (Claude)** | ✅ Yes | Long context | Set `ANTHROPIC_API_KEY` |
| **Ollama (Local)** | ❌ No | Privacy, offline | [Install Ollama](https://ollama.com) |

### Using Local Models with Ollama

Run AI models locally without API costs or privacy concerns:

```python
# Install Ollama first: https://ollama.com
# Pull a model: ollama pull llama3.2:3b

from memvid import MemvidChat

chat = MemvidChat(
    video_file="memory.mp4",
    index_file="index.json",
    llm_provider='ollama',
    llm_model='llama3.2:3b'  # or mistral, phi3, etc.
)

response = chat.chat("Your question")
```

See [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md) for detailed Ollama setup and usage.

### Switching Providers

```python
# Google Gemini (default)
chat = MemvidChat(..., llm_provider='google', llm_model='gemini-2.0-flash-exp')

# OpenAI
chat = MemvidChat(..., llm_provider='openai', llm_model='gpt-4o')

# Anthropic
chat = MemvidChat(..., llm_provider='anthropic', llm_model='claude-3-5-sonnet-20241022')

# Ollama (local)
chat = MemvidChat(..., llm_provider='ollama', llm_model='llama3.2:3b')
```

## Advanced Features

### Scale Optimization
```python
# Maximum compression for huge datasets
encoder.build_video(
    "compressed.mp4",
    "index.json", 
    fps=60,              # More frames/second
    frame_size=256,      # Smaller QR codes
    video_codec='h265',  # Better compression
    crf=28              # Quality tradeoff
)
```

### Custom Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
encoder = MemvidEncoder(embedding_model=model)
```

### Parallel Processing
```python
encoder = MemvidEncoder(n_workers=8)
encoder.add_chunks_parallel(million_chunks)
```

## CLI Usage
```bash
# Process documents
python examples/file_chat.py --input-dir /docs --provider openai

# Advanced codecs
python examples/file_chat.py --files doc.pdf --codec h265

# Load existing
python examples/file_chat.py --load-existing output/memory
```

## Performance

- **Indexing**: ~10K chunks/second on modern CPUs
- **Search**: <100ms for 1M chunks (includes decode)
- **Storage**: 100MB text → 1-2MB video
- **Memory**: Constant 500MB RAM regardless of size

## What's Coming in v2

- **Delta encoding**: Time-travel through knowledge versions
- **Streaming ingest**: Add to videos in real-time
- **Cloud dashboard**: Web UI with API management
- **Smart codecs**: Auto-select AV1/HEVC per content
- **GPU boost**: 100× faster bulk encoding

## Get Involved

Memvid is redefining AI memory. Join us:

- ⭐ Star on [GitHub](https://github.com/olow304/memvid)
- 🐛 Report issues or request features
- 🔧 Submit PRs (we review quickly!)
- 💬 Discuss video-based AI memory

