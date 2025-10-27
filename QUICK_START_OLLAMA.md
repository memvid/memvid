# Quick Start: Using Ollama with Memvid

## 1-Minute Setup

### Install Ollama

**Windows:**
```
Download from: https://ollama.com/download
```

**Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull a Model

```bash
ollama pull llama3.2:3b
```

### Install Python Package

```bash
pip install ollama
```

## Basic Usage

```python
from memvid import MemvidChat, MemvidEncoder

# Create knowledge base
encoder = MemvidEncoder()
encoder.add_text("Your content here...")
encoder.build_video("memory.mp4", "index.json")

# Chat with Ollama
chat = MemvidChat(
    video_file="memory.mp4",
    index_file="index.json",
    llm_provider='ollama',
    llm_model='llama3.2:3b'
)

# Ask questions
response = chat.chat("Your question?")
print(response)
```

## Run Example

```bash
python examples/ollama_chat.py
```

## Troubleshooting

**Problem**: "Failed to connect to Ollama"  
**Solution**: Run `ollama serve`

**Problem**: "Model not found"  
**Solution**: Run `ollama pull llama3.2:3b`

## Learn More

- Full guide: [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md)
- Complete changelog: [OLLAMA_CHANGELOG.md](OLLAMA_CHANGELOG.md)
- Example script: [examples/ollama_chat.py](examples/ollama_chat.py)

## Recommended Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| llama3.2:3b | 3GB | ⚡⚡⚡ | ⭐⭐⭐ |
| mistral:latest | 4GB | ⚡⚡ | ⭐⭐⭐⭐ |
| phi3:latest | 2GB | ⚡⚡⚡ | ⭐⭐⭐ |

## Why Ollama?

✅ No API keys required  
✅ Complete privacy (runs locally)  
✅ No usage costs  
✅ Works offline  
✅ Fast responses (no network latency)  

---

**Need help?** See [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md) for detailed instructions.

