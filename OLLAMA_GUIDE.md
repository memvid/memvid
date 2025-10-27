# Ollama Integration Guide

This guide explains how to use **Memvid** with **Ollama** to run large language models locally on your own hardware, without requiring API keys or internet connectivity after initial setup.

## Table of Contents

1. [What is Ollama?](#what-is-ollama)
2. [Why Use Ollama with Memvid?](#why-use-ollama-with-memvid)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Available Models](#available-models)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

---

## What is Ollama?

**Ollama** is a tool that makes it easy to run large language models (LLMs) locally on your computer. It supports various open-source models like:

- **Llama 3.2** - Meta's latest language model (various sizes)
- **Mistral** - High-performance 7B parameter model
- **Phi-3** - Microsoft's efficient small model
- **Gemma** - Google's open-source model
- **CodeLlama** - Specialized for code generation

Ollama handles model downloading, optimization, and serving through a simple API, making local LLM inference accessible to everyone.

---

## Why Use Ollama with Memvid?

### Benefits:

✅ **Privacy**: All data processing happens on your machine - no data sent to external APIs  
✅ **No API Costs**: Free to use after initial setup, no per-token charges  
✅ **Offline Capability**: Works without internet after models are downloaded  
✅ **Customization**: Full control over model selection and parameters  
✅ **Low Latency**: No network round trips for faster responses (on capable hardware)  
✅ **Experimentation**: Try different models without worrying about API costs  

### Use Cases:

- **Sensitive Data**: Process confidential documents locally
- **Development**: Test and iterate without API costs
- **Education**: Learn about LLMs with hands-on experience
- **Edge Computing**: Deploy AI applications without cloud dependencies
- **Cost Control**: Predictable compute costs vs. pay-per-use APIs

---

## Installation

### Step 1: Install Ollama

#### **Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### **Windows:**
Download and install from: https://ollama.com/download

### Step 2: Install Ollama Python Library

```bash
pip install ollama
```

Or if you're installing all Memvid dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
ollama --version
```

### Step 4: Start Ollama Server

The Ollama server usually starts automatically. To start it manually:

```bash
ollama serve
```

The server runs on `http://localhost:11434` by default.

---

## Quick Start

### 1. Pull a Model

First, download a model. Start with a small, fast model:

```bash
# Recommended for beginners (3GB)
ollama pull llama3.2:3b

# Or try other models
ollama pull mistral:latest        # Mistral 7B (4.1GB)
ollama pull phi3:latest           # Microsoft Phi-3 (2.3GB)
ollama pull llama3.2:latest       # Latest Llama 3.2 (7GB+)
```

### 2. List Available Models

Check which models you have installed:

```bash
ollama list
```

### 3. Test the Model

Try the model directly with Ollama:

```bash
ollama run llama3.2:3b "Hello, how are you?"
```

### 4. Use with Memvid

```python
from memvid import MemvidChat, MemvidEncoder

# Create a knowledge base
encoder = MemvidEncoder()
encoder.add_text("Your knowledge base content here...")
encoder.build_video("memory.mp4", "memory_index.json")

# Chat using Ollama (no API key needed!)
chat = MemvidChat(
    video_file="memory.mp4",
    index_file="memory_index.json",
    llm_provider='ollama',
    llm_model='llama3.2:3b'  # or any model you've pulled
)

# Start chatting
response = chat.chat("Your question here")
print(response)
```

---

## Available Models

### Recommended Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `llama3.2:3b` | 3GB | ⚡⚡⚡ | ⭐⭐⭐ | General use, fast responses |
| `phi3:latest` | 2.3GB | ⚡⚡⚡ | ⭐⭐⭐ | Lightweight, mobile-friendly |
| `mistral:latest` | 4.1GB | ⚡⚡ | ⭐⭐⭐⭐ | High quality, balanced |
| `llama3.2:latest` | 7GB+ | ⚡⚡ | ⭐⭐⭐⭐ | Best quality, slower |
| `codellama:latest` | 3.8GB | ⚡⚡ | ⭐⭐⭐⭐ | Code generation |
| `gemma:2b` | 1.4GB | ⚡⚡⚡ | ⭐⭐ | Ultra-fast, basic tasks |

### Specialized Models

- **Code**: `codellama`, `deepseek-coder`
- **Math**: `mathstral`
- **Vision**: `llava` (multimodal)
- **Multilingual**: `aya` (supports 100+ languages)

### Find More Models

Browse the full model library: https://ollama.com/library

---

## Usage Examples

### Example 1: Basic Chat

```python
from memvid import MemvidChat

# Initialize chat with Ollama
chat = MemvidChat(
    video_file="memory.mp4",
    index_file="memory_index.json",
    llm_provider='ollama',
    llm_model='llama3.2:3b'
)

# Single query
response = chat.chat("What is in the knowledge base?")
print(response)
```

### Example 2: Interactive Session

```python
from memvid import MemvidChat

chat = MemvidChat(
    video_file="memory.mp4",
    index_file="memory_index.json",
    llm_provider='ollama',
    llm_model='mistral:latest'
)

# Start interactive chat loop
chat.interactive_chat()
```

### Example 3: Streaming Responses

```python
from memvid import MemvidChat

chat = MemvidChat(
    video_file="memory.mp4",
    index_file="memory_index.json",
    llm_provider='ollama',
    llm_model='llama3.2:3b'
)

# Stream response token by token
response = chat.chat("Explain quantum computing", stream=True)
```

### Example 4: Custom Model Parameters

```python
from memvid import LLMClient

# Create client with custom parameters
client = LLMClient(
    provider='ollama',
    model='mistral:latest'
)

# Chat with custom generation parameters
response = client.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about AI."}
    ],
    temperature=0.7,      # Higher = more creative
    top_p=0.9,            # Nucleus sampling
    max_tokens=500        # Limit response length
)
```

### Example 5: Multiple Models

```python
from memvid import MemvidChat

# Use different models for different tasks
fast_chat = MemvidChat(
    video_file="memory.mp4",
    index_file="memory_index.json",
    llm_provider='ollama',
    llm_model='llama3.2:3b'  # Fast model for quick queries
)

quality_chat = MemvidChat(
    video_file="memory.mp4",
    index_file="memory_index.json",
    llm_provider='ollama',
    llm_model='mistral:latest'  # Better quality for complex queries
)
```

---

## Configuration

### Model Selection

Choose models based on your needs:

```python
# For speed (mobile, laptops)
llm_model='llama3.2:3b'

# For quality (desktop, GPU)
llm_model='mistral:latest'

# For specific tasks
llm_model='codellama:latest'  # Programming
llm_model='mathstral:latest'  # Mathematics
```

### Server Configuration

If running Ollama on a different host/port:

```python
from memvid.llm_client import OllamaProvider

# Custom Ollama server
provider = OllamaProvider(
    model='llama3.2:3b',
    base_url='http://192.168.1.100:11434'  # Custom host
)
```

### Generation Parameters

Fine-tune response generation:

```python
response = chat.chat(
    message="Your question",
    temperature=0.7,    # 0.0 = deterministic, 1.0+ = creative
    top_p=0.9,          # Nucleus sampling threshold
    top_k=40,           # Vocabulary limit
    max_tokens=1000,    # Maximum response length
    stop_sequences=['END', '\n\n\n']  # Stop generation triggers
)
```

---

## Troubleshooting

### Issue: "Failed to connect to Ollama"

**Solution:**
1. Check if Ollama is running:
   ```bash
   ollama list
   ```
2. Start the server manually:
   ```bash
   ollama serve
   ```
3. Verify the server is listening:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Issue: "Model not found"

**Solution:**
1. Check available models:
   ```bash
   ollama list
   ```
2. Pull the model you need:
   ```bash
   ollama pull llama3.2:3b
   ```

### Issue: Slow responses

**Solutions:**
- Use a smaller model (e.g., `llama3.2:3b` instead of `llama3.2:latest`)
- Reduce `max_tokens` parameter
- Check system resources (CPU/RAM usage)
- Consider GPU acceleration (see Performance Tips)

### Issue: Out of memory

**Solutions:**
- Use a smaller model:
  ```bash
  ollama pull gemma:2b  # Only 1.4GB
  ```
- Close other applications
- Increase system swap space

### Issue: "Ollama library not available"

**Solution:**
```bash
pip install ollama
```

---

## Performance Tips

### Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **RAM** | 8GB | 16GB | 32GB+ |
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **GPU** | None | NVIDIA 6GB+ | NVIDIA 16GB+ |
| **Storage** | 10GB free | 50GB free | 100GB+ free |

### GPU Acceleration

Ollama automatically uses GPU if available (NVIDIA CUDA, AMD ROCm, or Apple Metal).

Check GPU usage:
```bash
# NVIDIA
nvidia-smi

# AMD
rocm-smi

# Apple
Activity Monitor > Window > GPU History
```

### Model Size vs. Performance

- **Small models (2-3GB)**: Fast, good for real-time interactions
- **Medium models (4-7GB)**: Balanced quality/speed
- **Large models (13GB+)**: Best quality, requires powerful hardware

### Optimization Tips

1. **Preload models** before making requests:
   ```bash
   ollama run llama3.2:3b "test"
   ```

2. **Use appropriate context window**: Don't send unnecessarily long context

3. **Batch queries**: Reuse the same chat instance for multiple queries

4. **Monitor resources**:
   ```bash
   # Linux/Mac
   htop
   
   # Windows
   Task Manager
   ```

---

## Advanced Topics

### Running Multiple Models

Ollama can run multiple models simultaneously (if you have enough RAM):

```python
# Different models for different purposes
code_chat = MemvidChat(..., llm_model='codellama:latest')
general_chat = MemvidChat(..., llm_model='llama3.2:3b')
```

### Remote Ollama Server

Run Ollama on a powerful server and connect from clients:

```python
chat = MemvidChat(
    ...,
    llm_provider='ollama',
    llm_model='llama3.2:3b',
    # Note: You'll need to modify OllamaProvider instantiation
    # to accept base_url as a parameter through MemvidChat
)
```

### Model Management

```bash
# List downloaded models
ollama list

# Remove a model
ollama rm llama3.2:3b

# Update a model
ollama pull llama3.2:3b

# Show model info
ollama show llama3.2:3b
```

---

## Comparison: Ollama vs. Cloud APIs

| Feature | Ollama (Local) | Cloud APIs |
|---------|----------------|------------|
| **Privacy** | ✅ Complete | ⚠️ Shared data |
| **Cost** | ✅ Free after setup | ❌ Pay per use |
| **Latency** | ⚡ Low (no network) | 🌐 Network dependent |
| **Hardware** | ⚠️ Requires capable PC | ✅ No local requirements |
| **Model Selection** | ✅ Many open models | ✅ Proprietary models |
| **Quality** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Excellent |
| **Setup** | ⚠️ Manual | ✅ Simple |
| **Scalability** | ⚠️ Limited by hardware | ✅ Unlimited |

---

## Resources

- **Ollama Website**: https://ollama.com
- **Model Library**: https://ollama.com/library
- **Ollama GitHub**: https://github.com/ollama/ollama
- **Ollama Python SDK**: https://github.com/ollama/ollama-python
- **Memvid Documentation**: Check main README.md

---

## Contributing

Found an issue with Ollama integration? Want to add support for more features?

1. Check existing issues on GitHub
2. Submit bug reports or feature requests
3. Contribute code improvements via pull requests

---

## License

This guide is part of the Memvid project and follows the same license (MIT).

---

**Happy local LLM inference with Memvid + Ollama! 🚀**

