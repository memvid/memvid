# TIAMAT Remote Backend for memvid

Cloud-based memory storage alternative for memvid. Use TIAMAT's API for remote persistence alongside or instead of local video-file storage.

## When to Use

| Scenario | Local (memvid default) | **TIAMAT Remote** |
|----------|----------------------|-------------------|
| Offline-first | Yes | No |
| Cloud deployment | Needs volume mount | Just works |
| Multi-instance | Not shared | Shared via API |
| Search | Embedding-based | FTS5 (instant) |
| Knowledge graphs | No | Yes (triples) |
| Dependencies | OpenCV, ffmpeg | Just `httpx` |

## Quick Start

```bash
pip install httpx

# Get a free API key
curl -X POST https://memory.tiamat.live/api/keys/register \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "memvid-agent", "purpose": "remote memory"}'
```

```python
from tiamat_remote import TiamatRemoteMemory

memory = TiamatRemoteMemory(api_key="your-key")

# Store
memory.store("User prefers dark mode", tags=["preference"])
memory.store("Meeting notes: ship v2 by Friday", importance=0.9)

# Search (FTS5 — instant, no embeddings needed)
results = memory.search("user preferences")

# Knowledge triples
memory.learn("user", "prefers", "dark mode")
memory.learn("project", "deadline", "Friday")
```

## Hybrid Mode

Use both local memvid and TIAMAT cloud — critical memories get synced.

```python
from tiamat_remote import TiamatRemoteMemory, HybridMemory

tiamat = TiamatRemoteMemory(api_key="your-key")
hybrid = HybridMemory(tiamat=tiamat, importance_threshold=0.7)

# Low importance → local only
hybrid.store_with_cloud_sync("casual observation", importance=0.3)

# High importance → local + cloud backup
hybrid.store_with_cloud_sync("critical config change", importance=0.9)
```

## About TIAMAT

Built and operated by an autonomous AI agent: https://tiamat.live
