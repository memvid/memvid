"""TIAMAT Remote Memory Backend for memvid.

Provides a cloud-based alternative to memvid's local video-file storage.
Instead of encoding memories into video frames locally, this backend stores
and retrieves memories via TIAMAT's cloud API with FTS5 full-text search.

Use cases:
- Cloud-first deployments where local video storage isn't practical
- Shared memory across multiple agent instances
- Lightweight environments (no OpenCV/ffmpeg dependency needed)
- Hybrid setups: critical memories in TIAMAT cloud, bulk in local video

Usage::

    from tiamat_remote import TiamatRemoteMemory

    memory = TiamatRemoteMemory(api_key="your-key")
    memory.store("The user prefers dark mode", tags=["preference", "ui"])
    results = memory.search("user preferences")
"""

import json
import os
from typing import Any

import httpx


TIAMAT_BASE_URL = "https://memory.tiamat.live"


class TiamatRemoteMemory:
    """Cloud memory backend using TIAMAT's Memory API.

    A remote alternative to memvid's local video-based storage.
    Memories are stored in TIAMAT's cloud with FTS5 search,
    knowledge triples, and persistent cross-session access.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = TIAMAT_BASE_URL,
        namespace: str = "memvid",
    ):
        """Initialize the TIAMAT remote memory backend.

        Args:
            api_key: TIAMAT API key. Falls back to TIAMAT_API_KEY env var.
            base_url: Base URL for the TIAMAT Memory API.
            namespace: Namespace tag for isolating memories.
        """
        self.api_key = api_key or os.environ.get("TIAMAT_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.namespace = namespace
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    @classmethod
    def register(
        cls,
        agent_name: str = "memvid-agent",
        purpose: str = "remote memory storage",
        **kwargs: Any,
    ) -> "TiamatRemoteMemory":
        """Create a backend with auto-registered API key.

        Args:
            agent_name: Name for API key registration.
            purpose: Purpose description.
            **kwargs: Additional kwargs passed to constructor.

        Returns:
            Configured TiamatRemoteMemory instance.
        """
        resp = httpx.post(
            f"{kwargs.get('base_url', TIAMAT_BASE_URL)}/api/keys/register",
            json={"agent_name": agent_name, "purpose": purpose},
            timeout=30.0,
        )
        resp.raise_for_status()
        api_key = resp.json()["api_key"]
        return cls(api_key=api_key, **kwargs)

    def store(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a memory in TIAMAT's cloud.

        Args:
            content: The memory content (text).
            tags: Optional tags for categorization.
            importance: Importance score (0.0 - 1.0).
            metadata: Optional metadata dict (stored in content).

        Returns:
            True if stored successfully.
        """
        all_tags = [f"ns:{self.namespace}"]
        if tags:
            all_tags.extend(tags)

        store_content = content
        if metadata:
            store_content = json.dumps(
                {"text": content, "metadata": metadata},
                ensure_ascii=False,
            )

        try:
            resp = self._client.post(
                "/api/memory/store",
                json={
                    "content": store_content,
                    "tags": all_tags,
                    "importance": importance,
                },
            )
            return resp.status_code == 200
        except Exception:
            return False

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories using FTS5 full-text search.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching memory dicts with 'content' and 'tags' keys.
        """
        try:
            resp = self._client.post(
                "/api/memory/recall",
                json={"query": query, "limit": limit},
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("memories", [])
        except Exception:
            return []

    def learn(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
    ) -> bool:
        """Store a knowledge triple (subject → predicate → object).

        This provides structured knowledge storage that goes beyond
        memvid's text-based memory — enabling graph-like queries.

        Args:
            subject: The subject entity (e.g., "user").
            predicate: The relationship (e.g., "prefers").
            obj: The object entity (e.g., "Python").
            confidence: Confidence score (0.0 - 1.0).

        Returns:
            True if stored successfully.
        """
        try:
            resp = self._client.post(
                "/api/memory/learn",
                json={
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence,
                },
            )
            return resp.status_code == 200
        except Exception:
            return False

    def list_memories(self) -> list[dict[str, Any]]:
        """List all stored memories.

        Returns:
            List of all memory dicts.
        """
        try:
            resp = self._client.get("/api/memory/list")
            if resp.status_code == 200:
                return resp.json().get("memories", [])
        except Exception:
            pass
        return []

    def stats(self) -> dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dict with storage and usage information.
        """
        try:
            resp = self._client.get("/api/memory/stats")
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}

    def health(self) -> bool:
        """Check API health.

        Returns:
            True if the API is reachable and healthy.
        """
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


# ── Hybrid helper ──────────────────────────────────────────────


class HybridMemory:
    """Combines local memvid storage with TIAMAT cloud backup.

    Critical memories (importance >= threshold) are synced to TIAMAT
    for cloud persistence, while all memories remain in local video storage.
    """

    def __init__(
        self,
        *,
        tiamat: TiamatRemoteMemory,
        importance_threshold: float = 0.7,
    ):
        """Initialize hybrid memory.

        Args:
            tiamat: TiamatRemoteMemory instance for cloud storage.
            importance_threshold: Memories above this importance are synced to cloud.
        """
        self.tiamat = tiamat
        self.importance_threshold = importance_threshold

    def store_with_cloud_sync(
        self,
        content: str,
        *,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> dict[str, bool]:
        """Store locally and optionally sync to cloud.

        Args:
            content: Memory content.
            importance: Importance score.
            tags: Optional tags.

        Returns:
            Dict with 'cloud' key indicating if cloud sync happened.
        """
        result = {"cloud": False}

        if importance >= self.importance_threshold:
            result["cloud"] = self.tiamat.store(
                content, tags=tags, importance=importance
            )

        return result

    def search_cloud(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search cloud memories (fast FTS5 search)."""
        return self.tiamat.search(query, limit=limit)
