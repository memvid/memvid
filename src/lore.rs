//! Lore Context backend for Memvid.
//!
//! Provides search and sync capabilities backed by a [Lore Context](https://github.com/Lore-Context/lore-context)
//! server, enabling cross-session and cross-device memory retrieval.
//!
//! # Usage
//!
//! ```no_run
//! # #[cfg(feature = "lore")]
//! # {
//! use memvid_core::lore::{LoreClient, LoreConfig};
//!
//! let client = LoreClient::new(LoreConfig {
//!     base_url: "http://localhost:3120".into(),
//!     api_key: Some("your-api-key".into()),
//!     project_id: None,
//!     top_k: 10,
//! });
//!
//! if let Ok(results) = client.search("meeting notes from last week") {
//!     for hit in &results.hits {
//!         println!("[{:.2}] {}: {}", hit.score, hit.title, hit.snippet);
//!     }
//! }
//! # }
//! ```

use crate::error::MemvidError;
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, MemvidError>;

/// Configuration for connecting to a Lore Context server.
#[derive(Debug, Clone)]
pub struct LoreConfig {
    /// Base URL of the Lore server (e.g., `http://localhost:3120`).
    pub base_url: String,
    /// Optional API key for authentication. Sets the `Authorization: Bearer` header.
    pub api_key: Option<String>,
    /// Optional project scope for memory operations.
    pub project_id: Option<String>,
    /// Default maximum number of results to return from search.
    pub top_k: usize,
}

impl Default for LoreConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:3120".into(),
            api_key: None,
            project_id: None,
            top_k: 10,
        }
    }
}

/// A search hit returned from Lore Context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoreHit {
    /// Relevance score (higher is better).
    pub score: f32,
    /// Title or identifier of the memory.
    pub title: String,
    /// Matching text snippet.
    pub snippet: String,
    /// Memory type (e.g., "fact", "observation", "decision").
    #[serde(default)]
    pub memory_type: String,
    /// When the memory was created.
    #[serde(default)]
    pub created_at: String,
}

/// Search results from Lore Context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoreSearchResult {
    /// The original query.
    pub query: String,
    /// Matching hits ordered by relevance.
    pub hits: Vec<LoreHit>,
    /// Total number of hits available.
    pub total: usize,
}

/// Memory record to write to Lore Context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoreMemory {
    /// The memory content.
    pub content: String,
    /// Memory type (e.g., "fact", "observation", "decision").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,
    /// Comma-separated key concepts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concepts: Option<String>,
}

/// Response from a memory write operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoreWriteResult {
    /// ID of the created memory.
    pub id: String,
    /// Whether the write succeeded.
    pub success: bool,
}

/// Client for interacting with a Lore Context server.
///
/// Uses blocking HTTP via `reqwest`. All methods are synchronous.
pub struct LoreClient {
    config: LoreConfig,
    client: reqwest::blocking::Client,
}

impl LoreClient {
    /// Create a new Lore client with the given configuration.
    #[must_use]
    pub fn new(config: LoreConfig) -> Self {
        Self {
            config,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Search Lore Context memories matching the given query.
    pub fn search(&self, query: &str) -> Result<LoreSearchResult> {
        let url = format!("{}/v1/memory/search", self.config.base_url);

        let mut body = serde_json::json!({
            "query": query,
            "top_k": self.config.top_k,
        });

        if let Some(ref project_id) = self.config.project_id {
            body["project_id"] = serde_json::json!(project_id);
        }

        let mut request = self.client.post(&url).json(&body);

        if let Some(ref api_key) = self.config.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request.send().map_err(|e| MemvidError::Io {
            source: std::io::Error::new(std::io::ErrorKind::ConnectionRefused, format!("Lore search failed: {e}")),
            path: None,
        })?;

        if !response.status().is_success() {
            return Err(MemvidError::Io {
                source: std::io::Error::other(format!("Lore search returned HTTP {}", response.status())),
                path: None,
            });
        }

        response.json::<LoreSearchResult>().map_err(|e| MemvidError::Io {
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Failed to parse Lore response: {e}")),
            path: None,
        })
    }

    /// Write a memory to Lore Context.
    pub fn write_memory(&self, memory: &LoreMemory) -> Result<LoreWriteResult> {
        let url = format!("{}/v1/memory/write", self.config.base_url);

        let mut body = serde_json::json!({
            "content": memory.content,
        });

        if let Some(ref memory_type) = memory.memory_type {
            body["memory_type"] = serde_json::json!(memory_type);
        }
        if let Some(ref concepts) = memory.concepts {
            body["concepts"] = serde_json::json!(concepts);
        }
        if let Some(ref project_id) = self.config.project_id {
            body["project_id"] = serde_json::json!(project_id);
        }

        let mut request = self.client.post(&url).json(&body);

        if let Some(ref api_key) = self.config.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request.send().map_err(|e| MemvidError::Io {
            source: std::io::Error::new(std::io::ErrorKind::ConnectionRefused, format!("Lore write failed: {e}")),
            path: None,
        })?;

        if !response.status().is_success() {
            return Err(MemvidError::Io {
                source: std::io::Error::other(format!("Lore write returned HTTP {}", response.status())),
                path: None,
            });
        }

        response.json::<LoreWriteResult>().map_err(|e| MemvidError::Io {
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Failed to parse Lore response: {e}")),
            path: None,
        })
    }

    /// Get the Lore configuration.
    #[must_use]
    pub fn config(&self) -> &LoreConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LoreConfig::default();
        assert_eq!(config.base_url, "http://localhost:3120");
        assert!(config.api_key.is_none());
        assert_eq!(config.top_k, 10);
    }

    #[test]
    fn test_client_creation() {
        let client = LoreClient::new(LoreConfig::default());
        assert_eq!(client.config().base_url, "http://localhost:3120");
    }
}
