//! API-based embedding providers (OpenAI, Anthropic, etc.)
//!
//! This module provides cloud API embedding generation, enabling semantic search
//! using external embedding services. Requires the `api_embed` feature.
//!
//! # Example
//!
//! ```ignore
//! use memvid_core::api_embed::{OpenAIConfig, OpenAIEmbedder};
//! use memvid_core::types::embedding::EmbeddingProvider;
//!
//! // Requires OPENAI_API_KEY environment variable
//! let config = OpenAIConfig::default();
//! let embedder = OpenAIEmbedder::new(config)?;
//!
//! let embedding = embedder.embed_text("Hello, world!")?;
//! println!("Embedding dimension: {}", embedding.len());
//! ```

use crate::error::{MemvidError, Result};
use crate::types::embedding::EmbeddingProvider;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ============================================================================
// OpenAI Models Registry
// ============================================================================

/// OpenAI embedding model information
#[derive(Debug, Clone)]
pub struct OpenAIModelInfo {
    /// Model identifier (e.g., "text-embedding-3-small")
    pub name: &'static str,
    /// Output embedding dimension
    pub dimension: usize,
    /// Maximum input tokens
    pub max_tokens: usize,
    /// Maximum texts per batch request
    pub max_batch_size: usize,
    /// Whether this is the default model
    pub is_default: bool,
}

/// Available OpenAI embedding models
pub static OPENAI_MODELS: &[OpenAIModelInfo] = &[
    OpenAIModelInfo {
        name: "text-embedding-3-small",
        dimension: 1536,
        max_tokens: 8191,
        max_batch_size: 2048,
        is_default: true,
    },
    OpenAIModelInfo {
        name: "text-embedding-3-large",
        dimension: 3072,
        max_tokens: 8191,
        max_batch_size: 2048,
        is_default: false,
    },
    OpenAIModelInfo {
        name: "text-embedding-ada-002",
        dimension: 1536,
        max_tokens: 8191,
        max_batch_size: 2048,
        is_default: false,
    },
];

/// Get model info by name, defaults to text-embedding-3-small
#[must_use]
pub fn get_openai_model_info(name: &str) -> &'static OpenAIModelInfo {
    OPENAI_MODELS
        .iter()
        .find(|m| m.name == name)
        .unwrap_or_else(|| OPENAI_MODELS.iter().find(|m| m.is_default).unwrap())
}

/// Get the default model info
#[must_use]
pub fn default_openai_model_info() -> &'static OpenAIModelInfo {
    OPENAI_MODELS.iter().find(|m| m.is_default).unwrap()
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for OpenAI embedding provider
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// Model name (e.g., "text-embedding-3-small")
    pub model: String,
    /// Environment variable name for API key (default: "OPENAI_API_KEY")
    pub api_key_env: String,
    /// Custom API base URL (for Azure OpenAI, proxies, etc.)
    /// Default: "https://api.openai.com/v1"
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries on rate limit (429) errors
    pub max_retries: u32,
    /// Initial backoff in milliseconds for exponential retry
    pub initial_backoff_ms: u64,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            model: "text-embedding-3-small".to_string(),
            api_key_env: "OPENAI_API_KEY".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            timeout_secs: 30,
            max_retries: 3,
            initial_backoff_ms: 1000,
        }
    }
}

impl OpenAIConfig {
    /// Create config for text-embedding-3-small (default, fastest)
    #[must_use]
    pub fn small() -> Self {
        Self::default()
    }

    /// Create config for text-embedding-3-large (highest quality)
    #[must_use]
    pub fn large() -> Self {
        Self {
            model: "text-embedding-3-large".to_string(),
            ..Default::default()
        }
    }

    /// Create config for text-embedding-ada-002 (legacy)
    #[must_use]
    pub fn ada() -> Self {
        Self {
            model: "text-embedding-ada-002".to_string(),
            ..Default::default()
        }
    }

    /// Set custom base URL (for Azure OpenAI or proxies)
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set custom API key environment variable name
    #[must_use]
    pub fn with_api_key_env(mut self, env_var: impl Into<String>) -> Self {
        self.api_key_env = env_var.into();
        self
    }

    /// Set request timeout
    #[must_use]
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

// ============================================================================
// API Request/Response Types
// ============================================================================

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    encoding_format: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[allow(dead_code)]
    usage: Usage,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

#[derive(Deserialize)]
struct Usage {
    #[allow(dead_code)]
    prompt_tokens: usize,
    #[allow(dead_code)]
    total_tokens: usize,
}

#[derive(Deserialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Deserialize)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

// ============================================================================
// OpenAI Embedder
// ============================================================================

/// OpenAI embedding provider
///
/// Generates embeddings using OpenAI's embedding API. Requires the `OPENAI_API_KEY`
/// environment variable to be set (or a custom env var via config).
///
/// # Example
///
/// ```ignore
/// use memvid_core::api_embed::{OpenAIConfig, OpenAIEmbedder};
/// use memvid_core::types::embedding::EmbeddingProvider;
///
/// let embedder = OpenAIEmbedder::new(OpenAIConfig::default())?;
/// let embedding = embedder.embed_text("Hello, world!")?;
/// ```
pub struct OpenAIEmbedder {
    config: OpenAIConfig,
    model_info: &'static OpenAIModelInfo,
    client: Client,
    api_key: String,
}

impl OpenAIEmbedder {
    /// Create a new OpenAI embedder
    ///
    /// Reads the API key from the environment variable specified in config.
    /// Returns an error if the API key is not set.
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        let api_key =
            std::env::var(&config.api_key_env).map_err(|_| MemvidError::EmbeddingFailed {
                reason: format!(
                    "API key not found. Set the {} environment variable.",
                    config.api_key_env
                )
                .into(),
            })?;

        if api_key.is_empty() {
            return Err(MemvidError::EmbeddingFailed {
                reason: format!("{} environment variable is empty", config.api_key_env).into(),
            });
        }

        let model_info = get_openai_model_info(&config.model);

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| MemvidError::EmbeddingFailed {
                reason: format!("Failed to create HTTP client: {}", e).into(),
            })?;

        tracing::info!(
            model = %model_info.name,
            dimension = model_info.dimension,
            "OpenAI embedder initialized"
        );

        Ok(Self {
            config,
            model_info,
            client,
            api_key,
        })
    }

    /// Get model info
    #[must_use]
    pub fn model_info(&self) -> &'static OpenAIModelInfo {
        self.model_info
    }

    /// Make an embedding request with retry logic
    fn request_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.config.base_url);

        let request_body = EmbeddingRequest {
            model: &self.config.model,
            input: texts.to_vec(),
            encoding_format: "float",
        };

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key)).map_err(|_| {
                MemvidError::EmbeddingFailed {
                    reason: "Invalid API key format".into(),
                }
            })?,
        );

        let mut backoff_ms = self.config.initial_backoff_ms;
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                tracing::warn!(
                    attempt = attempt,
                    backoff_ms = backoff_ms,
                    "Retrying OpenAI request after rate limit"
                );
                std::thread::sleep(Duration::from_millis(backoff_ms));
                backoff_ms *= 2; // Exponential backoff
            }

            let response = self
                .client
                .post(&url)
                .headers(headers.clone())
                .json(&request_body)
                .send();

            match response {
                Ok(resp) => {
                    let status = resp.status();

                    if status.is_success() {
                        let embedding_response: EmbeddingResponse =
                            resp.json().map_err(|e| MemvidError::EmbeddingFailed {
                                reason: format!("Failed to parse response: {}", e).into(),
                            })?;

                        // Sort by index to ensure correct order
                        let mut data = embedding_response.data;
                        data.sort_by_key(|d| d.index);

                        let embeddings: Vec<Vec<f32>> =
                            data.into_iter().map(|d| d.embedding).collect();

                        tracing::debug!(
                            texts = texts.len(),
                            dimension = embeddings.first().map(|e| e.len()).unwrap_or(0),
                            "Generated OpenAI embeddings"
                        );

                        return Ok(embeddings);
                    }

                    // Handle rate limiting
                    if status.as_u16() == 429 {
                        last_error = Some(MemvidError::EmbeddingFailed {
                            reason: "Rate limit exceeded".into(),
                        });
                        continue;
                    }

                    // Parse error response
                    let error_text = resp.text().unwrap_or_default();
                    let error_msg =
                        if let Ok(api_error) = serde_json::from_str::<ApiError>(&error_text) {
                            format!(
                                "OpenAI API error ({}): {}",
                                api_error.error.error_type.unwrap_or_default(),
                                api_error.error.message
                            )
                        } else {
                            format!("OpenAI API error ({}): {}", status, error_text)
                        };

                    return Err(MemvidError::EmbeddingFailed {
                        reason: error_msg.into(),
                    });
                }
                Err(e) => {
                    // Network error - might be transient
                    last_error = Some(MemvidError::EmbeddingFailed {
                        reason: format!("Request failed: {}", e).into(),
                    });

                    if e.is_timeout() || e.is_connect() {
                        continue; // Retry on timeout or connection errors
                    }

                    return Err(last_error.unwrap());
                }
            }
        }

        // All retries exhausted
        Err(last_error.unwrap_or_else(|| MemvidError::EmbeddingFailed {
            reason: "Max retries exceeded".into(),
        }))
    }
}

impl std::fmt::Debug for OpenAIEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbedder")
            .field("config", &self.config)
            .field("model_info", &self.model_info)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

// ============================================================================
// EmbeddingProvider Implementation
// ============================================================================

impl EmbeddingProvider for OpenAIEmbedder {
    fn kind(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        self.model_info.name
    }

    fn dimension(&self) -> usize {
        self.model_info.dimension
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.request_embeddings(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| MemvidError::EmbeddingFailed {
                reason: "No embedding returned".into(),
            })
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Split into chunks respecting API batch size limit
        let max_batch = self.model_info.max_batch_size;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(max_batch) {
            let chunk_embeddings = self.request_embeddings(chunk)?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    fn is_ready(&self) -> bool {
        // We have an API key, so we're ready
        !self.api_key.is_empty()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry() {
        assert_eq!(OPENAI_MODELS.len(), 3);

        let default = default_openai_model_info();
        assert_eq!(default.name, "text-embedding-3-small");
        assert_eq!(default.dimension, 1536);
        assert!(default.is_default);
    }

    #[test]
    fn test_get_model_info() {
        let small = get_openai_model_info("text-embedding-3-small");
        assert_eq!(small.dimension, 1536);

        let large = get_openai_model_info("text-embedding-3-large");
        assert_eq!(large.dimension, 3072);

        let ada = get_openai_model_info("text-embedding-ada-002");
        assert_eq!(ada.dimension, 1536);

        // Unknown model should return default
        let unknown = get_openai_model_info("unknown-model");
        assert_eq!(unknown.name, "text-embedding-3-small");
    }

    #[test]
    fn test_config_defaults() {
        let config = OpenAIConfig::default();
        assert_eq!(config.model, "text-embedding-3-small");
        assert_eq!(config.api_key_env, "OPENAI_API_KEY");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_config_builders() {
        let small = OpenAIConfig::small();
        assert_eq!(small.model, "text-embedding-3-small");

        let large = OpenAIConfig::large();
        assert_eq!(large.model, "text-embedding-3-large");

        let ada = OpenAIConfig::ada();
        assert_eq!(ada.model, "text-embedding-ada-002");
    }

    #[test]
    fn test_config_with_methods() {
        let config = OpenAIConfig::default()
            .with_base_url("https://custom.api.com")
            .with_api_key_env("MY_API_KEY")
            .with_timeout(60);

        assert_eq!(config.base_url, "https://custom.api.com");
        assert_eq!(config.api_key_env, "MY_API_KEY");
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn test_embedder_requires_api_key() {
        // Use a custom env var name that doesn't exist
        let config = OpenAIConfig::default().with_api_key_env("NONEXISTENT_API_KEY_12345");
        let result = OpenAIEmbedder::new(config);
        assert!(result.is_err());

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("NONEXISTENT_API_KEY_12345"));
    }

    #[test]
    fn test_embedder_validates_config() {
        // Test that config with custom env var works correctly
        let config = OpenAIConfig::default()
            .with_api_key_env("CUSTOM_KEY_VAR")
            .with_base_url("https://custom.openai.com/v1");

        assert_eq!(config.api_key_env, "CUSTOM_KEY_VAR");
        assert_eq!(config.base_url, "https://custom.openai.com/v1");

        // Creating embedder should fail since CUSTOM_KEY_VAR is not set
        let result = OpenAIEmbedder::new(config);
        assert!(result.is_err());
    }

    /// Integration test - requires OPENAI_API_KEY to be set
    /// Run with: cargo test --features api_embed test_openai_integration -- --ignored
    #[test]
    #[ignore]
    fn test_openai_integration() {
        let config = OpenAIConfig::default();
        let embedder = OpenAIEmbedder::new(config).expect("Failed to create embedder");

        // Test single embedding
        let embedding = embedder
            .embed_text("Hello, world!")
            .expect("Failed to embed text");
        assert_eq!(embedding.len(), 1536);

        // Test batch embedding
        let texts = vec!["Hello", "World", "Test"];
        let embeddings = embedder.embed_batch(&texts).expect("Failed to batch embed");
        assert_eq!(embeddings.len(), 3);
        assert!(embeddings.iter().all(|e| e.len() == 1536));

        // Test EmbeddingProvider trait
        assert_eq!(embedder.kind(), "openai");
        assert_eq!(embedder.model(), "text-embedding-3-small");
        assert_eq!(embedder.dimension(), 1536);
        assert!(embedder.is_ready());
    }

    /// Integration test with text-embedding-3-large
    #[test]
    #[ignore]
    fn test_openai_large_integration() {
        let config = OpenAIConfig::large();
        let embedder = OpenAIEmbedder::new(config).expect("Failed to create embedder");

        let embedding = embedder
            .embed_text("Test with large model")
            .expect("Failed to embed text");
        assert_eq!(embedding.len(), 3072);
    }
}
