//! HTTP streaming source implementation using reqwest.

use std::sync::OnceLock;
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{ACCEPT_RANGES, AUTHORIZATION, CONTENT_LENGTH, RANGE};

use super::error::StreamingError;
use super::source::{StreamingResult, StreamingSource};

/// Configuration for HTTP streaming.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Request timeout in seconds (default: 30).
    pub timeout_secs: u64,
    /// Maximum number of retry attempts (default: 3).
    pub max_retries: u32,
    /// Authentication configuration.
    pub auth: Option<HttpAuthConfig>,
    /// Custom User-Agent header.
    pub user_agent: Option<String>,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_retries: 3,
            auth: None,
            user_agent: None,
        }
    }
}

/// Authentication configuration for HTTP requests.
#[derive(Debug, Clone)]
pub enum HttpAuthConfig {
    /// Bearer token authentication (e.g., for S3 presigned URLs or API tokens).
    Bearer(String),
    /// Custom header-based authentication.
    CustomHeader {
        /// Header name (e.g., "X-API-Key")
        name: String,
        /// Header value
        value: String,
    },
}

/// HTTP streaming source that fetches data using range requests.
///
/// This source connects to an HTTP endpoint and uses the `Range` header to fetch
/// only the bytes needed, minimizing bandwidth usage.
pub struct HttpStreamingSource {
    url: String,
    client: Client,
    config: HttpConfig,
    content_length: OnceLock<u64>,
}

impl HttpStreamingSource {
    /// Creates a new HTTP streaming source from a URL with default configuration.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL of the .mv2 file to stream
    ///
    /// # Errors
    ///
    /// Returns an error if the server does not support range requests or is unreachable.
    pub fn from_url(url: impl Into<String>) -> StreamingResult<Self> {
        Self::with_config(url, HttpConfig::default())
    }

    /// Creates a new HTTP streaming source with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL of the .mv2 file to stream
    /// * `config` - Custom HTTP configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the server does not support range requests or is unreachable.
    pub fn with_config(url: impl Into<String>, config: HttpConfig) -> StreamingResult<Self> {
        let url = url.into();

        let mut client_builder = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .connect_timeout(Duration::from_secs(config.timeout_secs));

        if let Some(ref ua) = config.user_agent {
            client_builder = client_builder.user_agent(ua.clone());
        } else {
            client_builder =
                client_builder.user_agent(concat!("memvid-streaming/", env!("CARGO_PKG_VERSION")));
        }

        let client = client_builder.build()?;

        let source = Self {
            url,
            client,
            config,
            content_length: OnceLock::new(),
        };

        // Validate that the server supports range requests by doing a HEAD request
        source.validate_range_support()?;

        Ok(source)
    }

    /// Validates that the server supports range requests.
    fn validate_range_support(&self) -> StreamingResult<()> {
        let mut request = self.client.head(&self.url);
        request = self.apply_auth(request);

        let response = request.send()?;
        let status = response.status();

        if status.as_u16() == 404 {
            return Err(StreamingError::NotFound {
                url: self.url.clone(),
            });
        }

        if !status.is_success() {
            return Err(StreamingError::Http {
                status: status.as_u16(),
                message: status.canonical_reason().unwrap_or("Unknown error").into(),
            });
        }

        // Check Accept-Ranges header
        if let Some(accept_ranges) = response.headers().get(ACCEPT_RANGES) {
            if accept_ranges.to_str().map_or(true, |v| v == "none") {
                return Err(StreamingError::RangeNotSupported);
            }
        }

        // Cache content length
        if let Some(content_length) = response.headers().get(CONTENT_LENGTH) {
            if let Ok(len_str) = content_length.to_str() {
                if let Ok(len) = len_str.parse::<u64>() {
                    let _ = self.content_length.set(len);
                }
            }
        }

        Ok(())
    }

    /// Applies authentication headers to a request.
    fn apply_auth(
        &self,
        request: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        match &self.config.auth {
            Some(HttpAuthConfig::Bearer(token)) => {
                request.header(AUTHORIZATION, format!("Bearer {token}"))
            }
            Some(HttpAuthConfig::CustomHeader { name, value }) => request.header(name, value),
            None => request,
        }
    }

    /// Executes a range request with retry logic.
    fn fetch_range_with_retry(&self, offset: u64, length: u64) -> StreamingResult<Vec<u8>> {
        let mut last_error = None;
        let end = offset.saturating_add(length).saturating_sub(1);

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff: 100ms, 200ms, 400ms, ...
                let delay_ms = 100 * (1 << (attempt - 1));
                std::thread::sleep(Duration::from_millis(delay_ms));
            }

            match self.fetch_range_once(offset, end) {
                Ok(data) => return Ok(data),
                Err(e) => {
                    // Only retry on transient errors
                    let should_retry = matches!(
                        &e,
                        StreamingError::Network(_) | StreamingError::Timeout { .. }
                    );

                    if should_retry && attempt < self.config.max_retries {
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| StreamingError::Http {
            status: 0,
            message: "Max retries exceeded".into(),
        }))
    }

    /// Executes a single range request.
    fn fetch_range_once(&self, offset: u64, end: u64) -> StreamingResult<Vec<u8>> {
        let range_value = format!("bytes={offset}-{end}");

        let mut request = self.client.get(&self.url).header(RANGE, range_value);
        request = self.apply_auth(request);

        let response = request.send()?;
        let status = response.status();

        if status.as_u16() == 404 {
            return Err(StreamingError::NotFound {
                url: self.url.clone(),
            });
        }

        // Accept both 200 (full content) and 206 (partial content)
        if !status.is_success() && status.as_u16() != 206 {
            return Err(StreamingError::Http {
                status: status.as_u16(),
                message: status.canonical_reason().unwrap_or("Unknown error").into(),
            });
        }

        Ok(response.bytes()?.to_vec())
    }
}

impl StreamingSource for HttpStreamingSource {
    fn total_size(&self) -> StreamingResult<u64> {
        if let Some(&size) = self.content_length.get() {
            return Ok(size);
        }

        // Fetch via HEAD request
        let mut request = self.client.head(&self.url);
        request = self.apply_auth(request);

        let response = request.send()?;

        if !response.status().is_success() {
            return Err(StreamingError::Http {
                status: response.status().as_u16(),
                message: response
                    .status()
                    .canonical_reason()
                    .unwrap_or("Unknown error")
                    .into(),
            });
        }

        let content_length = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| StreamingError::Http {
                status: 0,
                message: "Missing Content-Length header".into(),
            })?;

        let _ = self.content_length.set(content_length);
        Ok(content_length)
    }

    fn read_range(&self, offset: u64, length: u64) -> StreamingResult<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }
        self.fetch_range_with_retry(offset, length)
    }

    fn source_id(&self) -> &str {
        &self.url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = HttpConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
        assert!(config.auth.is_none());
        assert!(config.user_agent.is_none());
    }
}
