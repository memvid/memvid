//! Error types for streaming operations.

use std::borrow::Cow;

use thiserror::Error;

/// Errors that can occur during streaming operations.
#[derive(Debug, Error)]
pub enum StreamingError {
    /// HTTP error response from the server.
    #[error("HTTP error: {status} - {message}")]
    Http {
        /// HTTP status code
        status: u16,
        /// Error message or response body
        message: String,
    },

    /// Server does not support HTTP range requests.
    #[error("Server does not support range requests (missing Accept-Ranges: bytes header)")]
    RangeNotSupported,

    /// Network error during HTTP request.
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Request timed out.
    #[error("Request timed out after {timeout_secs}s")]
    Timeout {
        /// Configured timeout in seconds
        timeout_secs: u64,
    },

    /// Resource not found (HTTP 404).
    #[error("Resource not found: {url}")]
    NotFound {
        /// URL that was not found
        url: String,
    },

    /// I/O error when reading from local source.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid header in the .mv2 file.
    #[error("Invalid header: {reason}")]
    InvalidHeader {
        /// Reason the header is invalid
        reason: Cow<'static, str>,
    },

    /// Invalid footer in the .mv2 file.
    #[error("Invalid footer: {reason}")]
    InvalidFooter {
        /// Reason the footer is invalid
        reason: Cow<'static, str>,
    },

    /// Invalid TOC in the .mv2 file.
    #[error("Invalid TOC: {reason}")]
    InvalidToc {
        /// Reason the TOC is invalid
        reason: Cow<'static, str>,
    },

    /// Checksum mismatch during validation.
    #[error("Checksum mismatch while validating {context}")]
    ChecksumMismatch {
        /// What was being validated
        context: &'static str,
    },

    /// Frame not found.
    #[error("Frame {frame_id} not found")]
    FrameNotFound {
        /// ID of the missing frame
        frame_id: u64,
    },

    /// Frame not found by URI.
    #[error("Frame with URI '{uri}' not found")]
    FrameNotFoundByUri {
        /// URI that was not found
        uri: String,
    },

    /// Lexical search index not available.
    #[error("Lexical search index not available in this file")]
    LexIndexNotAvailable,

    /// Vector search index not available.
    #[error("Vector search index not available in this file")]
    VecIndexNotAvailable,

    /// Feature not enabled at compile time.
    #[error("Feature '{feature}' is not enabled")]
    FeatureNotEnabled {
        /// Name of the disabled feature
        feature: &'static str,
    },

    /// Bincode decode error.
    #[error("Deserialization error: {0}")]
    Decode(#[from] bincode::error::DecodeError),

    /// File is encrypted and streaming is not supported for encrypted files.
    #[error("Encrypted files are not supported for streaming")]
    EncryptedFile,

    /// The data source returned fewer bytes than expected.
    #[error("Unexpected end of data: expected {expected} bytes, got {actual}")]
    UnexpectedEndOfData {
        /// Expected number of bytes
        expected: u64,
        /// Actual number of bytes received
        actual: u64,
    },
}
