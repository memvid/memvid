//! Trait defining the interface for streaming data sources.

use super::error::StreamingError;

/// Result type for streaming operations.
pub type StreamingResult<T> = std::result::Result<T, StreamingError>;

/// Trait for streaming data sources that support random-access reads.
///
/// Implementations must support reading arbitrary byte ranges from the source.
/// This enables efficient access to `.mv2` files hosted on CDNs without
/// downloading the entire file.
pub trait StreamingSource: Send + Sync {
    /// Returns the total size of the data source in bytes.
    ///
    /// This is typically fetched via a HEAD request for HTTP sources.
    fn total_size(&self) -> StreamingResult<u64>;

    /// Reads a range of bytes from the data source.
    ///
    /// # Arguments
    ///
    /// * `offset` - The byte offset to start reading from
    /// * `length` - The number of bytes to read
    ///
    /// # Returns
    ///
    /// A vector containing the requested bytes. The vector length should equal
    /// `length` unless the read extends past the end of the source.
    fn read_range(&self, offset: u64, length: u64) -> StreamingResult<Vec<u8>>;

    /// Returns a unique identifier for this source.
    ///
    /// For HTTP sources, this is typically the URL. For local sources, the file path.
    fn source_id(&self) -> &str;
}
