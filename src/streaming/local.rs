//! Local file streaming source for testing without network.

#![allow(clippy::cast_possible_truncation)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::error::StreamingError;
use super::source::{StreamingResult, StreamingSource};

/// Local file streaming source for testing streaming functionality without network.
///
/// This source reads from a local file using the same range-based interface as
/// `HttpStreamingSource`, making it useful for testing and development.
pub struct LocalStreamingSource {
    path: PathBuf,
    file: Mutex<File>,
    size: u64,
}

impl LocalStreamingSource {
    /// Opens a local file as a streaming source.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .mv2 file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or its size cannot be determined.
    pub fn open<P: AsRef<Path>>(path: P) -> StreamingResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let size = file.metadata()?.len();

        Ok(Self {
            path,
            file: Mutex::new(file),
            size,
        })
    }

    /// Returns the path to the file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl StreamingSource for LocalStreamingSource {
    fn total_size(&self) -> StreamingResult<u64> {
        Ok(self.size)
    }

    fn read_range(&self, offset: u64, length: u64) -> StreamingResult<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }

        // Validate bounds
        if offset >= self.size {
            return Err(StreamingError::UnexpectedEndOfData {
                expected: length,
                actual: 0,
            });
        }

        // Clamp length to available data
        let available = self.size.saturating_sub(offset);
        let read_len = length.min(available);

        let mut file = self
            .file
            .lock()
            .map_err(|_| StreamingError::Io(std::io::Error::other("mutex poisoned")))?;

        file.seek(SeekFrom::Start(offset))?;

        let mut buffer = vec![0u8; read_len as usize];
        file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    fn source_id(&self) -> &str {
        self.path.to_str().unwrap_or("<invalid path>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_local_source_read_range() {
        let mut temp = NamedTempFile::new().unwrap();
        let data = b"Hello, World! This is test data for streaming.";
        temp.write_all(data).unwrap();
        temp.flush().unwrap();

        let source = LocalStreamingSource::open(temp.path()).unwrap();

        assert_eq!(source.total_size().unwrap(), data.len() as u64);

        // Read first 5 bytes
        let chunk = source.read_range(0, 5).unwrap();
        assert_eq!(&chunk, b"Hello");

        // Read from middle
        let chunk = source.read_range(7, 6).unwrap();
        assert_eq!(&chunk, b"World!");

        // Read to end
        let chunk = source.read_range(data.len() as u64 - 10, 10).unwrap();
        assert_eq!(&chunk, b"streaming.");
    }

    #[test]
    fn test_local_source_empty_read() {
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"test").unwrap();
        temp.flush().unwrap();

        let source = LocalStreamingSource::open(temp.path()).unwrap();

        let chunk = source.read_range(0, 0).unwrap();
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_local_source_clamps_to_file_size() {
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"short").unwrap();
        temp.flush().unwrap();

        let source = LocalStreamingSource::open(temp.path()).unwrap();

        // Request more bytes than available
        let chunk = source.read_range(2, 100).unwrap();
        assert_eq!(&chunk, b"ort");
    }
}
