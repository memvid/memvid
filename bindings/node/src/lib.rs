//! Node.js bindings for memvid-core using NAPI-RS.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::path::PathBuf;
use std::collections::BTreeMap;
use std::sync::Mutex;

use memvid_core::{Memvid as CoreMemvid, PutOptions, FrameId};

/// Options for the put operation.
#[napi(object)]
pub struct PutInput {
    pub text: String,
    pub title: Option<String>,
    pub uri: Option<String>,
    pub tags: Option<Vec<String>>,
    pub labels: Option<Vec<String>>,
    pub timestamp: Option<i64>,
}

/// Node.js wrapper for memvid-core Memvid struct.
#[napi]
pub struct Memvid {
    inner: Mutex<CoreMemvid>,
}

#[napi]
impl Memvid {
    /// Create a new memory file at the given path.
    #[napi(factory)]
    pub fn create(path: String) -> Result<Self> {
        let inner = CoreMemvid::create(PathBuf::from(path))
            .map_err(|e| Error::from_reason(format!("{}", e)))?;
        Ok(Self { inner: Mutex::new(inner) })
    }

    /// Open an existing memory file.
    #[napi(factory)]
    pub fn open(path: String) -> Result<Self> {
        let inner = CoreMemvid::open(PathBuf::from(path))
            .map_err(|e| Error::from_reason(format!("{}", e)))?;
        Ok(Self { inner: Mutex::new(inner) })
    }

    /// Open a memory file read-only.
    #[napi(factory)]
    pub fn open_read_only(path: String) -> Result<Self> {
        let inner = CoreMemvid::open_read_only(PathBuf::from(path))
            .map_err(|e| Error::from_reason(format!("{}", e)))?;
        Ok(Self { inner: Mutex::new(inner) })
    }

    /// Store content and return the frame ID.
    ///
    /// @param input - The content to store with optional metadata.
    /// @returns The frame ID that can be used with remove().
    #[napi]
    pub fn put(&self, input: PutInput) -> Result<i64> {
        let mut inner = self.inner.lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        let options = PutOptions {
            uri: input.uri,
            title: input.title,
            search_text: Some(input.text.clone()),
            tags: input.tags.unwrap_or_default(),
            labels: input.labels.unwrap_or_default(),
            timestamp: input.timestamp,
            extra_metadata: BTreeMap::new(),
            ..Default::default()
        };

        let frame_id = inner.put_bytes_with_options(input.text.as_bytes(), options)
            .map_err(|e| Error::from_reason(format!("{}", e)))?;

        Ok(frame_id as i64)
    }

    /// Remove a frame by its ID.
    ///
    /// This is a soft delete - the frame is marked as deleted and removed from indexes.
    ///
    /// @param frameId - The frame ID returned by put().
    /// @returns The WAL sequence number of the delete operation.
    #[napi]
    pub fn remove(&self, frame_id: i64) -> Result<i64> {
        let mut inner = self.inner.lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        let seq = inner.delete_frame(frame_id as FrameId)
            .map_err(|e| Error::from_reason(format!("{}", e)))?;

        Ok(seq as i64)
    }

    /// Commit pending changes to disk.
    #[napi]
    pub fn commit(&self) -> Result<()> {
        let mut inner = self.inner.lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;
        inner.commit().map_err(|e| Error::from_reason(format!("{}", e)))
    }

    /// Seal the memory file (commit and close).
    #[napi]
    pub fn seal(&self) -> Result<()> {
        let mut inner = self.inner.lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;
        inner.commit().map_err(|e| Error::from_reason(format!("{}", e)))
    }

    /// Get the number of frames in the memory.
    #[napi]
    pub fn frame_count(&self) -> Result<i64> {
        let inner = self.inner.lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;
        let stats = inner.stats().map_err(|e| Error::from_reason(format!("{}", e)))?;
        Ok(stats.frame_count as i64)
    }

    /// Check if the memory is read-only.
    #[napi]
    pub fn is_read_only(&self) -> Result<bool> {
        let inner = self.inner.lock()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;
        Ok(inner.is_read_only())
    }
}
