//! Python bindings for memvid-core using PyO3.
//!
//! Exposes the Memvid struct and key operations to Python.

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError, PyRuntimeError};
use std::path::PathBuf;
use std::collections::BTreeMap;

use memvid_core::{Memvid as CoreMemvid, PutOptions, FrameId, MemvidError};

/// Convert memvid-core errors to Python exceptions.
fn to_py_err(e: MemvidError) -> PyErr {
    match e {
        MemvidError::Io { source: _, path: _ } => PyIOError::new_err(format!("{}", e)),
        MemvidError::InvalidFrame { .. } => PyValueError::new_err(format!("{}", e)),
        _ => PyRuntimeError::new_err(format!("{}", e)),
    }
}

/// Python wrapper for memvid-core Memvid struct.
#[pyclass]
pub struct Memvid {
    inner: CoreMemvid,
}

#[pymethods]
impl Memvid {
    /// Create a new memory file at the given path.
    #[staticmethod]
    fn create(path: &str) -> PyResult<Self> {
        let inner = CoreMemvid::create(PathBuf::from(path)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Open an existing memory file.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let inner = CoreMemvid::open(PathBuf::from(path)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Open a memory file read-only.
    #[staticmethod]
    fn open_read_only(path: &str) -> PyResult<Self> {
        let inner = CoreMemvid::open_read_only(PathBuf::from(path)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Store content and return the frame ID.
    ///
    /// Args:
    ///     text: The text content to store.
    ///     title: Optional title for the document.
    ///     uri: Optional URI identifier.
    ///     tags: Optional list of tags.
    ///     labels: Optional list of labels.
    ///     timestamp: Optional Unix timestamp.
    ///
    /// Returns:
    ///     The frame ID (u64) that can be used with remove().
    #[pyo3(signature = (text, title=None, uri=None, tags=None, labels=None, timestamp=None))]
    fn put(
        &mut self,
        text: &str,
        title: Option<String>,
        uri: Option<String>,
        tags: Option<Vec<String>>,
        labels: Option<Vec<String>>,
        timestamp: Option<i64>,
    ) -> PyResult<u64> {
        let options = PutOptions {
            uri,
            title,
            search_text: Some(text.to_string()),
            tags: tags.unwrap_or_default(),
            labels: labels.unwrap_or_default(),
            timestamp,
            extra_metadata: BTreeMap::new(),
            ..Default::default()
        };
        
        let frame_id = self.inner.put_bytes_with_options(text.as_bytes(), options)
            .map_err(to_py_err)?;
        
        Ok(frame_id)
    }

    /// Remove a frame by its ID.
    ///
    /// This is a soft delete - the frame is marked as deleted and removed from indexes,
    /// but the data remains in the file until compaction.
    ///
    /// Args:
    ///     frame_id: The frame ID returned by put().
    ///
    /// Returns:
    ///     The WAL sequence number of the delete operation.
    fn remove(&mut self, frame_id: u64) -> PyResult<u64> {
        let seq = self.inner.delete_frame(frame_id as FrameId)
            .map_err(to_py_err)?;
        Ok(seq)
    }

    /// Commit pending changes to disk.
    fn commit(&mut self) -> PyResult<()> {
        self.inner.commit().map_err(to_py_err)
    }

    /// Seal the memory file (commit and close).
    fn seal(&mut self) -> PyResult<()> {
        self.inner.commit().map_err(to_py_err)
    }

    /// Get the number of frames in the memory.
    fn frame_count(&self) -> PyResult<u64> {
        let stats = self.inner.stats().map_err(to_py_err)?;
        Ok(stats.frame_count)
    }

    /// Check if the memory is read-only.
    fn is_read_only(&self) -> bool {
        self.inner.is_read_only()
    }
}

/// Python module initialization.
#[pymodule]
fn memvid_sdk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Memvid>()?;
    Ok(())
}
