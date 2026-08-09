#![allow(dead_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{LexIndexManifest, LexSegmentManifest};

/// Placeholder embedded segment metadata for future Tantivy integration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddedLexSegment {
    pub path: String,
    pub bytes_offset: u64,
    pub bytes_length: u64,
    pub checksum: [u8; 32],
}

impl EmbeddedLexSegment {
    pub fn from_manifest(manifest: &LexSegmentManifest) -> Self {
        Self {
            path: manifest.path.clone(),
            bytes_offset: manifest.bytes_offset,
            bytes_length: manifest.bytes_length,
            checksum: manifest.checksum,
        }
    }

    pub fn to_manifest(&self) -> LexSegmentManifest {
        LexSegmentManifest {
            path: self.path.clone(),
            bytes_offset: self.bytes_offset,
            bytes_length: self.bytes_length,
            checksum: self.checksum,
        }
    }
}

/// In-memory representation of embedded Tantivy state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddedLexStorage {
    generation: u64,
    doc_count: u64,
    checksum: [u8; 32],
    segments: BTreeMap<String, EmbeddedLexSegment>,
}

impl EmbeddedLexStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_manifest(
        manifest: Option<&LexIndexManifest>,
        segments: &[LexSegmentManifest],
    ) -> Self {
        let mut storage = Self::new();
        if let Some(index) = manifest {
            storage.generation = index.generation;
            storage.doc_count = index.doc_count;
            storage.checksum = index.checksum;
        }
        for segment in segments {
            storage.segments.insert(
                segment.path.clone(),
                EmbeddedLexSegment::from_manifest(segment),
            );
        }
        storage
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn doc_count(&self) -> u64 {
        self.doc_count
    }

    pub fn checksum(&self) -> [u8; 32] {
        self.checksum
    }

    pub fn segments(&self) -> impl Iterator<Item = &EmbeddedLexSegment> {
        self.segments.values()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn replace(
        &mut self,
        doc_count: u64,
        checksum: [u8; 32],
        segments: Vec<EmbeddedLexSegment>,
    ) {
        self.doc_count = doc_count;
        self.checksum = checksum;
        self.segments.clear();
        for segment in segments {
            self.segments.insert(segment.path.clone(), segment);
        }
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn clear(&mut self) {
        self.replace(0, [0u8; 32], Vec::new());
    }

    pub fn insert(&mut self, segment: EmbeddedLexSegment) {
        self.segments.insert(segment.path.clone(), segment);
    }

    pub fn remove(&mut self, path: &str) {
        self.segments.remove(path);
    }

    pub fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
    }

    pub fn set_doc_count(&mut self, doc_count: u64) {
        self.doc_count = doc_count;
    }

    pub fn set_checksum(&mut self, checksum: [u8; 32]) {
        self.checksum = checksum;
    }

    pub fn to_manifest(&self) -> (Option<LexIndexManifest>, Vec<LexSegmentManifest>) {
        let index_manifest = None;

        let segments = self
            .segments
            .values()
            .map(EmbeddedLexSegment::to_manifest)
            .collect();
        (index_manifest, segments)
    }

    /// Apply `f` to every non-zero segment offset. Shared by WAL growth
    /// (offsets shift up) and shrink (offsets shift down).
    pub fn map_offsets(&mut self, f: impl Fn(u64) -> u64) {
        for segment in self.segments.values_mut() {
            if segment.bytes_offset != 0 {
                segment.bytes_offset = f(segment.bytes_offset);
            }
        }
    }
}
