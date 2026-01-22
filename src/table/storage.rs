// Safe unwrap/expect: JSON value access with fallback defaults.
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! MV2 storage integration for extracted tables.
//!
//! This module handles storing and retrieving tables from MV2 files
//! using the existing frame and track infrastructure.

use std::collections::BTreeMap;

use serde_json::json;

use super::types::{ExtractedTable, TableQuality, TableSummary};
use crate::VecEmbedder;
use crate::error::{MemvidError, Result};
use crate::memvid::Memvid;
use crate::types::embedding_identity::{
    EmbeddingIdentity, MEMVID_EMBEDDING_DIMENSION_KEY, MEMVID_EMBEDDING_MODEL_KEY,
    MEMVID_EMBEDDING_NORMALIZED_KEY, MEMVID_EMBEDDING_PROVIDER_KEY,
};
use crate::types::{FrameId, PutOptions};

/// Track name used for table frames.
pub const TABLE_TRACK: &str = "tables";

/// Kind value for table metadata frames.
pub const TABLE_META_KIND: &str = "table_meta";

/// Kind value for table row frames.
pub const TABLE_ROW_KIND: &str = "table_row";

/// Store an extracted table in the MV2 file.
///
/// Creates two types of frames:
/// 1. A `table_meta` frame containing table metadata and structure
/// 2. Multiple `table_row` frames containing individual row data
///
/// # Arguments
/// * `mem` - The Memvid instance to store in
/// * `table` - The extracted table to store
/// * `embed_rows` - Whether to generate embeddings for row frames
///
/// # Returns
/// A tuple of (meta_frame_id, row_frame_ids)
pub fn store_table(
    mem: &mut Memvid,
    table: &ExtractedTable,
    embed_rows: bool,
) -> Result<(FrameId, Vec<FrameId>)> {
    store_table_impl(mem, table, embed_rows, None, None)
}

/// Store an extracted table in the MV2 file, embedding rows when an embedder is provided.
///
/// This is an ingestion-helper API for frontends (CLI / bindings). memvid-core does not ship
/// with a built-in text embedding runtime; callers must provide an embedder if they want row
/// embeddings (for semantic search).
pub fn store_table_with_embedder(
    mem: &mut Memvid,
    table: &ExtractedTable,
    embed_rows: bool,
    embedder: Option<&dyn VecEmbedder>,
    embedding_identity: Option<&EmbeddingIdentity>,
) -> Result<(FrameId, Vec<FrameId>)> {
    store_table_impl(mem, table, embed_rows, embedder, embedding_identity)
}

fn store_table_impl(
    mem: &mut Memvid,
    table: &ExtractedTable,
    embed_rows: bool,
    embedder: Option<&dyn VecEmbedder>,
    embedding_identity: Option<&EmbeddingIdentity>,
) -> Result<(FrameId, Vec<FrameId>)> {
    let table_id = &table.table_id;

    // 1. Create table_meta frame
    let meta_payload = serde_json::to_vec(&json!({
        "table_id": table_id,
        "source_file": table.source_file,
        "source_uri": table.source_uri,
        "page_start": table.page_start,
        "page_end": table.page_end,
        "headers": table.headers,
        "n_rows": table.n_rows,
        "n_cols": table.n_cols,
        "quality": table.quality.to_string(),
        "detection_mode": table.detection_mode.to_string(),
        "confidence_score": table.confidence_score,
        "warnings": table.warnings,
        "extraction_ms": table.extraction_ms,
    }))
    .map_err(|e| MemvidError::TableExtraction {
        reason: format!("failed to serialize table metadata: {e}"),
    })?;

    let mut meta_extra: BTreeMap<String, String> = BTreeMap::new();
    meta_extra.insert("table_id".to_string(), table_id.clone());
    meta_extra.insert("n_rows".to_string(), table.n_rows.to_string());
    meta_extra.insert("n_cols".to_string(), table.n_cols.to_string());
    meta_extra.insert("page_start".to_string(), table.page_start.to_string());
    meta_extra.insert("page_end".to_string(), table.page_end.to_string());
    meta_extra.insert("quality".to_string(), table.quality.to_string());
    meta_extra.insert(
        "detection_mode".to_string(),
        table.detection_mode.to_string(),
    );

    // Serialize headers for searchability
    if let Ok(headers_json) = serde_json::to_string(&table.headers) {
        meta_extra.insert("headers_json".to_string(), headers_json);
    }

    let meta_options = PutOptions {
        timestamp: None,
        track: Some(TABLE_TRACK.to_string()),
        kind: Some(TABLE_META_KIND.to_string()),
        uri: Some(format!("mv2://tables/{}", table_id)),
        title: Some(format!(
            "Table from {} (pages {}-{})",
            table.source_file, table.page_start, table.page_end
        )),
        metadata: None,
        search_text: Some(table.to_search_text()),
        tags: vec![
            "table".to_string(),
            table.source_file.clone(),
            format!("{}_quality", table.quality),
        ],
        labels: vec![format!("{}_detected", table.detection_mode)],
        extra_metadata: meta_extra,
        enable_embedding: false, // Don't embed metadata frame
        auto_tag: false,
        extract_dates: false,
        extract_triplets: false, // Table metadata doesn't need triplet extraction
        parent_id: None,
        role: crate::FrameRole::default(),
        no_raw: false,
        source_path: None,
        dedup: false,
        instant_index: false,    // Tables are batch operations, commit at end
        extraction_budget_ms: 0, // No budget for table metadata
    };

    let meta_frame_id = mem.next_frame_id();
    mem.put_bytes_with_options(&meta_payload, meta_options)?;

    // 2. Create table_row frames
    let mut row_frame_ids = Vec::with_capacity(table.rows.len());

    for row in &table.rows {
        // Skip header rows for storage (info is in headers field)
        if row.is_header_row {
            continue;
        }

        // Build cell map: header -> value
        let cell_map: serde_json::Map<String, serde_json::Value> = table
            .headers
            .iter()
            .enumerate()
            .filter_map(|(i, header)| {
                row.cells
                    .get(i)
                    .map(|cell| (header.clone(), serde_json::Value::String(cell.text.clone())))
            })
            .collect();

        let row_payload = serde_json::to_vec(&json!({
            "table_id": table_id,
            "row_index": row.row_index,
            "page": row.page,
            "cells": cell_map,
        }))
        .map_err(|e| MemvidError::TableExtraction {
            reason: format!("failed to serialize row data: {e}"),
        })?;

        // Generate searchable text from row
        let search_text: String = row
            .cells
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let mut row_extra: BTreeMap<String, String> = BTreeMap::new();
        row_extra.insert("table_id".to_string(), table_id.clone());
        row_extra.insert("row_index".to_string(), row.row_index.to_string());
        row_extra.insert("page".to_string(), row.page.to_string());
        row_extra.insert("parent_frame".to_string(), meta_frame_id.to_string());

        let mut row_options = PutOptions {
            timestamp: None,
            track: Some(TABLE_TRACK.to_string()),
            kind: Some(TABLE_ROW_KIND.to_string()),
            uri: Some(format!("mv2://tables/{}/row/{}", table_id, row.row_index)),
            title: None,
            metadata: None,
            search_text: Some(search_text),
            tags: vec!["table_row".to_string(), table_id.clone()],
            labels: Vec::new(),
            extra_metadata: row_extra,
            enable_embedding: embed_rows,
            auto_tag: false,
            extract_dates: true,     // Extract dates from cell values
            extract_triplets: false, // Table rows don't need triplet extraction
            parent_id: None,
            role: crate::FrameRole::default(),
            no_raw: false,
            source_path: None,
            dedup: false,
            instant_index: false, // Tables are batch operations, commit at end
            extraction_budget_ms: 0, // No budget for table rows
        };

        let should_embed = embed_rows && embedder.is_some();
        if should_embed {
            let embedder = embedder.expect("checked above");
            let text = row_options.search_text.as_deref().unwrap_or_default();
            let embedding = embedder.embed_query(text)?;

            if let Some(identity) = embedding_identity {
                if let Some(provider) = identity.provider.as_deref() {
                    row_options.extra_metadata.insert(
                        MEMVID_EMBEDDING_PROVIDER_KEY.to_string(),
                        provider.to_string(),
                    );
                }
                if let Some(model) = identity.model.as_deref() {
                    row_options
                        .extra_metadata
                        .insert(MEMVID_EMBEDDING_MODEL_KEY.to_string(), model.to_string());
                }
                if let Some(dimension) = identity.dimension {
                    row_options.extra_metadata.insert(
                        MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
                        dimension.to_string(),
                    );
                } else {
                    row_options.extra_metadata.insert(
                        MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
                        embedding.len().to_string(),
                    );
                }
                if let Some(normalized) = identity.normalized {
                    row_options.extra_metadata.insert(
                        MEMVID_EMBEDDING_NORMALIZED_KEY.to_string(),
                        normalized.to_string(),
                    );
                }
            } else {
                row_options.extra_metadata.insert(
                    MEMVID_EMBEDDING_DIMENSION_KEY.to_string(),
                    embedding.len().to_string(),
                );
            }

            let row_frame_id = mem.next_frame_id();
            mem.put_with_embedding_and_options(&row_payload, embedding, row_options)?;
            row_frame_ids.push(row_frame_id);
        } else {
            let row_frame_id = mem.next_frame_id();
            mem.put_bytes_with_options(&row_payload, row_options)?;
            row_frame_ids.push(row_frame_id);
        }
    }

    Ok((meta_frame_id, row_frame_ids))
}

/// List all tables stored in an MV2 file.
///
/// # Arguments
/// * `mem` - The Memvid instance to read from (mutable due to internal caching)
///
/// # Returns
/// Vector of table summaries
pub fn list_tables(mem: &mut Memvid) -> Result<Vec<TableSummary>> {
    // First, collect the frame IDs that are table_meta frames
    let meta_frame_ids: Vec<FrameId> = mem
        .toc
        .frames
        .iter()
        .enumerate()
        .filter(|(_, frame)| frame.kind.as_deref() == Some(TABLE_META_KIND))
        .map(|(id, _)| id as FrameId)
        .collect();

    let mut summaries = Vec::new();

    // Now iterate over the collected frame IDs
    for frame_id in meta_frame_ids {
        // Read frame payload
        let payload_bytes = mem.frame_canonical_payload(frame_id)?;
        let payload = String::from_utf8_lossy(&payload_bytes);
        let meta: serde_json::Value =
            serde_json::from_str(&payload).map_err(|e| MemvidError::TableExtraction {
                reason: format!("failed to parse table metadata: {e}"),
            })?;

        let table_id = meta["table_id"].as_str().unwrap_or("unknown").to_string();
        let source_file = meta["source_file"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();
        let page_start = meta["page_start"].as_u64().unwrap_or(0);
        let page_end = meta["page_end"].as_u64().unwrap_or(0);
        let n_rows = meta["n_rows"].as_u64().unwrap_or(0) as usize;
        let n_cols = meta["n_cols"].as_u64().unwrap_or(0) as usize;
        let quality = meta["quality"].as_str().unwrap_or("unknown").to_string();
        let headers = meta["headers"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();

        summaries.push(TableSummary {
            table_id,
            source_file,
            page_start: page_start as u32,
            page_end: page_end as u32,
            n_rows,
            n_cols,
            quality: quality.parse().unwrap_or(TableQuality::Medium),
            headers,
            frame_id,
        });
    }

    Ok(summaries)
}

/// Get a table by its ID.
///
/// # Arguments
/// * `mem` - The Memvid instance to read from (mutable due to internal caching)
/// * `table_id` - The table ID to look up
///
/// # Returns
/// The reconstructed ExtractedTable if found
pub fn get_table(mem: &mut Memvid, table_id: &str) -> Result<Option<ExtractedTable>> {
    // First, find the meta frame ID by scanning frames
    let meta_frame_id: Option<FrameId> = mem
        .toc
        .frames
        .iter()
        .enumerate()
        .find(|(_, f)| {
            f.kind.as_deref() == Some(TABLE_META_KIND)
                && f.extra_metadata
                    .get("table_id")
                    .map(|id| id == table_id)
                    .unwrap_or(false)
        })
        .map(|(id, _)| id as FrameId);

    let meta_frame_id = match meta_frame_id {
        Some(id) => id,
        None => return Ok(None),
    };

    // Read metadata
    let payload_bytes = mem.frame_canonical_payload(meta_frame_id)?;
    let payload = String::from_utf8_lossy(&payload_bytes);
    let meta: serde_json::Value =
        serde_json::from_str(&payload).map_err(|e| MemvidError::TableExtraction {
            reason: format!("failed to parse table metadata: {e}"),
        })?;

    // Reconstruct table
    let mut table = ExtractedTable::new(
        meta["table_id"].as_str().unwrap_or(""),
        meta["source_file"].as_str().unwrap_or(""),
    );

    table.source_uri = meta["source_uri"].as_str().map(String::from);
    table.page_start = meta["page_start"].as_u64().unwrap_or(1) as u32;
    table.page_end = meta["page_end"].as_u64().unwrap_or(1) as u32;
    table.n_cols = meta["n_cols"].as_u64().unwrap_or(0) as usize;
    table.n_rows = meta["n_rows"].as_u64().unwrap_or(0) as usize;
    table.confidence_score = meta["confidence_score"].as_f64().unwrap_or(0.5) as f32;
    table.extraction_ms = meta["extraction_ms"].as_u64().unwrap_or(0);

    table.headers = meta["headers"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    table.warnings = meta["warnings"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    table.quality = meta["quality"]
        .as_str()
        .and_then(|s| s.parse().ok())
        .unwrap_or(TableQuality::Medium);

    // Find row frame IDs and their row indices (collect both to avoid borrow issues)
    let mut row_frame_ids: Vec<(FrameId, usize)> = mem
        .toc
        .frames
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            f.kind.as_deref() == Some(TABLE_ROW_KIND)
                && f.extra_metadata
                    .get("table_id")
                    .map(|id| id == table_id)
                    .unwrap_or(false)
        })
        .map(|(id, f)| {
            let row_index = f
                .extra_metadata
                .get("row_index")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            (id as FrameId, row_index)
        })
        .collect();

    // Sort by row_index
    row_frame_ids.sort_by_key(|(_, row_index)| *row_index);

    // Now read each row frame
    for (frame_id, _) in row_frame_ids {
        let row_payload_bytes = mem.frame_canonical_payload(frame_id)?;
        let row_payload = String::from_utf8_lossy(&row_payload_bytes);
        let row_data: serde_json::Value =
            serde_json::from_str(&row_payload).map_err(|e| MemvidError::TableExtraction {
                reason: format!("failed to parse row data: {e}"),
            })?;

        let row_index = row_data["row_index"].as_u64().unwrap_or(0) as usize;
        let page = row_data["page"].as_u64().unwrap_or(1) as u32;

        let cells: Vec<super::types::TableCell> =
            if let Some(cell_map) = row_data["cells"].as_object() {
                table
                    .headers
                    .iter()
                    .enumerate()
                    .map(|(col_idx, header)| {
                        let text = cell_map
                            .get(header)
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        super::types::TableCell::new(text, col_idx)
                    })
                    .collect()
            } else {
                Vec::new()
            };

        table
            .rows
            .push(super::types::TableRow::new(row_index, page, cells));
    }

    Ok(Some(table))
}

/// Export a table to CSV format.
///
/// # Arguments
/// * `table` - The table to export
///
/// # Returns
/// CSV formatted string
pub fn export_to_csv(table: &ExtractedTable) -> String {
    let mut output = String::new();

    // Write headers
    if !table.headers.is_empty() {
        let header_line: Vec<String> = table.headers.iter().map(|h| escape_csv_field(h)).collect();
        output.push_str(&header_line.join(","));
        output.push('\n');
    }

    // Write data rows
    for row in &table.rows {
        if row.is_header_row {
            continue;
        }

        let row_line: Vec<String> = row
            .cells
            .iter()
            .map(|c| escape_csv_field(&c.text))
            .collect();
        output.push_str(&row_line.join(","));
        output.push('\n');
    }

    output
}

/// Escape a field for CSV output.
fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Export a table to JSON format.
///
/// # Arguments
/// * `table` - The table to export
/// * `as_records` - If true, export as array of records; if false, as columns
///
/// # Returns
/// JSON formatted string
pub fn export_to_json(table: &ExtractedTable, as_records: bool) -> Result<String> {
    if as_records {
        // Array of {header: value} objects
        let records: Vec<serde_json::Value> = table
            .data_rows()
            .iter()
            .map(|row| {
                let mut obj = serde_json::Map::new();
                for (i, header) in table.headers.iter().enumerate() {
                    let value = row.cells.get(i).map(|c| c.text.clone()).unwrap_or_default();
                    obj.insert(header.clone(), serde_json::Value::String(value));
                }
                serde_json::Value::Object(obj)
            })
            .collect();

        serde_json::to_string_pretty(&records).map_err(|e| MemvidError::TableExtraction {
            reason: format!("failed to serialize to JSON: {e}"),
        })
    } else {
        // Full table structure
        serde_json::to_string_pretty(table).map_err(|e| MemvidError::TableExtraction {
            reason: format!("failed to serialize to JSON: {e}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::types::{DetectionMode, TableCell, TableRow};

    fn make_test_table() -> ExtractedTable {
        let mut table = ExtractedTable::new("test_001", "test.pdf");
        table.headers = vec!["Name".to_string(), "Age".to_string(), "City".to_string()];
        table.n_cols = 3;
        table.page_start = 1;
        table.page_end = 1;
        table.detection_mode = DetectionMode::Lattice;
        table.quality = TableQuality::High;

        // Add header row
        let header_cells = vec![
            TableCell::new("Name", 0).as_header(),
            TableCell::new("Age", 1).as_header(),
            TableCell::new("City", 2).as_header(),
        ];
        table
            .rows
            .push(TableRow::new(0, 1, header_cells).as_header());

        // Add data rows
        table.rows.push(TableRow::new(
            1,
            1,
            vec![
                TableCell::new("Alice", 0),
                TableCell::new("30", 1),
                TableCell::new("New York", 2),
            ],
        ));
        table.rows.push(TableRow::new(
            2,
            1,
            vec![
                TableCell::new("Bob", 0),
                TableCell::new("25", 1),
                TableCell::new("Los Angeles", 2),
            ],
        ));

        table.n_rows = 2;
        table
    }

    #[test]
    fn test_export_to_csv() {
        let table = make_test_table();
        let csv = export_to_csv(&table);

        assert!(csv.contains("Name,Age,City"));
        assert!(csv.contains("Alice,30,New York"));
        assert!(csv.contains("Bob,25,Los Angeles"));
    }

    #[test]
    fn test_csv_escaping() {
        assert_eq!(escape_csv_field("simple"), "simple");
        assert_eq!(escape_csv_field("with,comma"), "\"with,comma\"");
        assert_eq!(escape_csv_field("with\"quote"), "\"with\"\"quote\"");
        assert_eq!(escape_csv_field("with\nnewline"), "\"with\nnewline\"");
    }

    #[test]
    fn test_export_to_json_records() {
        let table = make_test_table();
        let json = export_to_json(&table, true).unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["Name"], "Alice");
        assert_eq!(parsed[0]["Age"], "30");
    }

    #[test]
    fn test_export_to_json_full() {
        let table = make_test_table();
        let json = export_to_json(&table, false).unwrap();

        let parsed: ExtractedTable = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.table_id, "test_001");
        assert_eq!(parsed.headers.len(), 3);
    }
}
