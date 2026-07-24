//! Row-aligned semantic chunking for XLSX spreadsheets.
//!
//! Produces structure-aware chunks that:
//! - Never split a row across chunks
//! - Prefix every chunk with sheet/table context and header row
//! - Format rows as `Header: Value | Header: Value` for search accuracy
//! - Skip empty cells for compact output
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use crate::types::structure::{ChunkingResult, StructuredChunk};

use super::xlsx_ooxml::{
    NumFmtKind, OoxmlMetadata, excel_serial_to_iso, format_currency, format_percentage,
};
use super::xlsx_table_detect::{CellValue, DetectedTable, SheetGrid};

/// Default target chunk size in characters.
const DEFAULT_MAX_CHUNK_CHARS: usize = 1200;

/// Maximum number of chunks to produce from a single workbook.
const MAX_SPREADSHEET_CHUNKS: usize = 500;

/// Options for XLSX semantic chunking.
#[derive(Debug, Clone)]
pub struct XlsxChunkingOptions {
    pub max_chars: usize,
    pub max_chunks: usize,
}

impl Default for XlsxChunkingOptions {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_MAX_CHUNK_CHARS,
            max_chunks: MAX_SPREADSHEET_CHUNKS,
        }
    }
}

/// Format a cell value using OOXML metadata for type-aware rendering.
#[must_use]
pub fn format_cell_value(
    cell: &CellValue,
    fmt_kind: NumFmtKind,
    _metadata: &OoxmlMetadata,
) -> String {
    match (cell, fmt_kind) {
        (CellValue::Empty, _) => String::new(),
        (CellValue::Text(s), _) => s.trim().to_string(),
        (CellValue::Number(v), NumFmtKind::Date | NumFmtKind::DateTime) => {
            excel_serial_to_iso(*v).unwrap_or_else(|| format!("{v}"))
        }
        (CellValue::Number(v), NumFmtKind::Percentage) => format_percentage(*v),
        (CellValue::Number(v), NumFmtKind::Currency) => format_currency(*v, "$"),
        (CellValue::Number(v), _) => {
            // Clean up float display — use integer format if no fractional part
            if (v.fract()).abs() < 1e-10 {
                format!("{}", *v as i64)
            } else {
                format!("{v}")
            }
        }
        (CellValue::Integer(v), NumFmtKind::Date | NumFmtKind::DateTime) => {
            excel_serial_to_iso(*v as f64).unwrap_or_else(|| format!("{v}"))
        }
        (CellValue::Integer(v), NumFmtKind::Percentage) => format_percentage(*v as f64),
        (CellValue::Integer(v), NumFmtKind::Currency) => format_currency(*v as f64, "$"),
        (CellValue::Integer(v), _) => format!("{v}"),
        (CellValue::Boolean(b), _) => if *b { "true" } else { "false" }.to_string(),
        (CellValue::DateTime(s), _) => s.clone(),
        (CellValue::Error(s), _) => s.clone(),
    }
}

/// Format a single row as `Header: Value | Header: Value`, skipping empty cells.
fn format_row_with_headers(
    grid: &SheetGrid,
    row_idx: u32,
    headers: &[String],
    first_col: u32,
    last_col: u32,
    metadata: &OoxmlMetadata,
) -> String {
    let mut parts = Vec::new();

    for col in first_col..=last_col {
        let cell = grid.cell(row_idx, col);
        if cell.is_empty() {
            continue;
        }

        let fmt_kind = grid.num_fmt(row_idx, col);
        let formatted = format_cell_value(cell, fmt_kind, metadata);
        if formatted.is_empty() {
            continue;
        }

        let col_offset = (col - first_col) as usize;
        let header = headers.get(col_offset).filter(|h| !h.is_empty()).cloned();

        if let Some(h) = header {
            parts.push(format!("{h}: {formatted}"));
        } else {
            parts.push(formatted);
        }
    }

    parts.join(" | ")
}

/// Build a context prefix for a chunk: `[Sheet: X] [Table: Y]`
fn build_context_prefix(sheet_name: &str, table_name: &str) -> String {
    format!("[Sheet: {sheet_name}] [Table: {table_name}]")
}

/// Build a header line: `Header1 | Header2 | Header3`
fn build_header_line(headers: &[String]) -> String {
    let nonempty: Vec<&str> = headers
        .iter()
        .map(String::as_str)
        .filter(|h| !h.is_empty())
        .collect();
    if nonempty.is_empty() {
        String::new()
    } else {
        nonempty.join(" | ")
    }
}

/// Chunk a single detected table into structure-aware chunks.
fn chunk_table(
    grid: &SheetGrid,
    table: &DetectedTable,
    metadata: &OoxmlMetadata,
    options: &XlsxChunkingOptions,
    chunk_index_start: usize,
) -> Vec<StructuredChunk> {
    let context_prefix = build_context_prefix(&table.sheet_name, &table.name);
    let header_line = build_header_line(&table.headers);

    // Build the fixed prefix that goes into every chunk
    let fixed_prefix = if header_line.is_empty() {
        format!("{context_prefix}\n")
    } else {
        format!("{context_prefix}\n{header_line}\n")
    };
    let prefix_len = fixed_prefix.len();

    // Format all data rows
    let mut formatted_rows: Vec<String> = Vec::new();
    for row_idx in table.first_data_row..=table.last_data_row {
        let line = format_row_with_headers(
            grid,
            row_idx,
            &table.headers,
            table.first_col,
            table.last_col,
            metadata,
        );
        if !line.is_empty() {
            formatted_rows.push(line);
        }
    }

    if formatted_rows.is_empty() {
        return Vec::new();
    }

    // Bin-pack rows into chunks, respecting max_chars
    let mut chunks = Vec::new();
    let mut current_rows: Vec<String> = Vec::new();
    let mut current_len = prefix_len;

    for row_text in &formatted_rows {
        let row_len = row_text.len() + 1; // +1 for newline

        if !current_rows.is_empty() && current_len + row_len > options.max_chars {
            // Emit current chunk
            let text = format!("{fixed_prefix}{}", current_rows.join("\n"));
            chunks.push(text);
            current_rows.clear();
            current_len = prefix_len;
        }

        current_rows.push(row_text.clone());
        current_len += row_len;
    }

    // Emit final chunk
    if !current_rows.is_empty() {
        let text = format!("{fixed_prefix}{}", current_rows.join("\n"));
        chunks.push(text);
    }

    // Convert to StructuredChunk
    let total_parts = chunks.len() as u32;
    let table_id = format!("{}:{}", table.sheet_name, table.name);

    chunks
        .into_iter()
        .enumerate()
        .map(|(i, text)| {
            let char_count = text.len();
            let idx = chunk_index_start + i;

            if total_parts == 1 {
                StructuredChunk::table(text, idx, &table_id, 0, char_count)
            } else {
                StructuredChunk::table_continuation(
                    text,
                    idx,
                    &table_id,
                    (i + 1) as u32,
                    total_parts,
                    &fixed_prefix,
                    0,
                    char_count,
                )
            }
        })
        .collect()
}

/// Chunk an entire workbook's detected tables into structured chunks.
#[must_use]
pub fn chunk_workbook(
    grids: &[SheetGrid],
    tables: &[DetectedTable],
    metadata: &OoxmlMetadata,
    options: &XlsxChunkingOptions,
) -> ChunkingResult {
    let mut result = ChunkingResult::empty();
    let mut chunk_index = 0;

    for table in tables {
        // Find the grid for this table's sheet
        let Some(grid) = grids.iter().find(|g| g.sheet_name == table.sheet_name) else {
            result.warn(format!(
                "No grid found for sheet '{}', skipping table '{}'",
                table.sheet_name, table.name
            ));
            continue;
        };

        let table_chunks = chunk_table(grid, table, metadata, options, chunk_index);

        if table_chunks.len() > 1 {
            result.tables_split += 1;
        }
        result.tables_processed += 1;
        chunk_index += table_chunks.len();
        result.chunks.extend(table_chunks);

        // Respect global chunk limit
        if result.chunks.len() >= options.max_chunks {
            result.warn(format!(
                "Hit max chunk limit ({}) — remaining tables skipped",
                options.max_chunks
            ));
            result.chunks.truncate(options.max_chunks);
            break;
        }
    }

    result
}

/// Generate backward-compatible flat text from grids (for `ReaderOutput.document.text`).
#[must_use]
pub fn generate_flat_text(
    grids: &[SheetGrid],
    tables: &[DetectedTable],
    metadata: &OoxmlMetadata,
) -> String {
    let mut out = String::new();

    for table in tables {
        let grid = match grids.iter().find(|g| g.sheet_name == table.sheet_name) {
            Some(g) => g,
            None => continue,
        };

        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("Sheet: {}\n", table.sheet_name));

        // Header line
        if !table.headers.is_empty() {
            let header_line = table
                .headers
                .iter()
                .filter(|h| !h.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ");
            if !header_line.is_empty() {
                out.push_str(&header_line);
                out.push('\n');
            }
        }

        // Data rows
        for row_idx in table.first_data_row..=table.last_data_row {
            let line = format_row_with_headers(
                grid,
                row_idx,
                &table.headers,
                table.first_col,
                table.last_col,
                metadata,
            );
            if !line.is_empty() {
                out.push_str(&line);
                out.push('\n');
            }
        }
    }

    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::xlsx_table_detect::SheetGrid;
    use crate::types::structure::ChunkType;

    fn make_grid(data: Vec<Vec<CellValue>>, sheet_name: &str) -> SheetGrid {
        let num_rows = data.len() as u32;
        let num_cols = data.iter().map(Vec::len).max().unwrap_or(0) as u32;
        SheetGrid {
            sheet_name: sheet_name.to_string(),
            rows: data,
            num_fmt_kinds: Vec::new(),
            num_rows,
            num_cols,
        }
    }

    #[test]
    fn test_format_cell_value_date() {
        let metadata = OoxmlMetadata::default();
        let cell = CellValue::Number(44927.0);
        let result = format_cell_value(&cell, NumFmtKind::Date, &metadata);
        assert_eq!(result, "2023-01-01");
    }

    #[test]
    fn test_format_cell_value_percentage() {
        let metadata = OoxmlMetadata::default();
        let cell = CellValue::Number(0.153);
        let result = format_cell_value(&cell, NumFmtKind::Percentage, &metadata);
        assert_eq!(result, "15.3%");
    }

    #[test]
    fn test_format_cell_value_currency() {
        let metadata = OoxmlMetadata::default();
        let cell = CellValue::Number(1234.56);
        let result = format_cell_value(&cell, NumFmtKind::Currency, &metadata);
        assert_eq!(result, "$1234.56");
    }

    #[test]
    fn test_format_row_with_headers() {
        let grid = make_grid(
            vec![vec![
                CellValue::Text("Alice".into()),
                CellValue::Integer(30),
                CellValue::Text("Austin".into()),
            ]],
            "Sheet1",
        );
        let metadata = OoxmlMetadata::default();
        let headers = vec!["Name".to_string(), "Age".to_string(), "City".to_string()];

        let result = format_row_with_headers(&grid, 0, &headers, 0, 2, &metadata);
        assert_eq!(result, "Name: Alice | Age: 30 | City: Austin");
    }

    #[test]
    fn test_format_row_skips_empty() {
        let grid = make_grid(
            vec![vec![
                CellValue::Text("Alice".into()),
                CellValue::Empty,
                CellValue::Text("Austin".into()),
            ]],
            "Sheet1",
        );
        let metadata = OoxmlMetadata::default();
        let headers = vec!["Name".to_string(), "Age".to_string(), "City".to_string()];

        let result = format_row_with_headers(&grid, 0, &headers, 0, 2, &metadata);
        assert_eq!(result, "Name: Alice | City: Austin");
    }

    #[test]
    fn test_chunk_table_single_chunk() {
        let grid = make_grid(
            vec![
                vec![
                    CellValue::Text("Name".into()),
                    CellValue::Text("Value".into()),
                ],
                vec![CellValue::Text("A".into()), CellValue::Integer(100)],
                vec![CellValue::Text("B".into()), CellValue::Integer(200)],
            ],
            "Sheet1",
        );
        let metadata = OoxmlMetadata::default();
        let table = DetectedTable {
            name: "Revenue".to_string(),
            sheet_name: "Sheet1".to_string(),
            headers: vec!["Name".to_string(), "Value".to_string()],
            column_types: vec![],
            first_data_row: 1,
            last_data_row: 2,
            first_col: 0,
            last_col: 1,
            header_row: Some(0),
            confidence: 0.7,
        };

        let options = XlsxChunkingOptions::default();
        let chunks = chunk_table(&grid, &table, &metadata, &options, 0);

        assert_eq!(chunks.len(), 1);
        let text = &chunks[0].text;
        assert!(text.contains("[Sheet: Sheet1] [Table: Revenue]"));
        assert!(text.contains("Name | Value"));
        assert!(text.contains("Name: A | Value: 100"));
        assert!(text.contains("Name: B | Value: 200"));
        assert_eq!(chunks[0].chunk_type, ChunkType::Table);
    }

    #[test]
    fn test_chunk_table_splits_large() {
        let mut rows = vec![vec![
            CellValue::Text("Col1".into()),
            CellValue::Text("Col2".into()),
        ]];
        // Add 50 data rows to exceed a small chunk limit
        for i in 0..50 {
            rows.push(vec![
                CellValue::Text(format!("Row{i} long text that takes up space in the chunk")),
                CellValue::Integer(i as i64 * 1000),
            ]);
        }

        let grid = make_grid(rows, "Sheet1");
        let metadata = OoxmlMetadata::default();
        let table = DetectedTable {
            name: "Data".to_string(),
            sheet_name: "Sheet1".to_string(),
            headers: vec!["Col1".to_string(), "Col2".to_string()],
            column_types: vec![],
            first_data_row: 1,
            last_data_row: 50,
            first_col: 0,
            last_col: 1,
            header_row: Some(0),
            confidence: 0.7,
        };

        let options = XlsxChunkingOptions {
            max_chars: 300,
            max_chunks: 100,
        };
        let chunks = chunk_table(&grid, &table, &metadata, &options, 0);

        assert!(chunks.len() > 1, "Should split into multiple chunks");
        // Every chunk should have the header context
        for chunk in &chunks {
            assert!(chunk.text.contains("[Sheet: Sheet1]"));
            assert!(chunk.text.contains("Col1 | Col2"));
            assert_eq!(chunk.chunk_type, ChunkType::TableContinuation);
        }
    }

    #[test]
    fn test_generate_flat_text() {
        let grid = make_grid(
            vec![
                vec![
                    CellValue::Text("Name".into()),
                    CellValue::Text("Score".into()),
                ],
                vec![CellValue::Text("Alice".into()), CellValue::Integer(95)],
            ],
            "Results",
        );
        let metadata = OoxmlMetadata::default();
        let table = DetectedTable {
            name: "Scores".to_string(),
            sheet_name: "Results".to_string(),
            headers: vec!["Name".to_string(), "Score".to_string()],
            column_types: vec![],
            first_data_row: 1,
            last_data_row: 1,
            first_col: 0,
            last_col: 1,
            header_row: Some(0),
            confidence: 0.7,
        };

        let text = generate_flat_text(&[grid], &[table], &metadata);
        assert!(text.contains("Sheet: Results"));
        assert!(text.contains("Name | Score"));
        assert!(text.contains("Name: Alice | Score: 95"));
    }
}
