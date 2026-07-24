//! Table structure detection for XLSX sheets.
//!
//! Detects header rows, table boundaries, and column types for sheets
//! not covered by OOXML table definitions.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use serde::{Deserialize, Serialize};

use super::xlsx_ooxml::{MergedRegion, NumFmtKind, TableDefinition};

/// Column type inferred from data sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ColumnType {
    #[default]
    Text,
    Integer,
    Float,
    Date,
    DateTime,
    Time,
    Currency,
    Percentage,
    Boolean,
    Mixed,
    Empty,
}

/// A detected table within a sheet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedTable {
    /// Table name (from OOXML or auto-generated)
    pub name: String,
    /// Sheet name this table belongs to
    pub sheet_name: String,
    /// Column headers (may be empty if no header detected)
    pub headers: Vec<String>,
    /// Column types inferred from data
    pub column_types: Vec<ColumnType>,
    /// First data row (0-based, the row after the header)
    pub first_data_row: u32,
    /// Last data row (inclusive, 0-based)
    pub last_data_row: u32,
    /// First column (0-based)
    pub first_col: u32,
    /// Last column (inclusive, 0-based)
    pub last_col: u32,
    /// Header row index (0-based), None if no header detected
    pub header_row: Option<u32>,
    /// Detection confidence (0.0 - 1.0)
    pub confidence: f64,
}

/// A cell value representation for detection purposes.
#[derive(Debug, Clone)]
pub enum CellValue {
    Empty,
    Text(String),
    Number(f64),
    Integer(i64),
    Boolean(bool),
    DateTime(String),
    Error(String),
}

impl CellValue {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    #[must_use]
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Number(_) | Self::Integer(_))
    }

    #[must_use]
    pub fn as_text(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Text(s) => s.clone(),
            Self::Number(v) => format!("{v}"),
            Self::Integer(v) => format!("{v}"),
            Self::Boolean(b) => if *b { "true" } else { "false" }.to_string(),
            Self::DateTime(s) => s.clone(),
            Self::Error(s) => s.clone(),
        }
    }
}

/// A grid of cell values representing one sheet.
pub struct SheetGrid {
    pub sheet_name: String,
    pub rows: Vec<Vec<CellValue>>,
    /// Number format kinds per cell (row, col) if available from OOXML metadata.
    /// Outer vec is rows, inner is columns.
    pub num_fmt_kinds: Vec<Vec<NumFmtKind>>,
    pub num_rows: u32,
    pub num_cols: u32,
}

impl SheetGrid {
    #[must_use]
    pub fn new(sheet_name: String) -> Self {
        Self {
            sheet_name,
            rows: Vec::new(),
            num_fmt_kinds: Vec::new(),
            num_rows: 0,
            num_cols: 0,
        }
    }

    /// Get cell value at (row, col). Returns Empty if out of bounds.
    #[must_use]
    pub fn cell(&self, row: u32, col: u32) -> &CellValue {
        static EMPTY: CellValue = CellValue::Empty;
        self.rows
            .get(row as usize)
            .and_then(|r| r.get(col as usize))
            .unwrap_or(&EMPTY)
    }

    /// Get number format kind at (row, col). Returns General if not available.
    #[must_use]
    pub fn num_fmt(&self, row: u32, col: u32) -> NumFmtKind {
        self.num_fmt_kinds
            .get(row as usize)
            .and_then(|r| r.get(col as usize))
            .copied()
            .unwrap_or(NumFmtKind::General)
    }

    /// Check if a row is entirely empty.
    #[must_use]
    pub fn is_row_empty(&self, row: u32) -> bool {
        if let Some(r) = self.rows.get(row as usize) {
            r.iter().all(CellValue::is_empty)
        } else {
            true
        }
    }

    /// Count non-empty cells in a row.
    #[must_use]
    pub fn row_nonempty_count(&self, row: u32) -> usize {
        if let Some(r) = self.rows.get(row as usize) {
            r.iter().filter(|c| !c.is_empty()).count()
        } else {
            0
        }
    }
}

/// Detect tables within a sheet grid.
///
/// Uses cascading heuristics:
/// 1. OOXML table definitions (confidence 1.0)
/// 2. All-text row + typed data below (0.7)
/// 3. Type consistency boost (+0.15)
/// 4. First non-empty row fallback (0.4)
#[must_use]
pub fn detect_tables(
    grid: &SheetGrid,
    ooxml_tables: &[TableDefinition],
    merged_regions: &[MergedRegion],
) -> Vec<DetectedTable> {
    let mut tables = Vec::new();

    // Phase 1: Use OOXML table definitions for this sheet
    for tdef in ooxml_tables {
        if tdef.sheet_name == grid.sheet_name {
            let column_types = infer_column_types(
                grid,
                tdef.first_row + 1,
                tdef.last_row,
                tdef.first_col,
                tdef.last_col,
            );
            tables.push(DetectedTable {
                name: tdef.name.clone(),
                sheet_name: grid.sheet_name.clone(),
                headers: tdef.headers.clone(),
                column_types,
                first_data_row: tdef.first_row + 1,
                last_data_row: tdef.last_row,
                first_col: tdef.first_col,
                last_col: tdef.last_col,
                header_row: Some(tdef.first_row),
                confidence: 1.0,
            });
        }
    }

    // If OOXML tables covered the whole sheet, we're done
    if !tables.is_empty() {
        return tables;
    }

    // Phase 2: Heuristic detection — find table boundaries
    let table_ranges = find_table_boundaries(grid, merged_regions);
    let mut table_idx = 0;

    for (start_row, end_row, start_col, end_col) in table_ranges {
        let (header_row, headers, confidence) =
            detect_header(grid, start_row, end_row, start_col, end_col);

        let first_data_row = header_row.map_or(start_row, |hr| hr + 1);
        let column_types = infer_column_types(grid, first_data_row, end_row, start_col, end_col);

        // Boost confidence if column types are consistent
        let type_boost = if column_types
            .iter()
            .filter(|t| **t != ColumnType::Mixed && **t != ColumnType::Empty)
            .count()
            > column_types.len() / 2
        {
            0.15
        } else {
            0.0
        };

        table_idx += 1;
        tables.push(DetectedTable {
            name: format!("Table{table_idx}"),
            sheet_name: grid.sheet_name.clone(),
            headers,
            column_types,
            first_data_row,
            last_data_row: end_row,
            first_col: start_col,
            last_col: end_col,
            header_row,
            confidence: (confidence + type_boost).min(1.0),
        });
    }

    tables
}

/// Find table boundaries by detecting gaps (2+ consecutive empty rows/cols).
fn find_table_boundaries(
    grid: &SheetGrid,
    _merged_regions: &[MergedRegion],
) -> Vec<(u32, u32, u32, u32)> {
    if grid.num_rows == 0 || grid.num_cols == 0 {
        return Vec::new();
    }

    // Find vertical boundaries (consecutive empty rows split tables)
    let mut row_groups: Vec<(u32, u32)> = Vec::new();
    let mut current_start: Option<u32> = None;
    let mut empty_streak = 0u32;

    for row in 0..grid.num_rows {
        if grid.is_row_empty(row) {
            empty_streak += 1;
            if empty_streak >= 2 {
                if let Some(start) = current_start.take() {
                    let end = row.saturating_sub(empty_streak);
                    if end >= start {
                        row_groups.push((start, end));
                    }
                }
            }
        } else {
            if current_start.is_none() {
                current_start = Some(row);
            }
            empty_streak = 0;
        }
    }
    // Close the last group
    if let Some(start) = current_start {
        row_groups.push((start, grid.num_rows.saturating_sub(1)));
    }

    // For each row group, find column boundaries
    let mut boundaries = Vec::new();
    for (start_row, end_row) in row_groups {
        let col_ranges = find_column_boundaries(grid, start_row, end_row);
        for (start_col, end_col) in col_ranges {
            boundaries.push((start_row, end_row, start_col, end_col));
        }
    }

    // Fallback: if no boundaries detected, treat entire used area as one table
    if boundaries.is_empty() && grid.num_rows > 0 {
        boundaries.push((
            0,
            grid.num_rows.saturating_sub(1),
            0,
            grid.num_cols.saturating_sub(1),
        ));
    }

    boundaries
}

/// Find horizontal table boundaries within a row range.
fn find_column_boundaries(grid: &SheetGrid, start_row: u32, end_row: u32) -> Vec<(u32, u32)> {
    if grid.num_cols == 0 {
        return Vec::new();
    }

    // Check which columns have any data in the row range
    let mut col_has_data = vec![false; grid.num_cols as usize];
    for row in start_row..=end_row {
        if let Some(r) = grid.rows.get(row as usize) {
            for (ci, cell) in r.iter().enumerate() {
                if !cell.is_empty() {
                    col_has_data[ci] = true;
                }
            }
        }
    }

    // Find contiguous ranges of columns with data
    let mut ranges = Vec::new();
    let mut current_start: Option<u32> = None;
    let mut empty_streak = 0u32;

    for (ci, &has_data) in col_has_data.iter().enumerate() {
        if has_data {
            if current_start.is_none() {
                current_start = Some(ci as u32);
            }
            empty_streak = 0;
        } else {
            empty_streak += 1;
            if empty_streak >= 2 {
                if let Some(start) = current_start.take() {
                    let end = (ci as u32).saturating_sub(empty_streak);
                    if end >= start {
                        ranges.push((start, end));
                    }
                }
            }
        }
    }
    if let Some(start) = current_start {
        ranges.push((start, (grid.num_cols).saturating_sub(1)));
    }

    // Fallback: whole range
    if ranges.is_empty() {
        ranges.push((0, grid.num_cols.saturating_sub(1)));
    }

    ranges
}

/// Detect header row within a table range.
/// Returns (header_row_index, header_texts, confidence).
fn detect_header(
    grid: &SheetGrid,
    start_row: u32,
    end_row: u32,
    start_col: u32,
    end_col: u32,
) -> (Option<u32>, Vec<String>, f64) {
    // Heuristic 1: All-text row followed by typed (numeric/date) data below
    for row in start_row..=end_row.min(start_row + 3) {
        let nonempty = grid.row_nonempty_count(row);
        if nonempty == 0 {
            continue;
        }

        let all_text = (start_col..=end_col).all(|col| {
            let cell = grid.cell(row, col);
            cell.is_empty() || cell.is_text()
        });

        if !all_text {
            continue;
        }

        // Check if the next row has any numeric/date data
        let next_row = row + 1;
        if next_row > end_row {
            continue;
        }
        let has_typed_data = (start_col..=end_col).any(|col| {
            let cell = grid.cell(next_row, col);
            cell.is_numeric() || matches!(cell, CellValue::DateTime(_) | CellValue::Boolean(_))
        });

        if has_typed_data {
            let headers: Vec<String> = (start_col..=end_col)
                .map(|col| grid.cell(row, col).as_text())
                .collect();
            return (Some(row), headers, 0.7);
        }
    }

    // Heuristic 2: First non-empty row as fallback
    for row in start_row..=end_row.min(start_row + 5) {
        if grid.row_nonempty_count(row) > 0 {
            let headers: Vec<String> = (start_col..=end_col)
                .map(|col| grid.cell(row, col).as_text())
                .collect();
            return (Some(row), headers, 0.4);
        }
    }

    (None, Vec::new(), 0.3)
}

/// Infer column types by sampling data rows.
fn infer_column_types(
    grid: &SheetGrid,
    first_data_row: u32,
    last_data_row: u32,
    first_col: u32,
    last_col: u32,
) -> Vec<ColumnType> {
    let num_cols = (last_col - first_col + 1) as usize;
    let mut type_counts: Vec<[u32; 10]> = vec![[0; 10]; num_cols];
    let sample_limit = 100;
    for (sampled, row) in (first_data_row..=last_data_row).enumerate() {
        if sampled >= sample_limit as usize {
            break;
        }

        for col_offset in 0..num_cols {
            let col = first_col + col_offset as u32;
            let cell = grid.cell(row, col);
            let fmt = grid.num_fmt(row, col);

            let type_idx = match (cell, fmt) {
                (CellValue::Empty, _) => 9, // Empty
                (CellValue::Text(_), _) => 0,
                (CellValue::Integer(_), NumFmtKind::Date) => 2,
                (CellValue::Integer(_), NumFmtKind::DateTime) => 3,
                (CellValue::Integer(_), NumFmtKind::Time) => 4,
                (CellValue::Integer(_), NumFmtKind::Currency) => 5,
                (CellValue::Integer(_), NumFmtKind::Percentage) => 6,
                (CellValue::Integer(_), _) => 1,
                (CellValue::Number(_), NumFmtKind::Date) => 2,
                (CellValue::Number(_), NumFmtKind::DateTime) => 3,
                (CellValue::Number(_), NumFmtKind::Time) => 4,
                (CellValue::Number(_), NumFmtKind::Currency) => 5,
                (CellValue::Number(_), NumFmtKind::Percentage) => 6,
                (CellValue::Number(_), _) => 8, // Float
                (CellValue::Boolean(_), _) => 7,
                (CellValue::DateTime(_), _) => 2,
                (CellValue::Error(_), _) => 9,
            };
            type_counts[col_offset][type_idx] += 1;
        }
    }

    type_counts
        .iter()
        .map(|counts| {
            // Find the most common non-empty type
            let non_empty_total: u32 = counts.iter().take(9).sum();
            if non_empty_total == 0 {
                return ColumnType::Empty;
            }

            let (max_idx, &max_count) = counts
                .iter()
                .take(9)
                .enumerate()
                .max_by_key(|&(_, c)| *c)
                .unwrap_or((0, &0));

            // If >30% are a different type, it's mixed
            let threshold = (non_empty_total as f64 * 0.3).ceil() as u32;
            let other_count = non_empty_total - max_count;
            if other_count >= threshold && max_count < non_empty_total {
                return ColumnType::Mixed;
            }

            match max_idx {
                0 => ColumnType::Text,
                1 => ColumnType::Integer,
                2 => ColumnType::Date,
                3 => ColumnType::DateTime,
                4 => ColumnType::Time,
                5 => ColumnType::Currency,
                6 => ColumnType::Percentage,
                7 => ColumnType::Boolean,
                8 => ColumnType::Float,
                _ => ColumnType::Text,
            }
        })
        .collect()
}

/// Propagate merged cell values into a grid.
/// The top-left cell's value is copied to all cells in the merged region.
#[allow(dead_code)]
pub fn propagate_merged_cells(grid: &mut SheetGrid, merged_regions: &[MergedRegion]) {
    for region in merged_regions {
        // Get the top-left cell value
        let value = grid.cell(region.top_row, region.left_col).clone();
        if value.is_empty() {
            continue;
        }

        // Fill all cells in the region with the top-left value
        for row in region.top_row..=region.bottom_row {
            for col in region.left_col..=region.right_col {
                // Skip the top-left cell itself
                if row == region.top_row && col == region.left_col {
                    continue;
                }
                if let Some(r) = grid.rows.get_mut(row as usize) {
                    // Extend the row if necessary
                    while r.len() <= col as usize {
                        r.push(CellValue::Empty);
                    }
                    r[col as usize] = value.clone();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_detect_header_all_text_row() {
        let grid = make_grid(
            vec![
                vec![
                    CellValue::Text("Name".into()),
                    CellValue::Text("Age".into()),
                    CellValue::Text("City".into()),
                ],
                vec![
                    CellValue::Text("Alice".into()),
                    CellValue::Integer(30),
                    CellValue::Text("Austin".into()),
                ],
                vec![
                    CellValue::Text("Bob".into()),
                    CellValue::Integer(25),
                    CellValue::Text("Boston".into()),
                ],
            ],
            "Sheet1",
        );

        let tables = detect_tables(&grid, &[], &[]);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].headers, vec!["Name", "Age", "City"]);
        assert!(tables[0].confidence >= 0.7);
        assert_eq!(tables[0].header_row, Some(0));
    }

    #[test]
    fn test_detect_multi_table_gap() {
        let grid = make_grid(
            vec![
                vec![CellValue::Text("A".into()), CellValue::Integer(1)],
                vec![CellValue::Text("B".into()), CellValue::Integer(2)],
                vec![CellValue::Empty, CellValue::Empty],
                vec![CellValue::Empty, CellValue::Empty],
                vec![CellValue::Text("X".into()), CellValue::Integer(10)],
                vec![CellValue::Text("Y".into()), CellValue::Integer(20)],
            ],
            "Sheet1",
        );

        let tables = detect_tables(&grid, &[], &[]);
        assert_eq!(tables.len(), 2);
    }

    #[test]
    fn test_propagate_merged_cells() {
        let mut grid = make_grid(
            vec![
                vec![
                    CellValue::Text("Merged Title".into()),
                    CellValue::Empty,
                    CellValue::Empty,
                ],
                vec![
                    CellValue::Text("A".into()),
                    CellValue::Text("B".into()),
                    CellValue::Text("C".into()),
                ],
            ],
            "Sheet1",
        );

        let regions = vec![MergedRegion {
            top_row: 0,
            left_col: 0,
            bottom_row: 0,
            right_col: 2,
        }];

        propagate_merged_cells(&mut grid, &regions);

        assert!(matches!(grid.cell(0, 1), CellValue::Text(s) if s == "Merged Title"));
        assert!(matches!(grid.cell(0, 2), CellValue::Text(s) if s == "Merged Title"));
    }

    #[test]
    fn test_column_type_inference() {
        let grid = make_grid(
            vec![
                vec![
                    CellValue::Text("Name".into()),
                    CellValue::Text("Value".into()),
                ],
                vec![CellValue::Text("A".into()), CellValue::Integer(100)],
                vec![CellValue::Text("B".into()), CellValue::Integer(200)],
                vec![CellValue::Text("C".into()), CellValue::Integer(300)],
            ],
            "Sheet1",
        );

        let types = infer_column_types(&grid, 1, 3, 0, 1);
        assert_eq!(types[0], ColumnType::Text);
        assert_eq!(types[1], ColumnType::Integer);
    }

    #[test]
    fn test_empty_grid() {
        let grid = make_grid(Vec::new(), "Empty");
        let tables = detect_tables(&grid, &[], &[]);
        assert!(tables.is_empty());
    }
}
