// Safe unwrap: Float comparisons and known-valid iterator operations.
#![allow(clippy::unwrap_used)]
//! PDF table extraction using Lattice and Stream detection.
//!
//! This module implements two complementary table detection strategies:
//! - **Lattice**: Detects tables with visible grid lines (ruled tables)
//! - **Stream**: Infers tables from text alignment and whitespace

use std::collections::HashMap;
use std::time::Instant;

use super::layout::{PageLayout, TextBox, cluster_values, extract_pdf_layout};
use super::types::{
    DetectionMode, ExtractedTable, ExtractionMode, TableCell, TableExtractionOptions,
    TableExtractionResult, TableQuality, TableRow,
};
use crate::error::Result;
use regex::Regex;

/// Minimum line length to consider for grid detection (in points).
const MIN_LINE_LENGTH: f32 = 20.0;

/// Minimum number of grid intersections to consider a valid lattice table.
const MIN_GRID_INTERSECTIONS: usize = 4;

/// Extract tables from a PDF document.
///
/// # Arguments
/// * `bytes` - Raw PDF bytes
/// * `source_file` - Original filename for metadata
/// * `options` - Extraction configuration
///
/// # Returns
/// A result containing all extracted tables and diagnostics.
pub fn extract_tables_from_pdf(
    bytes: &[u8],
    source_file: &str,
    options: &TableExtractionOptions,
) -> Result<TableExtractionResult> {
    let start = Instant::now();

    // Extract page layouts
    let layouts = extract_pdf_layout(bytes, options.max_pages)?;
    let pages_processed = layouts.len() as u32;

    let mut all_tables = Vec::new();
    let mut warnings = Vec::new();

    // Try Lattice detection first (more reliable when applicable)
    if options.mode != ExtractionMode::StreamOnly {
        let lattice_tables = extract_lattice_tables(&layouts, source_file, options);
        for table in lattice_tables {
            if passes_quality_filter(&table, options) {
                all_tables.push(table);
            }
        }
    }

    // Then try Stream detection for pages without Lattice tables
    if options.mode != ExtractionMode::LatticeOnly {
        let pages_with_lattice: std::collections::HashSet<u32> = all_tables
            .iter()
            .flat_map(|t| t.page_start..=t.page_end)
            .collect();

        let stream_layouts: Vec<&PageLayout> = layouts
            .iter()
            .filter(|l| !pages_with_lattice.contains(&l.page_number))
            .collect();

        let stream_tables = extract_stream_tables(&stream_layouts, source_file, options);
        for table in stream_tables {
            if passes_quality_filter(&table, options) {
                all_tables.push(table);
            }
        }
    }

    // Fallback: Line-based detection when Stream fails (for lopdf linearized text)
    // This detects tables from consecutive lines with key-value patterns
    if all_tables.is_empty() && options.mode != ExtractionMode::LatticeOnly {
        let line_tables = extract_line_based_tables(bytes, source_file, options);
        for table in line_tables {
            if passes_quality_filter(&table, options) {
                all_tables.push(table);
            }
        }
    }

    // Merge multi-page tables if enabled
    if options.merge_multi_page && all_tables.len() > 1 {
        all_tables = super::multi_page::merge_multi_page_tables(all_tables, options);
    }

    // Sort by page order
    all_tables.sort_by_key(|t| (t.page_start, t.page_end));

    // Generate unique IDs
    for (i, table) in all_tables.iter_mut().enumerate() {
        if table.table_id.is_empty() {
            table.table_id = format!("tbl_{}_{}", source_file.replace('.', "_"), i + 1);
        }
    }

    let total_ms = start.elapsed().as_millis() as u64;

    if all_tables.is_empty() && pages_processed > 0 {
        warnings.push("No tables detected in document".to_string());
    }

    Ok(TableExtractionResult {
        tables: all_tables,
        pages_processed,
        total_ms,
        warnings,
    })
}

/// Check if a table passes the quality filter.
fn passes_quality_filter(table: &ExtractedTable, options: &TableExtractionOptions) -> bool {
    // Check minimum dimensions
    if table.n_rows < options.min_rows || table.n_cols < options.min_cols {
        return false;
    }

    // Check quality threshold
    match (table.quality, options.min_quality) {
        (TableQuality::High, _) => true,
        (TableQuality::Medium, TableQuality::High) => false,
        (TableQuality::Medium, _) => true,
        (TableQuality::Low, TableQuality::Low) => true,
        (TableQuality::Low, _) => {
            // In aggressive mode, allow low quality with warnings
            options.mode == ExtractionMode::Aggressive
        }
    }
}

// ============================================================================
// Lattice Detection (Ruled Tables)
// ============================================================================

/// A grid cell defined by boundary lines.
#[derive(Debug, Clone)]
struct GridCell {
    row: usize,
    col: usize,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
}

/// Extract tables using grid line detection (Lattice mode).
fn extract_lattice_tables(
    layouts: &[PageLayout],
    source_file: &str,
    options: &TableExtractionOptions,
) -> Vec<ExtractedTable> {
    let mut tables = Vec::new();

    for layout in layouts {
        // Get horizontal and vertical lines
        let h_lines: Vec<f32> = layout
            .horizontal_lines(options.row_clustering_threshold)
            .iter()
            .filter(|l| l.length() >= MIN_LINE_LENGTH)
            .map(|l| l.y_coord())
            .collect();

        let v_lines: Vec<f32> = layout
            .vertical_lines(options.col_clustering_threshold)
            .iter()
            .filter(|l| l.length() >= MIN_LINE_LENGTH)
            .map(|l| l.x_coord())
            .collect();

        if h_lines.len() < 2 || v_lines.len() < 2 {
            continue;
        }

        // Cluster lines to find grid boundaries
        let h_clusters = cluster_values(&h_lines, options.row_clustering_threshold);
        let v_clusters = cluster_values(&v_lines, options.col_clustering_threshold);

        if h_clusters.len() < 2 || v_clusters.len() < 2 {
            continue;
        }

        // Build grid cells
        let grid = build_grid_cells(&h_clusters, &v_clusters);
        if grid.len() < MIN_GRID_INTERSECTIONS {
            continue;
        }

        // Assign text to cells
        let cell_contents = assign_text_to_cells(&layout.text_boxes, &grid);

        // Build table from grid
        if let Some(table) = build_table_from_grid(
            &grid,
            &cell_contents,
            layout.page_number,
            source_file,
            options,
        ) {
            tables.push(table);
        }
    }

    tables
}

/// Build grid cells from clustered line positions.
fn build_grid_cells(h_clusters: &[f32], v_clusters: &[f32]) -> Vec<GridCell> {
    let mut cells = Vec::new();

    // Sort clusters (Y descending for PDF coordinates, X ascending)
    let mut h_sorted: Vec<f32> = h_clusters.to_vec();
    h_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut v_sorted: Vec<f32> = v_clusters.to_vec();
    v_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for (row_idx, h_pair) in h_sorted.windows(2).enumerate() {
        for (col_idx, v_pair) in v_sorted.windows(2).enumerate() {
            cells.push(GridCell {
                row: row_idx,
                col: col_idx,
                x_min: v_pair[0],
                x_max: v_pair[1],
                y_min: h_pair[1], // Lower Y (remember PDF coords)
                y_max: h_pair[0], // Higher Y
            });
        }
    }

    cells
}

/// Assign text boxes to grid cells based on position.
fn assign_text_to_cells(
    text_boxes: &[TextBox],
    grid: &[GridCell],
) -> HashMap<(usize, usize), String> {
    let mut cell_text: HashMap<(usize, usize), Vec<String>> = HashMap::new();

    for tbox in text_boxes {
        let center_x = tbox.center_x();
        let center_y = tbox.center_y();

        for cell in grid {
            if center_x >= cell.x_min
                && center_x <= cell.x_max
                && center_y >= cell.y_min
                && center_y <= cell.y_max
            {
                cell_text
                    .entry((cell.row, cell.col))
                    .or_default()
                    .push(tbox.text.trim().to_string());
                break;
            }
        }
    }

    // Join text fragments in each cell
    cell_text
        .into_iter()
        .map(|(k, v)| (k, v.join(" ")))
        .collect()
}

/// Build an ExtractedTable from grid data.
fn build_table_from_grid(
    grid: &[GridCell],
    cell_contents: &HashMap<(usize, usize), String>,
    page: u32,
    source_file: &str,
    _options: &TableExtractionOptions,
) -> Option<ExtractedTable> {
    if grid.is_empty() {
        return None;
    }

    let max_row = grid.iter().map(|c| c.row).max().unwrap_or(0);
    let max_col = grid.iter().map(|c| c.col).max().unwrap_or(0);

    let n_rows = max_row + 1;
    let n_cols = max_col + 1;

    // Build rows
    let mut rows = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let mut cells = Vec::with_capacity(n_cols);
        for col_idx in 0..n_cols {
            let text = cell_contents
                .get(&(row_idx, col_idx))
                .cloned()
                .unwrap_or_default();
            cells.push(TableCell::new(text, col_idx));
        }
        rows.push(TableRow::new(row_idx, page, cells));
    }

    // Detect headers (first row with content)
    let mut headers = Vec::new();
    if let Some(first_row) = rows.first() {
        headers = first_row
            .cell_texts()
            .iter()
            .map(|s| s.to_string())
            .collect();
        if headers.iter().any(|h| !h.is_empty()) {
            if let Some(row) = rows.first_mut() {
                *row = std::mem::take(row).as_header();
            }
        }
    }

    let mut table = ExtractedTable::new(String::new(), source_file);
    table.page_start = page;
    table.page_end = page;
    table.headers = headers;
    table.n_cols = n_cols;
    table.n_rows = rows.iter().filter(|r| !r.is_header_row).count();
    table.rows = rows;
    table.detection_mode = DetectionMode::Lattice;
    table.quality = TableQuality::High; // Lattice detection is most reliable
    table.confidence_score = 0.9;

    Some(table)
}

// ============================================================================
// Stream Detection (Whitespace-Aligned Tables)
// ============================================================================

/// Extract tables using text alignment inference (Stream mode).
fn extract_stream_tables(
    layouts: &[&PageLayout],
    source_file: &str,
    options: &TableExtractionOptions,
) -> Vec<ExtractedTable> {
    let mut tables = Vec::new();

    for layout in layouts {
        if layout.text_boxes.is_empty() {
            continue;
        }

        // Cluster text boxes into rows by Y-position
        let rows = cluster_into_rows(&layout.text_boxes, options.row_clustering_threshold);

        if rows.len() < options.min_rows {
            continue;
        }

        // Detect column boundaries from X-positions
        let col_boundaries = detect_column_boundaries(&rows, options.col_clustering_threshold);

        if col_boundaries.len() < options.min_cols + 1 {
            continue;
        }

        // Build table from detected structure
        if let Some(table) = build_stream_table(
            &rows,
            &col_boundaries,
            layout.page_number,
            source_file,
            options,
        ) {
            tables.push(table);
        }
    }

    tables
}

/// Cluster text boxes into rows by Y-position.
fn cluster_into_rows<'a>(text_boxes: &'a [TextBox], threshold: f32) -> Vec<Vec<&'a TextBox>> {
    if text_boxes.is_empty() {
        return Vec::new();
    }

    // Sort by Y-position (descending for PDF coordinates - top to bottom)
    let mut sorted: Vec<&TextBox> = text_boxes.iter().collect();
    sorted.sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(std::cmp::Ordering::Equal));

    let mut rows: Vec<Vec<&TextBox>> = Vec::new();
    let mut current_row = vec![sorted[0]];
    let mut current_y = sorted[0].y;

    for tbox in &sorted[1..] {
        if (current_y - tbox.y).abs() <= threshold {
            current_row.push(tbox);
        } else {
            // Sort row by X-position (left to right)
            current_row.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
            rows.push(current_row);
            current_row = vec![tbox];
            current_y = tbox.y;
        }
    }

    if !current_row.is_empty() {
        current_row.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        rows.push(current_row);
    }

    rows
}

/// Detect column boundaries from consistent X-positions across rows.
fn detect_column_boundaries(rows: &[Vec<&TextBox>], threshold: f32) -> Vec<f32> {
    // Collect all X-positions (left edges and right edges)
    let mut x_positions: Vec<f32> = Vec::new();

    for row in rows {
        for tbox in row {
            x_positions.push(tbox.x);
            x_positions.push(tbox.right());
        }
    }

    // Cluster X-positions
    let candidates = cluster_values(&x_positions, threshold);

    // Filter to keep only boundaries that appear consistently
    let min_occurrences = rows.len() / 2;
    filter_consistent_boundaries(&candidates, rows, threshold, min_occurrences)
}

/// Filter column boundaries to keep only those appearing consistently.
fn filter_consistent_boundaries(
    candidates: &[f32],
    rows: &[Vec<&TextBox>],
    threshold: f32,
    min_occurrences: usize,
) -> Vec<f32> {
    candidates
        .iter()
        .filter(|&&boundary| {
            let occurrences = rows
                .iter()
                .filter(|row| {
                    row.iter().any(|tbox| {
                        (tbox.x - boundary).abs() <= threshold
                            || (tbox.right() - boundary).abs() <= threshold
                    })
                })
                .count();
            occurrences >= min_occurrences
        })
        .copied()
        .collect()
}

/// Build an ExtractedTable from stream-detected structure.
fn build_stream_table(
    text_rows: &[Vec<&TextBox>],
    col_boundaries: &[f32],
    page: u32,
    source_file: &str,
    _options: &TableExtractionOptions,
) -> Option<ExtractedTable> {
    if text_rows.is_empty() || col_boundaries.len() < 2 {
        return None;
    }

    let n_cols = col_boundaries.len() - 1;
    let mut rows = Vec::with_capacity(text_rows.len());

    for (row_idx, text_row) in text_rows.iter().enumerate() {
        let mut cells = vec![TableCell::new(String::new(), 0); n_cols];

        for tbox in text_row {
            let center_x = tbox.center_x();

            // Find which column this text belongs to
            for (col_idx, col_pair) in col_boundaries.windows(2).enumerate() {
                if center_x >= col_pair[0] && center_x <= col_pair[1] {
                    // Append text to this cell
                    if !cells[col_idx].text.is_empty() {
                        cells[col_idx].text.push(' ');
                    }
                    cells[col_idx].text.push_str(tbox.text.trim());
                    cells[col_idx].col_index = col_idx;
                    break;
                }
            }
        }

        rows.push(TableRow::new(row_idx, page, cells));
    }

    // Detect headers (first row)
    let mut headers = Vec::new();
    if let Some(first_row) = rows.first() {
        headers = first_row
            .cell_texts()
            .iter()
            .map(|s| s.to_string())
            .collect();
        // Check if first row looks like headers (non-empty, possibly different formatting)
        let non_empty_count = headers.iter().filter(|h| !h.is_empty()).count();
        if non_empty_count > n_cols / 2 {
            if let Some(row) = rows.first_mut() {
                *row = std::mem::take(row).as_header();
            }
        }
    }

    // Calculate quality based on consistency
    let (quality, confidence) = calculate_stream_quality(&rows, n_cols);

    let mut table = ExtractedTable::new(String::new(), source_file);
    table.page_start = page;
    table.page_end = page;
    table.headers = headers;
    table.n_cols = n_cols;
    table.n_rows = rows.iter().filter(|r| !r.is_header_row).count();
    table.rows = rows;
    table.detection_mode = DetectionMode::Stream;
    table.quality = quality;
    table.confidence_score = confidence;

    Some(table)
}

/// Calculate quality score for stream-detected tables.
fn calculate_stream_quality(rows: &[TableRow], expected_cols: usize) -> (TableQuality, f32) {
    if rows.is_empty() {
        return (TableQuality::Low, 0.0);
    }

    let mut score = 1.0f32;

    // Penalty for inconsistent column counts
    let col_counts: Vec<usize> = rows
        .iter()
        .map(|r| r.cells.iter().filter(|c| !c.text.is_empty()).count())
        .collect();

    let avg_cols = col_counts.iter().sum::<usize>() as f32 / col_counts.len() as f32;
    let variance: f32 = col_counts
        .iter()
        .map(|&c| (c as f32 - avg_cols).powi(2))
        .sum::<f32>()
        / col_counts.len() as f32;

    if variance > 1.0 {
        score -= 0.2 * variance.min(2.0);
    }

    // Penalty for many empty cells
    let total_cells = rows.len() * expected_cols;
    let empty_cells = rows
        .iter()
        .flat_map(|r| &r.cells)
        .filter(|c| c.text.is_empty())
        .count();
    let empty_ratio = empty_cells as f32 / total_cells.max(1) as f32;
    if empty_ratio > 0.3 {
        score -= 0.2 * empty_ratio;
    }

    // Penalty for few rows
    if rows.len() < 4 {
        score -= 0.1;
    }

    // Stream detection inherently less reliable than Lattice
    score -= 0.1;

    score = score.clamp(0.0, 1.0);

    let quality = if score >= 0.7 {
        TableQuality::High
    } else if score >= 0.4 {
        TableQuality::Medium
    } else {
        TableQuality::Low
    };

    (quality, score)
}

// ============================================================================
// Line-Based Detection (Fallback for linearized PDF text)
// ============================================================================

/// A detected table region from line patterns.
#[derive(Debug)]
struct LineBasedTableRegion {
    /// Header line (if detected)
    header: Option<String>,
    /// Data rows as raw lines
    data_lines: Vec<String>,
    /// Detected column count
    col_count: usize,
    /// Starting line index in the document (reserved for future use)
    #[allow(dead_code)]
    start_line: usize,
}

/// Extract tables from linearized PDF text using line patterns.
///
/// This fallback works when lopdf's text extraction returns each cell on
/// separate lines (common with wkhtmltopdf, some PDF generators).
/// It detects tables by identifying:
/// 1. Key-value patterns (Label followed by Value)
/// 2. Currency/numeric patterns common in invoices/pay stubs
/// 3. Section headers followed by consistent data rows
fn extract_line_based_tables(
    bytes: &[u8],
    source_file: &str,
    options: &TableExtractionOptions,
) -> Vec<ExtractedTable> {
    // Extract raw text from PDF
    let text = match extract_raw_text(bytes) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < options.min_rows {
        return Vec::new();
    }

    let mut tables = Vec::new();

    // Detect table regions by identifying patterns
    let regions = detect_table_regions(&lines, options);

    for region in regions {
        if let Some(table) = build_line_based_table(region, source_file, options) {
            tables.push(table);
        }
    }

    tables
}

/// Extract raw text from PDF bytes using lopdf.
fn extract_raw_text(bytes: &[u8]) -> Option<String> {
    use lopdf::Document;

    let document = Document::load_mem(bytes).ok()?;
    let pages = document.get_pages();
    let mut all_text = String::new();

    for page_num in 1..=pages.len() as u32 {
        if let Ok(text) = document.extract_text(&[page_num]) {
            all_text.push_str(&text);
            all_text.push('\n');
        }
    }

    if all_text.is_empty() {
        None
    } else {
        Some(all_text)
    }
}

/// Detect table regions from line patterns.
fn detect_table_regions<'a>(
    lines: &'a [&'a str],
    options: &TableExtractionOptions,
) -> Vec<LineBasedTableRegion> {
    let mut regions = Vec::new();

    // Patterns for detecting table-like content
    let currency_re = Regex::new(r"^\$?[\d,]+\.?\d*$").unwrap();
    let date_re = Regex::new(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$").unwrap();
    let percent_re = Regex::new(r"^\d+\.?\d*%$").unwrap();
    let hours_re = Regex::new(r"^\d+\.?\d*\s*(hrs?|hours?)$").unwrap();

    // Common table section headers (case-insensitive patterns)
    let section_headers = [
        "earnings",
        "deductions",
        "taxes",
        "withheld",
        "summary",
        "totals",
        "gross",
        "net",
        "employee",
        "employer",
        "description",
        "amount",
        "rate",
        "hours",
        "pay",
        "date",
        "period",
        "ytd",
        "current",
        "item",
        "quantity",
        "price",
        "total",
        "subtotal",
        "invoice",
        "bill",
    ];

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Skip empty lines
        if line.is_empty() {
            i += 1;
            continue;
        }

        // Check if this might be a table header
        let line_lower = line.to_lowercase();
        let is_potential_header = section_headers.iter().any(|h| line_lower.contains(h));

        if is_potential_header {
            // Look for data rows following this header
            let (region, consumed) = collect_table_region(
                lines,
                i,
                Some(line.to_string()),
                &currency_re,
                &date_re,
                &percent_re,
                &hours_re,
                options,
            );

            if let Some(r) = region {
                if r.data_lines.len() >= options.min_rows {
                    regions.push(r);
                }
            }

            i += consumed.max(1);
        } else {
            // Check if current line looks like tabular data
            let looks_like_data =
                is_tabular_data_line(line, &currency_re, &date_re, &percent_re, &hours_re);

            if looks_like_data {
                // Try to collect a table region without explicit header
                let (region, consumed) = collect_table_region(
                    lines,
                    i,
                    None,
                    &currency_re,
                    &date_re,
                    &percent_re,
                    &hours_re,
                    options,
                );

                if let Some(r) = region {
                    if r.data_lines.len() >= options.min_rows {
                        regions.push(r);
                    }
                }

                i += consumed.max(1);
            } else {
                i += 1;
            }
        }
    }

    regions
}

/// Check if a line looks like tabular data.
fn is_tabular_data_line(
    line: &str,
    currency_re: &Regex,
    date_re: &Regex,
    percent_re: &Regex,
    hours_re: &Regex,
) -> bool {
    let trimmed = line.trim();

    // Check for currency values
    if currency_re.is_match(trimmed) {
        return true;
    }

    // Check for dates
    if date_re.is_match(trimmed) {
        return true;
    }

    // Check for percentages
    if percent_re.is_match(trimmed) {
        return true;
    }

    // Check for hours
    if hours_re.is_match(trimmed) {
        return true;
    }

    // Check for numeric values
    if trimmed.parse::<f64>().is_ok() {
        return true;
    }

    false
}

/// Collect consecutive lines that form a table region.
fn collect_table_region(
    lines: &[&str],
    start: usize,
    header: Option<String>,
    currency_re: &Regex,
    date_re: &Regex,
    _percent_re: &Regex,
    _hours_re: &Regex,
    options: &TableExtractionOptions,
) -> (Option<LineBasedTableRegion>, usize) {
    let mut data_lines = Vec::new();
    let mut i = if header.is_some() { start + 1 } else { start };
    let mut consecutive_non_data = 0;
    let max_gap = 2; // Allow small gaps in data

    while i < lines.len() && consecutive_non_data <= max_gap {
        let line = lines[i].trim();

        if line.is_empty() {
            consecutive_non_data += 1;
            i += 1;
            continue;
        }

        // Check if this is a new section header (end of current table)
        let line_lower = line.to_lowercase();
        let is_new_section = ["earnings", "deductions", "taxes", "summary", "totals"]
            .iter()
            .any(|h| line_lower.starts_with(h) && !data_lines.is_empty());

        if is_new_section {
            break;
        }

        // Collect the line as data
        data_lines.push(line.to_string());
        consecutive_non_data = 0;
        i += 1;
    }

    let consumed = i - start;

    if data_lines.len() < options.min_rows {
        return (None, consumed);
    }

    // Try to determine column structure
    let col_count = infer_column_count(&data_lines, currency_re, date_re);

    if col_count < options.min_cols {
        return (None, consumed);
    }

    (
        Some(LineBasedTableRegion {
            header,
            data_lines,
            col_count,
            start_line: start,
        }),
        consumed,
    )
}

/// Infer the number of columns from data patterns.
fn infer_column_count(lines: &[String], currency_re: &Regex, _date_re: &Regex) -> usize {
    // For key-value style tables (common in pay stubs), assume 2 columns
    // Look for patterns: Label on one line, Value on next

    let mut consecutive_label_value = 0;
    let mut i = 0;

    while i + 1 < lines.len() {
        let line1 = lines[i].trim();
        let line2 = lines[i + 1].trim();

        // Check if line1 is a label (text) and line2 is a value (numeric/currency)
        let line1_is_label =
            !line1.is_empty() && !currency_re.is_match(line1) && line1.parse::<f64>().is_err();

        let line2_is_value =
            currency_re.is_match(line2) || line2.parse::<f64>().is_ok() || line2.contains('$');

        if line1_is_label && line2_is_value {
            consecutive_label_value += 1;
        }

        i += 2;
    }

    // If we found many label-value pairs, it's a 2-column table
    if consecutive_label_value >= 3 {
        return 2;
    }

    // Default: single column (or need more sophisticated analysis)
    // Check if lines contain tab or multiple spaces for column detection
    let mut max_parts = 1;
    for line in lines {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() > max_parts {
            max_parts = parts.len();
        }

        // Also check for "  " (double space) separation
        let space_parts: Vec<&str> = line.split("  ").filter(|s| !s.is_empty()).collect();
        if space_parts.len() > max_parts {
            max_parts = space_parts.len();
        }
    }

    max_parts.max(2) // Assume at least 2 columns for table detection
}

/// Build an ExtractedTable from a line-based region.
fn build_line_based_table(
    region: LineBasedTableRegion,
    source_file: &str,
    _options: &TableExtractionOptions,
) -> Option<ExtractedTable> {
    if region.data_lines.is_empty() {
        return None;
    }

    let mut rows = Vec::new();
    let mut headers = Vec::new();

    // If we have a header, use it
    if let Some(ref h) = region.header {
        headers = vec![h.clone(), "Value".to_string()];
        let header_row = TableRow::new(
            0,
            1,
            vec![
                TableCell::new(h.clone(), 0).as_header(),
                TableCell::new("Value", 1).as_header(),
            ],
        )
        .as_header();
        rows.push(header_row);
    }

    // Group lines into rows (label + value pairs for 2-col tables)
    let mut row_idx = rows.len();
    let currency_re = Regex::new(r"^\$?[\d,]+\.?\d*$").unwrap();

    let mut i = 0;
    while i < region.data_lines.len() {
        let line1 = region.data_lines[i].trim();

        // Try to detect label-value pair
        let is_label = !currency_re.is_match(line1) && line1.parse::<f64>().is_err();

        if is_label && i + 1 < region.data_lines.len() {
            let line2 = region.data_lines[i + 1].trim();
            let is_value =
                currency_re.is_match(line2) || line2.parse::<f64>().is_ok() || line2.contains('$');

            if is_value {
                // Create 2-column row
                let cells = vec![TableCell::new(line1, 0), TableCell::new(line2, 1)];
                rows.push(TableRow::new(row_idx, 1, cells));
                row_idx += 1;
                i += 2;
                continue;
            }
        }

        // Single cell row
        let cells = vec![TableCell::new(line1, 0)];
        rows.push(TableRow::new(row_idx, 1, cells));
        row_idx += 1;
        i += 1;
    }

    if rows.is_empty() {
        return None;
    }

    // Set default headers if not already set
    if headers.is_empty() {
        headers = vec!["Label".to_string(), "Value".to_string()];
    }

    let n_cols = region.col_count.max(2);
    let n_rows = rows.iter().filter(|r| !r.is_header_row).count();

    let mut table = ExtractedTable::new(String::new(), source_file);
    table.page_start = 1;
    table.page_end = 1;
    table.headers = headers;
    table.n_cols = n_cols;
    table.n_rows = n_rows;
    table.rows = rows;
    table.detection_mode = DetectionMode::LineBased;
    table.quality = TableQuality::Medium; // Lower confidence for line-based
    table.confidence_score = 0.6;

    Some(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_into_rows() {
        let boxes = vec![
            TextBox {
                text: "A".to_string(),
                x: 10.0,
                y: 100.0,
                width: 20.0,
                height: 10.0,
                font_size: 12.0,
                page: 1,
            },
            TextBox {
                text: "B".to_string(),
                x: 50.0,
                y: 100.0,
                width: 20.0,
                height: 10.0,
                font_size: 12.0,
                page: 1,
            },
            TextBox {
                text: "C".to_string(),
                x: 10.0,
                y: 80.0,
                width: 20.0,
                height: 10.0,
                font_size: 12.0,
                page: 1,
            },
        ];

        let rows = cluster_into_rows(&boxes, 5.0);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].len(), 2); // A and B on same row
        assert_eq!(rows[1].len(), 1); // C on second row
    }

    #[test]
    fn test_build_grid_cells() {
        let h_clusters = vec![100.0, 80.0, 60.0];
        let v_clusters = vec![10.0, 50.0, 90.0];

        let grid = build_grid_cells(&h_clusters, &v_clusters);
        assert_eq!(grid.len(), 4); // 2 rows × 2 cols
    }

    #[test]
    fn test_stream_quality_calculation() {
        let rows = vec![
            TableRow::new(
                0,
                1,
                vec![
                    TableCell::new("A", 0),
                    TableCell::new("B", 1),
                    TableCell::new("C", 2),
                ],
            ),
            TableRow::new(
                1,
                1,
                vec![
                    TableCell::new("1", 0),
                    TableCell::new("2", 1),
                    TableCell::new("3", 2),
                ],
            ),
        ];

        let (quality, score) = calculate_stream_quality(&rows, 3);
        assert!(score > 0.5);
        assert!(matches!(quality, TableQuality::Medium | TableQuality::High));
    }
}
