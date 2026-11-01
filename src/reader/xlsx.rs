use std::io::Cursor;

use calamine::{Data, Reader as CalamineReader, Xlsx};

use super::xlsx_chunker::{XlsxChunkingOptions, chunk_workbook, generate_flat_text};
use super::xlsx_ooxml::{OoxmlMetadata, parse_ooxml_metadata};
use super::xlsx_table_detect::{CellValue, DetectedTable, SheetGrid, detect_tables};
use crate::{
    DocumentFormat, DocumentReader, PassthroughReader, ReaderDiagnostics, ReaderHint, ReaderOutput,
    Result, types::structure::ChunkingResult,
};

/// Result of the structured XLSX extraction pipeline.
pub struct XlsxStructuredResult {
    /// Backward-compatible flat text.
    pub text: String,
    /// Detected tables with metadata.
    pub tables: Vec<DetectedTable>,
    /// Semantic chunks with header-value pairing.
    pub chunks: ChunkingResult,
    /// OOXML metadata (number formats, merged regions, etc.).
    pub metadata: OoxmlMetadata,
    /// Extraction diagnostics.
    pub diagnostics: XlsxStructuredDiagnostics,
}

/// Diagnostics from structured extraction.
pub struct XlsxStructuredDiagnostics {
    pub warnings: Vec<String>,
}

pub struct XlsxReader;

impl XlsxReader {
    /// Build `SheetGrid`s from raw XLSX bytes using calamine.
    fn build_grids(bytes: &[u8]) -> Result<Vec<SheetGrid>> {
        let cursor = Cursor::new(bytes);
        let mut workbook =
            Xlsx::new(cursor).map_err(|err| crate::MemvidError::ExtractionFailed {
                reason: format!("failed to read xlsx workbook: {err}").into(),
            })?;

        let sheet_names: Vec<String> = workbook.sheet_names().clone();
        let mut grids = Vec::new();

        for sheet_name in &sheet_names {
            let Ok(range) = workbook.worksheet_range(sheet_name) else {
                continue;
            };

            let mut grid = SheetGrid::new(sheet_name.clone());
            #[allow(clippy::cast_possible_truncation)]
            let num_rows = range.height() as u32;
            #[allow(clippy::cast_possible_truncation)]
            let num_cols = range.width() as u32;

            for row in range.rows() {
                let cells: Vec<CellValue> = row
                    .iter()
                    .map(|cell| match cell {
                        Data::String(s) => CellValue::Text(s.clone()),
                        Data::Float(v) => CellValue::Number(*v),
                        Data::Int(v) => CellValue::Integer(*v),
                        Data::Bool(b) => CellValue::Boolean(*b),
                        Data::DateTime(v) => CellValue::Number(v.as_f64()),
                        Data::DateTimeIso(s) => CellValue::DateTime(s.clone()),
                        Data::DurationIso(s) => CellValue::Text(s.clone()),
                        Data::Error(e) => CellValue::Error(format!("#{e:?}")),
                        Data::Empty => CellValue::Empty,
                    })
                    .collect();
                grid.rows.push(cells);
            }

            grid.num_rows = num_rows;
            grid.num_cols = num_cols;
            grids.push(grid);
        }

        Ok(grids)
    }

    /// Extract structured data from XLSX bytes with default options.
    pub fn extract_structured(bytes: &[u8]) -> Result<XlsxStructuredResult> {
        Self::extract_structured_with_options(bytes, XlsxChunkingOptions::default())
    }

    /// Extract structured data from XLSX bytes with custom chunking options.
    pub fn extract_structured_with_options(
        bytes: &[u8],
        options: XlsxChunkingOptions,
    ) -> Result<XlsxStructuredResult> {
        let grids = Self::build_grids(bytes)?;
        let metadata = parse_ooxml_metadata(bytes).unwrap_or_default();

        let mut all_tables = Vec::new();
        let mut warnings = Vec::new();

        for grid in &grids {
            let sheet_merged = metadata
                .merged_regions
                .get(&grid.sheet_name)
                .cloned()
                .unwrap_or_default();
            let sheet_ooxml_tables: Vec<_> = metadata
                .table_defs
                .iter()
                .filter(|t| t.sheet_name == grid.sheet_name)
                .cloned()
                .collect();

            let tables = detect_tables(grid, &sheet_ooxml_tables, &sheet_merged);
            if tables.is_empty() {
                warnings.push(format!("No tables detected in sheet '{}'", grid.sheet_name));
            }
            all_tables.extend(tables);
        }

        let chunks = chunk_workbook(&grids, &all_tables, &metadata, &options);
        let text = generate_flat_text(&grids, &all_tables, &metadata);

        // Merge chunker warnings
        warnings.extend(chunks.warnings.iter().cloned());

        Ok(XlsxStructuredResult {
            text,
            tables: all_tables,
            chunks,
            metadata,
            diagnostics: XlsxStructuredDiagnostics { warnings },
        })
    }

    fn extract_text(bytes: &[u8]) -> Result<String> {
        let cursor = Cursor::new(bytes);
        let mut workbook =
            Xlsx::new(cursor).map_err(|err| crate::MemvidError::ExtractionFailed {
                reason: format!("failed to read xlsx workbook: {err}").into(),
            })?;

        let mut out = String::new();
        for sheet_name in workbook.sheet_names().clone() {
            if let Ok(range) = workbook.worksheet_range(&sheet_name) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!("Sheet: {sheet_name}\n"));
                for row in range.rows() {
                    let mut first_cell = true;
                    for cell in row {
                        if !first_cell {
                            out.push('\t');
                        }
                        first_cell = false;
                        match cell {
                            Data::String(s) => out.push_str(s.trim()),
                            Data::Float(v) => out.push_str(&format!("{v}")),
                            Data::Int(v) => out.push_str(&format!("{v}")),
                            Data::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
                            Data::Error(e) => out.push_str(&format!("#{e:?}")),
                            Data::Empty => {}
                            Data::DateTime(v) => out.push_str(&format!("{v}")),
                            Data::DateTimeIso(s) => out.push_str(s),
                            Data::DurationIso(s) => out.push_str(s),
                        }
                    }
                    out.push('\n');
                }
            }
        }

        Ok(out.trim().to_string())
    }
}

impl DocumentReader for XlsxReader {
    fn name(&self) -> &'static str {
        "xlsx"
    }

    fn supports(&self, hint: &ReaderHint<'_>) -> bool {
        matches!(hint.format, Some(DocumentFormat::Xlsx))
            || hint.mime.is_some_and(|mime| {
                mime.eq_ignore_ascii_case(
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            })
    }

    fn extract(&self, bytes: &[u8], hint: &ReaderHint<'_>) -> Result<ReaderOutput> {
        match Self::extract_text(bytes) {
            Ok(text) => {
                if text.trim().is_empty() {
                    // Calamine returned empty - try extractous as fallback
                    let mut fallback = PassthroughReader.extract(bytes, hint)?;
                    fallback.reader_name = self.name().to_string();
                    fallback.diagnostics.mark_fallback();
                    fallback.diagnostics.record_warning(
                        "xlsx reader produced empty text; falling back to default extractor",
                    );
                    Ok(fallback)
                } else {
                    // Calamine succeeded - build output directly WITHOUT calling extractous
                    let mut document = crate::ExtractedDocument::empty();
                    document.text = Some(text);
                    document.mime_type = Some(
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            .to_string(),
                    );
                    Ok(ReaderOutput::new(document, self.name())
                        .with_diagnostics(ReaderDiagnostics::default()))
                }
            }
            Err(err) => {
                // Calamine failed - try extractous as fallback
                let mut fallback = PassthroughReader.extract(bytes, hint)?;
                fallback.reader_name = self.name().to_string();
                fallback.diagnostics.mark_fallback();
                fallback
                    .diagnostics
                    .record_warning(format!("xlsx reader error: {err}"));
                Ok(fallback)
            }
        }
    }
}
