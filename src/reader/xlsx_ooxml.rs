//! OOXML metadata parser for XLSX files.
//!
//! Extracts metadata that calamine cannot provide:
//! - Number format classification (dates, currency, percentages)
//! - Merged cell regions
//! - Named table definitions
//!
//! Parses XML files from the XLSX zip:
//! - `xl/styles.xml` → number formats + cell XF mappings
//! - `xl/worksheets/sheetN.xml` → merged cell regions
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::collections::HashMap;
use std::io::{Cursor, Read};

use quick_xml::Reader as XmlReader;
use quick_xml::XmlVersion;
use quick_xml::events::Event;
use serde::{Deserialize, Serialize};
use zip::ZipArchive;

use crate::Result;

/// Classification of Excel number formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NumFmtKind {
    #[default]
    General,
    Number,
    Date,
    Time,
    DateTime,
    Currency,
    Percentage,
    Scientific,
    Text,
}

/// A merged cell region in a worksheet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedRegion {
    pub top_row: u32,
    pub left_col: u32,
    pub bottom_row: u32,
    pub right_col: u32,
}

/// A named table definition from OOXML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableDefinition {
    pub name: String,
    pub sheet_name: String,
    pub headers: Vec<String>,
    pub first_row: u32,
    pub last_row: u32,
    pub first_col: u32,
    pub last_col: u32,
}

/// All OOXML metadata extracted from an XLSX file.
#[derive(Debug, Clone, Default)]
pub struct OoxmlMetadata {
    /// Map from numFmtId → format kind
    pub num_fmts: HashMap<u32, NumFmtKind>,
    /// Cell XF entries: index → numFmtId (the cellXfs array from styles.xml)
    pub cell_xfs: Vec<u32>,
    /// Merged regions per sheet name
    pub merged_regions: HashMap<String, Vec<MergedRegion>>,
    /// Named table definitions
    pub table_defs: Vec<TableDefinition>,
}

impl OoxmlMetadata {
    /// Get the number format kind for a cell XF index (the `s` attribute on `<c>` elements).
    #[must_use]
    pub fn num_fmt_for_xf(&self, xf_index: u32) -> NumFmtKind {
        self.cell_xfs
            .get(xf_index as usize)
            .and_then(|fmt_id| self.num_fmts.get(fmt_id))
            .copied()
            .unwrap_or(NumFmtKind::General)
    }
}

/// Classify a built-in Excel number format ID.
///
/// Excel reserves IDs 0-163 for built-in formats. The key date/time/currency ranges:
/// - 0: General
/// - 1-11: Number formats
/// - 14-22: Date/Time formats
/// - 37-44: Accounting/Currency
/// - 45-48: Time/Duration
/// - 49: Text (@)
fn classify_builtin_fmt(id: u32) -> NumFmtKind {
    match id {
        0 => NumFmtKind::General,
        1..=4 | 37..=40 => NumFmtKind::Number,
        5..=8 | 41..=44 => NumFmtKind::Currency,
        9 | 10 => NumFmtKind::Percentage,
        11 => NumFmtKind::Scientific,
        14..=17 => NumFmtKind::Date,
        18..=21 => NumFmtKind::Time,
        22 => NumFmtKind::DateTime,
        45..=48 => NumFmtKind::Time,
        49 => NumFmtKind::Text,
        _ => NumFmtKind::General,
    }
}

/// Classify a custom format code string by inspecting its characters.
fn classify_format_code(code: &str) -> NumFmtKind {
    let lower = code.to_ascii_lowercase();
    // Remove escaped sequences and quoted strings
    let cleaned = remove_quoted_sections(&lower);

    let has_date = cleaned.contains('y') || cleaned.contains('d');
    let has_month = cleaned.contains('m');
    let has_time = cleaned.contains('h') || cleaned.contains('s');
    let has_ampm = cleaned.contains("am/pm") || cleaned.contains("a/p");

    if has_date && has_time {
        return NumFmtKind::DateTime;
    }
    if has_date {
        return NumFmtKind::Date;
    }
    // 'm' alone with time indicators is minutes, not months
    if has_time || has_ampm {
        return NumFmtKind::Time;
    }
    // After ruling out date/time, check for m alone (month)
    if has_month && !cleaned.contains('#') && !cleaned.contains('0') {
        return NumFmtKind::Date;
    }

    if cleaned.contains('%') {
        return NumFmtKind::Percentage;
    }
    if cleaned.contains("e+") || cleaned.contains("e-") {
        return NumFmtKind::Scientific;
    }
    if cleaned.contains('$')
        || cleaned.contains('\u{20ac}')
        || cleaned.contains('\u{00a3}')
        || cleaned.contains('\u{00a5}')
        || cleaned.contains("eur")
        || cleaned.contains("usd")
        || cleaned.contains("gbp")
    {
        return NumFmtKind::Currency;
    }
    if cleaned.contains('@') {
        return NumFmtKind::Text;
    }
    if cleaned.contains('#') || cleaned.contains('0') {
        return NumFmtKind::Number;
    }

    NumFmtKind::General
}

/// Remove quoted sections (e.g., "text") and escaped chars (e.g., \x) from a format code.
fn remove_quoted_sections(code: &str) -> String {
    let mut result = String::with_capacity(code.len());
    let mut chars = code.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            // Skip until closing quote
            for c in chars.by_ref() {
                if c == '"' {
                    break;
                }
            }
        } else if ch == '\\' {
            // Skip next char (escaped literal)
            let _ = chars.next();
        } else {
            result.push(ch);
        }
    }
    result
}

/// Parse a cell reference like "A1" or "AZ100" into (row, col) 0-based.
#[must_use]
pub fn parse_cell_ref(cell_ref: &str) -> Option<(u32, u32)> {
    let mut col_str = String::new();
    let mut row_str = String::new();

    for ch in cell_ref.chars() {
        if ch.is_ascii_alphabetic() {
            col_str.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            row_str.push(ch);
        }
    }

    if col_str.is_empty() || row_str.is_empty() {
        return None;
    }

    let col = col_str
        .chars()
        .fold(0u32, |acc, c| acc * 26 + (c as u32 - b'A' as u32 + 1))
        .saturating_sub(1);
    let row = row_str.parse::<u32>().ok()?.saturating_sub(1);

    Some((row, col))
}

/// Parse a range reference like "A1:D10" into ((top_row, left_col), (bottom_row, right_col)).
#[must_use]
pub fn parse_range_ref(range_ref: &str) -> Option<((u32, u32), (u32, u32))> {
    let parts: Vec<&str> = range_ref.split(':').collect();
    if parts.len() != 2 {
        return None;
    }
    let start = parse_cell_ref(parts[0])?;
    let end = parse_cell_ref(parts[1])?;
    Some((start, end))
}

/// Extract OOXML metadata from an XLSX file's bytes.
///
/// Parses styles.xml for number formats and worksheet XMLs for merged cells.
/// Table definitions come from calamine's native table support (calamine 0.25+).
pub fn parse_ooxml_metadata(xlsx_bytes: &[u8]) -> Result<OoxmlMetadata> {
    let cursor = Cursor::new(xlsx_bytes);
    let mut archive =
        ZipArchive::new(cursor).map_err(|err| crate::MemvidError::ExtractionFailed {
            reason: format!("failed to open xlsx zip: {err}").into(),
        })?;

    let mut metadata = OoxmlMetadata::default();

    // Seed built-in formats
    for id in 0..=49 {
        let kind = classify_builtin_fmt(id);
        if kind != NumFmtKind::General {
            metadata.num_fmts.insert(id, kind);
        }
    }

    // Parse styles.xml
    if let Ok(styles_xml) = read_zip_entry(&mut archive, "xl/styles.xml") {
        parse_styles_xml(&styles_xml, &mut metadata);
    }

    // Parse worksheet XMLs for merged cells
    let sheet_names = collect_sheet_filenames(&mut archive);
    for (sheet_name, zip_path) in &sheet_names {
        if let Ok(sheet_xml) = read_zip_entry(&mut archive, zip_path) {
            let regions = parse_merge_cells_xml(&sheet_xml);
            if !regions.is_empty() {
                metadata.merged_regions.insert(sheet_name.clone(), regions);
            }
        }
    }

    Ok(metadata)
}

/// Read a file entry from a zip archive into a string.
fn read_zip_entry(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    path: &str,
) -> std::result::Result<String, ()> {
    let mut file = archive.by_name(path).map_err(|_| ())?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).map_err(|_| ())?;
    Ok(buf)
}

/// Collect worksheet file paths from the zip, mapping sheet index to zip path.
/// Returns (sheet_display_name, zip_path) pairs.
fn collect_sheet_filenames(archive: &mut ZipArchive<Cursor<&[u8]>>) -> Vec<(String, String)> {
    let mut sheets = Vec::new();

    // First try to read workbook.xml for sheet names
    let sheet_names_from_wb = if let Ok(wb_xml) = read_zip_entry(archive, "xl/workbook.xml") {
        parse_workbook_sheet_names(&wb_xml)
    } else {
        Vec::new()
    };

    // Match sheet names to worksheet files
    for i in 0..archive.len() {
        if let Ok(file) = archive.by_index(i) {
            let name = file.name().to_string();
            if name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml") {
                // Extract sheet number from filename (e.g., "sheet1.xml" -> 0)
                let num_str = name
                    .trim_start_matches("xl/worksheets/sheet")
                    .trim_end_matches(".xml");
                if let Ok(num) = num_str.parse::<usize>() {
                    let display_name = sheet_names_from_wb
                        .get(num.saturating_sub(1))
                        .cloned()
                        .unwrap_or_else(|| format!("Sheet{num}"));
                    sheets.push((display_name, name));
                }
            }
        }
    }

    sheets
}

/// Parse workbook.xml to extract sheet display names in order.
fn parse_workbook_sheet_names(xml: &str) -> Vec<String> {
    let mut reader = XmlReader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut names = Vec::new();
    let mut buf = Vec::new();
    let mut in_sheets = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e) | Event::Empty(ref e)) if e.name().as_ref() == b"sheets" => {
                in_sheets = true;
            }
            Ok(Event::Start(ref e) | Event::Empty(ref e))
                if in_sheets && e.name().as_ref() == b"sheet" =>
            {
                for attr in e.attributes().flatten() {
                    if attr.key.as_ref() == b"name" {
                        if let Ok(val) = attr
                            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
                        {
                            names.push(val.to_string());
                        }
                    }
                }
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"sheets" => {
                in_sheets = false;
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    names
}

/// Parse styles.xml to extract numFmt definitions and cellXfs mappings.
fn parse_styles_xml(xml: &str, metadata: &mut OoxmlMetadata) {
    let mut reader = XmlReader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut in_num_fmts = false;
    let mut in_cell_xfs = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e) | Event::Empty(ref e)) => {
                let tag = e.name();
                match tag.as_ref() {
                    b"numFmts" => in_num_fmts = true,
                    b"cellXfs" => in_cell_xfs = true,
                    b"numFmt" if in_num_fmts => {
                        let mut fmt_id = None;
                        let mut fmt_code = None;
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"numFmtId" => {
                                    if let Ok(v) = attr.decoded_and_normalized_value(
                                        XmlVersion::Implicit1_0,
                                        reader.decoder(),
                                    ) {
                                        fmt_id = v.parse::<u32>().ok();
                                    }
                                }
                                b"formatCode" => {
                                    if let Ok(v) = attr.decoded_and_normalized_value(
                                        XmlVersion::Implicit1_0,
                                        reader.decoder(),
                                    ) {
                                        fmt_code = Some(v.to_string());
                                    }
                                }
                                _ => {}
                            }
                        }
                        if let (Some(id), Some(code)) = (fmt_id, fmt_code) {
                            metadata.num_fmts.insert(id, classify_format_code(&code));
                        }
                    }
                    b"xf" if in_cell_xfs => {
                        let mut num_fmt_id = 0u32;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"numFmtId" {
                                if let Ok(v) = attr.decoded_and_normalized_value(
                                    XmlVersion::Implicit1_0,
                                    reader.decoder(),
                                ) {
                                    num_fmt_id = v.parse::<u32>().unwrap_or(0);
                                }
                            }
                        }
                        metadata.cell_xfs.push(num_fmt_id);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"numFmts" => in_num_fmts = false,
                b"cellXfs" => in_cell_xfs = false,
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
}

/// Parse a worksheet XML for `<mergeCells>` regions.
fn parse_merge_cells_xml(xml: &str) -> Vec<MergedRegion> {
    let mut reader = XmlReader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut regions = Vec::new();
    let mut in_merge_cells = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"mergeCells" => {
                in_merge_cells = true;
            }
            Ok(Event::Start(ref e) | Event::Empty(ref e))
                if in_merge_cells && e.name().as_ref() == b"mergeCell" =>
            {
                for attr in e.attributes().flatten() {
                    if attr.key.as_ref() == b"ref" {
                        if let Ok(val) = attr
                            .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
                        {
                            if let Some(((tr, lc), (br, rc))) = parse_range_ref(&val) {
                                regions.push(MergedRegion {
                                    top_row: tr,
                                    left_col: lc,
                                    bottom_row: br,
                                    right_col: rc,
                                });
                            }
                        }
                    }
                }
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"mergeCells" => {
                in_merge_cells = false;
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    regions
}

/// Convert an Excel serial date number to ISO-8601 string.
///
/// Excel dates are stored as days since 1900-01-00 (serial 1 = Jan 1, 1900).
/// Excel has the Lotus 1-2-3 bug: serial 60 = Feb 29, 1900 (which doesn't exist).
/// For serial > 60, subtract 1 to get the correct date.
#[must_use]
pub fn excel_serial_to_iso(serial: f64) -> Option<String> {
    if serial < 0.0 {
        return None;
    }

    let days_from_epoch = serial.floor() as i64;
    let frac = serial - serial.floor();

    // Base: 1899-12-31 (so serial 1 = 1900-01-01)
    // For serial > 60, Excel's Lotus bug means the real date is one day earlier
    // than what the serial suggests, because Excel thinks Feb 29, 1900 exists.
    let base = chrono::NaiveDate::from_ymd_opt(1899, 12, 31)?;
    let adjusted_days = if days_from_epoch > 60 {
        days_from_epoch - 1
    } else {
        days_from_epoch
    };
    let date = base.checked_add_signed(chrono::Duration::days(adjusted_days))?;

    if frac > 0.0001 {
        // Has a time component
        let total_seconds = (frac * 86400.0).round() as u32;
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let time = chrono::NaiveTime::from_hms_opt(hours, minutes, seconds)?;
        Some(
            chrono::NaiveDateTime::new(date, time)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string(),
        )
    } else {
        Some(date.format("%Y-%m-%d").to_string())
    }
}

/// Format a percentage value (0.153 -> "15.3%").
#[must_use]
pub fn format_percentage(val: f64) -> String {
    let pct = val * 100.0;
    if (pct - pct.round()).abs() < 0.001 {
        format!("{}%", pct.round() as i64)
    } else {
        format!("{pct:.1}%")
    }
}

/// Format a currency value with the appropriate symbol.
#[must_use]
pub fn format_currency(val: f64, code: &str) -> String {
    let lower = code.to_ascii_lowercase();
    let symbol = if lower.contains('$') || lower.contains("usd") {
        "$"
    } else if lower.contains('\u{20ac}') || lower.contains("eur") {
        "\u{20ac}"
    } else if lower.contains('\u{00a3}') || lower.contains("gbp") {
        "\u{00a3}"
    } else if lower.contains('\u{00a5}') || lower.contains("jpy") || lower.contains("cny") {
        "\u{00a5}"
    } else {
        "$" // default
    };

    if val < 0.0 {
        format!("-{symbol}{:.2}", val.abs())
    } else {
        format!("{symbol}{val:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_builtin_fmts() {
        assert_eq!(classify_builtin_fmt(0), NumFmtKind::General);
        assert_eq!(classify_builtin_fmt(1), NumFmtKind::Number);
        assert_eq!(classify_builtin_fmt(5), NumFmtKind::Currency);
        assert_eq!(classify_builtin_fmt(9), NumFmtKind::Percentage);
        assert_eq!(classify_builtin_fmt(11), NumFmtKind::Scientific);
        assert_eq!(classify_builtin_fmt(14), NumFmtKind::Date);
        assert_eq!(classify_builtin_fmt(18), NumFmtKind::Time);
        assert_eq!(classify_builtin_fmt(22), NumFmtKind::DateTime);
        assert_eq!(classify_builtin_fmt(49), NumFmtKind::Text);
    }

    #[test]
    fn test_classify_custom_formats() {
        assert_eq!(classify_format_code("yyyy-mm-dd"), NumFmtKind::Date);
        assert_eq!(classify_format_code("mm/dd/yyyy"), NumFmtKind::Date);
        assert_eq!(classify_format_code("hh:mm:ss"), NumFmtKind::Time);
        assert_eq!(
            classify_format_code("yyyy-mm-dd hh:mm"),
            NumFmtKind::DateTime
        );
        assert_eq!(classify_format_code("0.00%"), NumFmtKind::Percentage);
        assert_eq!(classify_format_code("0.00E+00"), NumFmtKind::Scientific);
        assert_eq!(classify_format_code("$#,##0.00"), NumFmtKind::Currency);
        assert_eq!(
            classify_format_code("\u{20ac}#,##0.00"),
            NumFmtKind::Currency
        );
        assert_eq!(classify_format_code("#,##0.00"), NumFmtKind::Number);
        assert_eq!(classify_format_code("@"), NumFmtKind::Text);
    }

    #[test]
    fn test_parse_cell_ref() {
        assert_eq!(parse_cell_ref("A1"), Some((0, 0)));
        assert_eq!(parse_cell_ref("B5"), Some((4, 1)));
        assert_eq!(parse_cell_ref("Z1"), Some((0, 25)));
        assert_eq!(parse_cell_ref("AA1"), Some((0, 26)));
        assert_eq!(parse_cell_ref("AZ100"), Some((99, 51)));
    }

    #[test]
    fn test_parse_range_ref() {
        let result = parse_range_ref("A1:D3");
        assert_eq!(result, Some(((0, 0), (2, 3))));
    }

    #[test]
    fn test_excel_serial_to_iso() {
        assert_eq!(excel_serial_to_iso(1.0), Some("1900-01-01".to_string()));
        assert_eq!(excel_serial_to_iso(44927.0), Some("2023-01-01".to_string()));
        assert!(excel_serial_to_iso(-1.0).is_none());
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(0.153), "15.3%");
        assert_eq!(format_percentage(0.5), "50%");
        assert_eq!(format_percentage(1.0), "100%");
    }

    #[test]
    fn test_format_currency() {
        assert_eq!(format_currency(10.5, "$#,##0.00"), "$10.50");
        assert_eq!(format_currency(-10.5, "$#,##0.00"), "-$10.50");
        assert_eq!(format_currency(10.5, "\u{20ac}#,##0.00"), "\u{20ac}10.50");
    }

    #[test]
    fn test_quoted_section_removal() {
        assert_eq!(
            remove_quoted_sections("yyyy\"year\"mm\"month\"dd\"day\""),
            "yyyymmdd"
        );
        assert_eq!(remove_quoted_sections("#,##0.00\"$\""), "#,##0.00");
    }
}
