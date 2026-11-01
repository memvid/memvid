use std::io::Cursor;

use calamine::{Data, Reader as CalamineReader, Xls};

use crate::{
    DocumentFormat, DocumentReader, PassthroughReader, ReaderDiagnostics, ReaderHint, ReaderOutput,
    Result,
};

/// Reader for legacy Excel 97-2003 (.xls) files using calamine.
pub struct XlsReader;

impl XlsReader {
    fn extract_text(bytes: &[u8]) -> Result<String> {
        let cursor = Cursor::new(bytes);
        let mut workbook =
            Xls::new(cursor).map_err(|err| crate::MemvidError::ExtractionFailed {
                reason: format!("failed to read xls workbook: {err}").into(),
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

impl DocumentReader for XlsReader {
    fn name(&self) -> &'static str {
        "xls"
    }

    fn supports(&self, hint: &ReaderHint<'_>) -> bool {
        matches!(hint.format, Some(DocumentFormat::Xls))
            || hint
                .mime
                .is_some_and(|mime| mime.eq_ignore_ascii_case("application/vnd.ms-excel"))
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
                        "xls reader produced empty text; falling back to default extractor",
                    );
                    Ok(fallback)
                } else {
                    // Calamine succeeded - build output directly WITHOUT calling extractous
                    let mut document = crate::ExtractedDocument::empty();
                    document.text = Some(text);
                    document.mime_type = Some("application/vnd.ms-excel".to_string());
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
                    .record_warning(format!("xls reader error: {err}"));
                Ok(fallback)
            }
        }
    }
}
