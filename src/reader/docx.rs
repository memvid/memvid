use std::io::{Cursor, Read};

use quick_xml::Reader as XmlReader;
use quick_xml::events::Event;
use zip::ZipArchive;

use crate::{
    DocumentFormat, DocumentReader, PassthroughReader, ReaderDiagnostics, ReaderHint, ReaderOutput,
    Result,
};

const DOC_XML_PATH: &str = "word/document.xml";

pub struct DocxReader;

impl DocxReader {
    fn extract_text(bytes: &[u8]) -> Result<String> {
        let cursor = Cursor::new(bytes);
        let mut archive =
            ZipArchive::new(cursor).map_err(|err| crate::MemvidError::ExtractionFailed {
                reason: format!("failed to open docx archive: {err}").into(),
            })?;

        let mut file =
            archive
                .by_name(DOC_XML_PATH)
                .map_err(|err| crate::MemvidError::ExtractionFailed {
                    reason: format!("docx missing document.xml: {err}").into(),
                })?;
        let mut xml = String::new();
        file.read_to_string(&mut xml)
            .map_err(|err| crate::MemvidError::ExtractionFailed {
                reason: format!("failed to read document.xml: {err}").into(),
            })?;

        Ok(extract_plain_text(&xml, b"w:p"))
    }
}

impl DocumentReader for DocxReader {
    fn name(&self) -> &'static str {
        "docx"
    }

    fn supports(&self, hint: &ReaderHint<'_>) -> bool {
        matches!(hint.format, Some(DocumentFormat::Docx))
            || hint.mime.is_some_and(|mime| {
                mime.eq_ignore_ascii_case(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            })
    }

    fn extract(&self, bytes: &[u8], hint: &ReaderHint<'_>) -> Result<ReaderOutput> {
        match Self::extract_text(bytes) {
            Ok(text) => {
                if text.trim().is_empty() {
                    // quick-xml returned empty - try extractous as fallback
                    let mut output = PassthroughReader.extract(bytes, hint)?;
                    output.reader_name = self.name().to_string();
                    output.diagnostics.mark_fallback();
                    output.diagnostics.record_warning(
                        "docx reader produced empty text; falling back to default extractor",
                    );
                    Ok(output)
                } else {
                    // quick-xml succeeded - build output directly WITHOUT calling extractous
                    let mut document = crate::ExtractedDocument::empty();
                    document.text = Some(text);
                    document.mime_type = Some(
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            .to_string(),
                    );
                    Ok(ReaderOutput::new(document, self.name())
                        .with_diagnostics(ReaderDiagnostics::default()))
                }
            }
            Err(err) => {
                // quick-xml failed - try extractous as fallback
                let mut fallback = PassthroughReader.extract(bytes, hint)?;
                fallback.reader_name = self.name().to_string();
                fallback.diagnostics.mark_fallback();
                fallback
                    .diagnostics
                    .record_warning(format!("docx reader error: {err}"));
                Ok(fallback)
            }
        }
    }
}

fn extract_plain_text(xml: &str, block_tag: &[u8]) -> String {
    let mut reader = XmlReader::from_str(xml);
    let mut buf = Vec::new();
    let mut text = String::new();
    let mut first_block = true;
    // quick-xml splits a text node at every entity reference (`&amp;`, `&#233;`),
    // so the pieces are collected here and flushed as one node.
    let mut node = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Text(t)) => {
                if let Ok(content) = t.decode() {
                    node.push_str(&content);
                }
            }
            Ok(Event::GeneralRef(r)) => push_entity(&mut node, &r),
            event => {
                flush_node(&mut node, &mut text);
                match event {
                    Ok(Event::Start(e)) => {
                        if e.name().as_ref().ends_with(block_tag) {
                            if !first_block {
                                text.push('\n');
                            }
                            first_block = false;
                        }
                    }
                    Ok(Event::Eof) | Err(_) => break,
                    _ => (),
                }
            }
        }
        buf.clear();
    }

    text.trim().to_string()
}

fn push_entity(node: &mut String, r: &quick_xml::events::BytesRef<'_>) {
    if let Ok(Some(ch)) = r.resolve_char_ref() {
        node.push(ch);
    } else if let Ok(name) = r.decode()
        && let Some(value) = quick_xml::escape::resolve_predefined_entity(&name)
    {
        node.push_str(value);
    }
}

fn flush_node(node: &mut String, text: &mut String) {
    let content = node.trim();
    if !content.is_empty() {
        text.push_str(content);
        text.push(' ');
    }
    node.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_entity_references_inside_text() {
        let xml = r"<w:document><w:body><w:p><w:r><w:t>Tom &amp; Jerry</w:t></w:r></w:p><w:p><w:r><w:t>caf&#233; &lt;3</w:t></w:r></w:p></w:body></w:document>";

        let text = extract_plain_text(xml, b"w:p");

        assert_eq!(text, "Tom & Jerry \ncafé <3");
    }
}
