// Safe unwrap/expect: regex patterns from validated input strings.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use crate::error::MemvidError;
use regex::Regex;
use std::convert::TryFrom;
use time::{Date, Month, OffsetDateTime};

pub(crate) fn parse_query(query: &str) -> Result<ParsedQuery, MemvidError> {
    let mut lexer = Lexer::new(query);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression()?;
    Ok(ParsedQuery { expr })
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedQuery {
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub(crate) enum Expr {
    Or(Vec<Expr>),
    And(Vec<Expr>),
    Not(Box<Expr>),
    Term(Term),
}

#[derive(Debug, Clone)]
pub(crate) enum Term {
    Text(TextTerm),
    Field(FieldTerm),
}

#[derive(Debug, Clone)]
pub(crate) enum TextTerm {
    Word(String),
    Phrase(String),
    Wildcard(WildcardPattern),
}

#[derive(Debug, Clone)]
pub(crate) struct WildcardPattern {
    pub raw: String,
    pub regex: Regex,
}

#[derive(Debug, Clone)]
pub(crate) enum FieldTerm {
    Uri(String),
    Scope(String),
    Track(String),
    Tag(String),
    Label(String),
    DateRange(DateRange),
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DateRange {
    pub start: Option<i64>,
    pub end: Option<i64>,
}

#[derive(Debug, Clone)]
enum Token {
    Word(String),
    Phrase(String),
    Field(String, String),
    DateRange(String, String, String),
    LParen,
    RParen,
    And,
    Or,
    Not,
}

struct Lexer<'a> {
    #[allow(dead_code)]
    input: &'a str,
    chars: Vec<char>,
    index: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.chars().collect(),
            index: 0,
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, MemvidError> {
        let mut tokens = Vec::new();
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.index += 1;
                continue;
            }
            match ch {
                '(' => {
                    self.index += 1;
                    tokens.push(Token::LParen);
                }
                ')' => {
                    self.index += 1;
                    tokens.push(Token::RParen);
                }
                '"' => {
                    let phrase = self.read_quoted()?;
                    tokens.push(Token::Phrase(phrase));
                }
                _ => {
                    if let Some(token) = self.read_field_or_word()? {
                        tokens.push(token);
                    }
                }
            }
        }
        Ok(tokens)
    }

    /// Known field names that should be treated as field queries when followed by `:`
    const KNOWN_FIELDS: &'static [&'static str] =
        &["uri", "scope", "track", "tag", "label", "date"];

    fn read_field_or_word(&mut self) -> Result<Option<Token>, MemvidError> {
        let start = self.index;
        let mut colon_pos: Option<usize> = None;

        // First pass: scan the word and note if there's a colon
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() || ch == '(' || ch == ')' {
                break;
            }
            if ch == ':' && colon_pos.is_none() {
                colon_pos = Some(self.index);
            }
            self.index += 1;
        }

        // If we found a colon, check if the prefix is a known field name
        if let Some(colon_idx) = colon_pos {
            let potential_field: String = self.chars[start..colon_idx].iter().collect();
            let potential_field_lower = potential_field.to_ascii_lowercase();

            if Self::KNOWN_FIELDS.contains(&potential_field_lower.as_str()) {
                // Reset index to just after the colon and parse as field
                self.index = colon_idx + 1;
                return self.read_field(start);
            }
        }

        // Not a field query - treat the whole thing (including any `:`) as a word
        let word: String = self.chars[start..self.index].iter().collect();
        if word.is_empty() {
            return Ok(None);
        }
        match word.as_str() {
            "AND" | "and" => Ok(Some(Token::And)),
            "OR" | "or" => Ok(Some(Token::Or)),
            "NOT" | "not" => Ok(Some(Token::Not)),
            _ => Ok(Some(Token::Word(word))),
        }
    }

    fn read_field(&mut self, field_start: usize) -> Result<Option<Token>, MemvidError> {
        let field: String = self.chars[field_start..self.index - 1].iter().collect();
        let field = field.to_ascii_lowercase();
        let value_token = match self.peek() {
            Some('"') => {
                self.index += 1; // skip opening quote
                let value = self.read_until_quote()?;
                Token::Field(field, value)
            }
            Some('[') if field == "date" => {
                self.index += 1; // skip '['
                let (start, end) = self.read_date_range()?;
                Token::DateRange(field, start, end)
            }
            _ => {
                let value_start = self.index;
                while let Some(ch) = self.peek() {
                    if ch.is_whitespace() || ch == '(' || ch == ')' {
                        break;
                    }
                    self.index += 1;
                }
                let value: String = self.chars[value_start..self.index].iter().collect();
                Token::Field(field, value)
            }
        };
        Ok(Some(value_token))
    }

    fn read_quoted(&mut self) -> Result<String, MemvidError> {
        self.index += 1; // skip opening quote
        let value = self.read_until_quote()?;
        Ok(value)
    }

    fn read_until_quote(&mut self) -> Result<String, MemvidError> {
        let start = self.index;
        while let Some(ch) = self.peek() {
            if ch == '"' {
                let value: String = self.chars[start..self.index].iter().collect();
                self.index += 1; // consume closing quote
                return Ok(value);
            }
            self.index += 1;
        }
        Err(MemvidError::InvalidQuery {
            reason: "unterminated quoted string".into(),
        })
    }

    fn read_date_range(&mut self) -> Result<(String, String), MemvidError> {
        let start_pos = self.index;
        while let Some(ch) = self.peek() {
            if ch == ']' {
                let contents: String = self.chars[start_pos..self.index].iter().collect();
                let contents = contents.trim();
                self.index += 1; // consume ']'
                let parts: Vec<_> = contents.split_whitespace().collect();
                if parts.len() != 3 || !parts[1].eq_ignore_ascii_case("TO") {
                    return Err(MemvidError::InvalidQuery {
                        reason: "date range must be in format [start TO end]".into(),
                    });
                }
                return Ok((parts[0].to_string(), parts[2].to_string()));
            }
            self.index += 1;
        }
        Err(MemvidError::InvalidQuery {
            reason: "unterminated date range".into(),
        })
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.index).copied()
    }
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    fn parse_expression(&mut self) -> Result<Expr, MemvidError> {
        let mut expr = self.parse_term()?;
        while self.match_token(TokenKind::Or) {
            let rhs = self.parse_term()?;
            expr = match expr {
                Expr::Or(mut list) => {
                    list.push(rhs);
                    Expr::Or(list)
                }
                _ => Expr::Or(vec![expr, rhs]),
            };
        }
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expr, MemvidError> {
        let mut expr = self.parse_factor()?;
        loop {
            if self.match_token(TokenKind::And) {
                let rhs = self.parse_factor()?;
                expr = match expr {
                    Expr::And(mut list) => {
                        list.push(rhs);
                        Expr::And(list)
                    }
                    _ => Expr::And(vec![expr, rhs]),
                };
                continue;
            }
            if self.check(TokenKind::Or) || self.check(TokenKind::RParen) || self.is_end() {
                break;
            }
            // Implicit word separation (no explicit AND/OR) defaults to AND for precision
            let rhs = self.parse_factor()?;
            expr = match expr {
                Expr::And(mut list) => {
                    list.push(rhs);
                    Expr::And(list)
                }
                _ => Expr::And(vec![expr, rhs]),
            };
        }
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expr, MemvidError> {
        if self.match_token(TokenKind::Not) {
            let inner = self.parse_factor()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, MemvidError> {
        if self.match_token(TokenKind::LParen) {
            let expr = self.parse_expression()?;
            self.consume(TokenKind::RParen, "expected ')' after expression")?;
            return Ok(expr);
        }
        match self.advance() {
            Some(Token::Word(word)) => Ok(Expr::Term(Term::Text(TextTerm::from_word(word)))),
            Some(Token::Phrase(phrase)) => Ok(Expr::Term(Term::Text(TextTerm::Phrase(
                phrase.to_ascii_lowercase(),
            )))),
            Some(Token::Field(field, value)) => {
                let term = FieldTerm::from_pair(&field, &value)?;
                Ok(Expr::Term(Term::Field(term)))
            }
            Some(Token::DateRange(field, start, end)) => {
                let term = FieldTerm::from_date_range(&field, &start, &end)?;
                Ok(Expr::Term(Term::Field(term)))
            }
            Some(token) => Err(MemvidError::InvalidQuery {
                reason: format!("unexpected token {token:?}"),
            }),
            None => Err(MemvidError::InvalidQuery {
                reason: "unexpected end of query".into(),
            }),
        }
    }

    fn advance(&mut self) -> Option<Token> {
        if self.is_end() {
            None
        } else {
            let token = self.tokens[self.position].clone();
            self.position += 1;
            Some(token)
        }
    }

    fn match_token(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn consume(&mut self, kind: TokenKind, message: &str) -> Result<(), MemvidError> {
        if self.check(kind) {
            self.position += 1;
            Ok(())
        } else {
            Err(MemvidError::InvalidQuery {
                reason: message.into(),
            })
        }
    }

    fn check(&self, kind: TokenKind) -> bool {
        if self.is_end() {
            return false;
        }
        kind.matches(&self.tokens[self.position])
    }

    fn is_end(&self) -> bool {
        self.position >= self.tokens.len()
    }
}

enum TokenKind {
    LParen,
    RParen,
    And,
    Or,
    Not,
}

impl TokenKind {
    fn matches(&self, token: &Token) -> bool {
        matches!(
            (self, token),
            (TokenKind::LParen, Token::LParen)
                | (TokenKind::RParen, Token::RParen)
                | (TokenKind::And, Token::And)
                | (TokenKind::Or, Token::Or)
                | (TokenKind::Not, Token::Not)
        )
    }
}

impl TextTerm {
    fn from_word(word: String) -> Self {
        // Strip trailing question marks - they're punctuation, not wildcards
        // Users type "What is machine?" as a question, not a wildcard pattern
        let lower = word.to_ascii_lowercase();
        let trimmed = lower.trim_end_matches('?');

        // Strip leading/trailing punctuation that won't tokenize well
        let cleaned = trimmed.trim_matches(|c: char| !c.is_alphanumeric() && c != '*' && c != '?');

        // Only treat * or ? as wildcards when they're NOT at the end
        // (i.e., "mach?ne" or "mach*" are wildcards, but "machine?" is just "machine")
        if cleaned.contains('*') || cleaned.contains('?') {
            TextTerm::Wildcard(WildcardPattern::new(cleaned.to_string()))
        } else if cleaned.is_empty() || !cleaned.chars().any(|c| c.is_alphanumeric()) {
            // If the word has no alphanumeric chars, treat as empty
            // (e.g., "-", "---", ":", etc. won't produce tokens anyway)
            TextTerm::Word(String::new())
        } else {
            TextTerm::Word(cleaned.to_string())
        }
    }
}

impl FieldTerm {
    fn from_pair(field: &str, value: &str) -> Result<Self, MemvidError> {
        let normalized = value.trim_matches('"').to_ascii_lowercase();
        match field {
            "uri" => Ok(FieldTerm::Uri(normalized)),
            "scope" => Ok(FieldTerm::Scope(normalized)),
            "track" => Ok(FieldTerm::Track(normalized)),
            "tag" => Ok(FieldTerm::Tag(normalized)),
            "label" => Ok(FieldTerm::Label(normalized)),
            _ => Err(MemvidError::InvalidQuery {
                reason: format!("unsupported field: {field}"),
            }),
        }
    }

    fn from_date_range(field: &str, start: &str, end: &str) -> Result<Self, MemvidError> {
        if field != "date" {
            return Err(MemvidError::InvalidQuery {
                reason: format!("unexpected field for date range: {field}"),
            });
        }
        let range = DateRange {
            start: parse_date_value(start),
            end: parse_date_value(end),
        };
        Ok(FieldTerm::DateRange(range))
    }
}

pub(crate) fn parse_date_value(value: &str) -> Option<i64> {
    let trimmed = value.trim_matches('"');
    if trimmed.is_empty() || trimmed == "*" {
        return None;
    }

    if let Ok(dt) = OffsetDateTime::parse(trimmed, &time::format_description::well_known::Rfc3339) {
        return Some(dt.unix_timestamp());
    }

    parse_ymd(trimmed)
}

fn parse_ymd(input: &str) -> Option<i64> {
    let parts: Vec<_> = input.split('-').collect();
    match parts.len() {
        3 => {
            let year: i32 = parts[0].parse().ok()?;
            let month: u8 = parts[1].parse().ok()?;
            let day: u8 = parts[2].parse().ok()?;
            let month = Month::try_from(month).ok()?;
            let date = Date::from_calendar_date(year, month, day).ok()?;
            Some(date.with_hms(0, 0, 0).ok()?.assume_utc().unix_timestamp())
        }
        2 => {
            let year: i32 = parts[0].parse().ok()?;
            let month: u8 = parts[1].parse().ok()?;
            let month = Month::try_from(month).ok()?;
            let date = Date::from_calendar_date(year, month, 1).ok()?;
            Some(date.with_hms(0, 0, 0).ok()?.assume_utc().unix_timestamp())
        }
        1 => {
            if input.len() == 4 && input.chars().all(|c| c.is_ascii_digit()) {
                let year: i32 = input.parse().ok()?;
                let date = Date::from_calendar_date(year, Month::January, 1).ok()?;
                Some(date.with_hms(0, 0, 0).ok()?.assume_utc().unix_timestamp())
            } else {
                None
            }
        }
        _ => None,
    }
}

impl WildcardPattern {
    fn new(raw: String) -> Self {
        let mut pattern = String::from("^");
        for ch in raw.chars() {
            match ch {
                '*' => pattern.push_str(".*"),
                '?' => pattern.push('.'),
                _ => pattern.push_str(&regex::escape(&ch.to_string())),
            }
        }
        pattern.push('$');
        let regex = Regex::new(&pattern).unwrap_or_else(|_| Regex::new("^$").unwrap());
        Self { raw, regex }
    }

    pub fn seed(&self) -> Option<String> {
        self.raw
            .split('*')
            .next()
            .map(|segment| segment.split('?').next().unwrap_or(""))
            .filter(|seed| !seed.is_empty())
            .map(|seed| seed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_query() {
        parse_query("alpha AND beta").expect("parse");
    }

    #[test]
    fn parses_field_filters() {
        parse_query("tag:important AND uri:mv2://docs/foo").expect("parse");
    }

    #[test]
    fn parses_date_range() {
        parse_query("date:[2024-01-01 TO 2024-12-31] AND rust").expect("parse");
    }

    #[test]
    fn unknown_field_colon_treated_as_word() {
        // "IRR:" should NOT be treated as a field query - it's just text with a colon
        let result = parse_query("LP IRR: percentage");
        assert!(
            result.is_ok(),
            "query with unknown field colon should parse"
        );
    }

    #[test]
    fn colon_in_middle_of_text() {
        // Colons in the middle of text should be preserved
        let result = parse_query("ratio:1:2:3");
        assert!(result.is_ok(), "colons in text should be allowed");
    }

    #[test]
    fn known_fields_still_work() {
        // Known field queries should still work
        assert!(parse_query("tag:important").is_ok());
        assert!(parse_query("uri:mv2://docs").is_ok());
        assert!(parse_query("scope:project").is_ok());
        assert!(parse_query("track:main").is_ok());
        assert!(parse_query("label:todo").is_ok());
    }

    #[test]
    fn mixed_known_and_unknown_fields() {
        // Mix of real field queries and text with colons
        let result = parse_query("tag:work IRR:percentage ratio:2");
        assert!(result.is_ok(), "mixed query should parse");
    }

    #[test]
    fn punctuation_only_tokens_handled() {
        // Standalone punctuation should parse (will be filtered to empty)
        assert!(parse_query("-").is_ok());
        assert!(parse_query("-- ---").is_ok());
        assert!(parse_query("LP IRR - year 1").is_ok());
    }

    #[test]
    fn text_term_filters_punctuation() {
        // Punctuation-only words should produce empty Word
        match TextTerm::from_word("-".to_string()) {
            TextTerm::Word(w) => assert!(w.is_empty(), "'-' should produce empty word"),
            _ => panic!("expected Word variant"),
        }

        match TextTerm::from_word("---".to_string()) {
            TextTerm::Word(w) => assert!(w.is_empty(), "'---' should produce empty word"),
            _ => panic!("expected Word variant"),
        }

        // But words with alphanumeric content should be preserved
        match TextTerm::from_word("test-word".to_string()) {
            TextTerm::Word(w) => assert_eq!(w, "test-word"),
            _ => panic!("expected Word variant"),
        }
    }

    // Tests for implicit AND operator behavior
    // These tests verify the fix that changes implicit multi-word queries
    // from OR to AND for better precision
    #[test]
    fn implicit_and_behavior() {
        let result = parse_query("machine learning").expect("parse");
        match result.expr {
            Expr::And(children) => {
                assert_eq!(children.len(), 2, "Should have 2 AND terms");
            }
            _ => panic!("Expected Expr::And, got {:?}", result.expr),
        }
    }

    #[test]
    fn implicit_and_three_words() {
        let result = parse_query("machine learning python").expect("parse");
        match result.expr {
            Expr::And(children) => {
                assert_eq!(children.len(), 3, "Should have 3 AND terms");
            }
            _ => panic!("Expected Expr::And with 3 children"),
        }
    }

    #[test]
    fn explicit_or_still_works() {
        let result = parse_query("machine OR learning").expect("parse");
        match result.expr {
            Expr::Or(children) => {
                assert_eq!(children.len(), 2, "Should have 2 OR terms");
            }
            _ => panic!("Expected Expr::Or"),
        }
    }

    #[test]
    fn explicit_and_still_works() {
        let result = parse_query("machine AND learning").expect("parse");
        match result.expr {
            Expr::And(children) => {
                assert_eq!(children.len(), 2, "Should have 2 AND terms");
            }
            _ => panic!("Expected Expr::And"),
        }
    }

    #[test]
    fn mixed_explicit_and_implicit() {
        let result = parse_query("machine learning OR python").expect("parse");
        match result.expr {
            Expr::Or(children) => {
                assert_eq!(children.len(), 2, "Should have 2 OR branches");
                match &children[0] {
                    Expr::And(and_children) => {
                        assert_eq!(
                            and_children.len(),
                            2,
                            "First branch should have 2 AND terms"
                        );
                    }
                    _ => panic!("First OR branch should be AND"),
                }
            }
            _ => panic!("Expected Expr::Or at top level"),
        }
    }

    #[test]
    fn phrase_and_word_implicit_and() {
        let result = parse_query("\"machine learning\" python").expect("parse");
        match result.expr {
            Expr::And(children) => {
                assert_eq!(children.len(), 2, "Should have 2 AND terms");
            }
            _ => panic!("Expected Expr::And"),
        }
    }

    #[test]
    fn field_and_word_implicit_and() {
        let result = parse_query("tag:important urgent").expect("parse");
        match result.expr {
            Expr::And(children) => {
                assert_eq!(children.len(), 2, "Should have 2 AND terms");
            }
            _ => panic!("Expected Expr::And"),
        }
    }

    #[test]
    fn parentheses_preserve_implicit_and() {
        // (machine learning) python actually flattens to And([machine, learning, python])
        // This is correct optimizer behavior
        let result = parse_query("(machine learning) python").expect("parse");
        match result.expr {
            Expr::And(children) => {
                // The parser flattens nested ANDs for efficiency
                assert_eq!(children.len(), 3, "Should have 3 AND terms (flattened)");
            }
            _ => panic!("Expected Expr::And"),
        }
    }

    #[test]
    fn parentheses_with_different_operators() {
        // Test that parentheses work when needed: (machine OR learning) AND python
        let result = parse_query("(machine OR learning) python").expect("parse");
        match result.expr {
            Expr::And(children) => {
                assert_eq!(children.len(), 2, "Should have 2 AND terms");
                // First child is OR expression
                match &children[0] {
                    Expr::Or(or_children) => {
                        assert_eq!(or_children.len(), 2, "OR should have 2 terms");
                    }
                    _ => panic!("First child should be OR"),
                }
            }
            _ => panic!("Expected Expr::And at top level"),
        }
    }
}
