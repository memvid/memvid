// Safe expect/unwrap: Regex patterns are compile-time literals; JSON ops on known schemas.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::{BTreeMap, BTreeSet};

use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, Default)]
pub struct AutoTagResult {
    pub tags: Vec<String>,
    pub labels: Vec<String>,
    pub content_dates: Vec<String>,
}

#[derive(Debug, Default)]
pub struct AutoTagger;

impl AutoTagger {
    const MAX_TAGS: usize = 12;
    const MAX_LABELS: usize = 6;

    pub fn analyse(&self, text: &str, include_dates: bool) -> AutoTagResult {
        if text.trim().is_empty() {
            return AutoTagResult::default();
        }

        let tags = extract_keywords(text, Self::MAX_TAGS);
        let labels = derive_labels(text, Self::MAX_LABELS);
        let content_dates = if include_dates {
            extract_dates(text)
        } else {
            Vec::new()
        };

        AutoTagResult {
            tags,
            labels,
            content_dates,
        }
    }
}

fn extract_keywords(text: &str, limit: usize) -> Vec<String> {
    static TOKEN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)[a-z0-9][a-z0-9'-]+").unwrap());
    static STOPWORDS: Lazy<BTreeSet<&'static str>> = Lazy::new(|| {
        [
            "the",
            "and",
            "for",
            "with",
            "that",
            "from",
            "this",
            "were",
            "have",
            "has",
            "will",
            "shall",
            "into",
            "about",
            "without",
            "within",
            "between",
            "because",
            "over",
            "under",
            "after",
            "before",
            "until",
            "while",
            "their",
            "there",
            "these",
            "those",
            "your",
            "into",
            "such",
            "been",
            "where",
            "when",
            "which",
            "using",
            "also",
            "than",
            "could",
            "would",
            "should",
            "might",
            "cannot",
            "however",
            "therefore",
            "thereof",
            "hereby",
            "herein",
            "hereof",
            "based",
            "system",
            "application",
            "service",
            "provide",
            "provided",
            "including",
            "include",
            "includes",
            "version",
            "update",
            "updates",
            "usage",
        ]
        .into_iter()
        .collect()
    });

    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for token in TOKEN_RE.find_iter(text) {
        let candidate = token.as_str().to_lowercase();
        if candidate.len() < 3 {
            continue;
        }
        if STOPWORDS.contains(candidate.as_str()) {
            continue;
        }
        *counts.entry(candidate).or_insert(0) += 1;
    }

    let mut scored: Vec<(String, u32)> = counts.into_iter().collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored
        .into_iter()
        .take(limit)
        .map(|(token, _)| token)
        .collect()
}

fn derive_labels(text: &str, limit: usize) -> Vec<String> {
    static PHRASE_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?m)^(?P<phrase>[A-Z][A-Za-z0-9 &/-]{3,})$").unwrap());

    let mut labels = BTreeSet::new();
    for caps in PHRASE_RE.captures_iter(text) {
        if let Some(phrase) = caps.name("phrase") {
            let candidate = phrase.as_str().trim();
            if candidate.split_whitespace().count() <= 6 {
                labels.insert(candidate.to_string());
            }
        }
    }

    if labels.is_empty() {
        // fallback: promote top keywords by capitalising them
        let keywords = extract_keywords(text, limit);
        for kw in keywords {
            let mut chars = kw.chars();
            if let Some(first) = chars.next() {
                let mut label = first.to_uppercase().collect::<String>();
                label.push_str(chars.as_str());
                labels.insert(label);
            }
        }
    }

    labels.into_iter().take(limit).collect()
}

fn extract_dates(text: &str) -> Vec<String> {
    // Match various date formats:
    // 1. Years: 2024, 2025, etc.
    // 2. ISO dates: 2024-09-01
    // 3. US format: 09/01/2024
    // 4. Spelled out: September 1, 2024 or Sept 1, 2024 or 1 September 2024
    static DATE_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?i)\b((?:19|20)\d{2}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b").unwrap()
    });

    // Match spelled-out dates like "September 1, 2024", "Sept 10, 2024", "September 1st, 2024"
    static SPELLED_DATE_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2})\b"
        ).unwrap()
    });

    // Match European format: "1 September 2024", "1st September 2024"
    static EURO_DATE_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(?:19|20)\d{2})\b"
        ).unwrap()
    });

    let mut dates = BTreeSet::new();

    // Extract numeric dates and years
    for capture in DATE_RE.captures_iter(text) {
        if let Some(m) = capture.get(0) {
            dates.insert(m.as_str().to_string());
        }
    }

    // Extract spelled-out dates (US format)
    for capture in SPELLED_DATE_RE.captures_iter(text) {
        if let Some(m) = capture.get(1) {
            dates.insert(m.as_str().to_string());
        }
    }

    // Extract European format dates
    for capture in EURO_DATE_RE.captures_iter(text) {
        if let Some(m) = capture.get(1) {
            dates.insert(m.as_str().to_string());
        }
    }

    dates.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::{AutoTagger, extract_dates};

    #[test]
    fn produces_keywords_and_labels() {
        let text = "Rust memory engines power efficient systems. Memory safety ensures reliability in 2025.";
        let result = AutoTagger::default().analyse(text, true);
        assert!(result.tags.iter().any(|tag| tag.contains("memory")));
        assert!(!result.content_dates.is_empty());
    }

    #[test]
    fn detects_dates() {
        let dates = extract_dates("Meeting on 2025-10-08 and follow-up 10/15/2025");
        assert_eq!(dates.len(), 2);
    }

    #[test]
    fn detects_spelled_out_dates() {
        let dates = extract_dates(
            "The update on September 1, 2024 changed the phone number. Previous records from January 15, 2023.",
        );
        assert!(dates.iter().any(|d| d.contains("September")));
        assert!(dates.iter().any(|d| d.contains("January")));
    }

    #[test]
    fn detects_european_dates() {
        let dates = extract_dates("Meeting on 1 September 2024 was productive.");
        assert!(dates.iter().any(|d| d.contains("September")));
    }
}
