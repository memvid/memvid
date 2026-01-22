// Safe expect: Regex patterns are compile-time literals, verified valid.
#![allow(clippy::expect_used)]
//! PII (Personally Identifiable Information) detection and masking
//!
//! This module provides functionality to detect and mask sensitive PII in text
//! before sending it to LLMs or external services. The masking happens at query
//! time, so the original data remains fully searchable in the .mv2 file.

use once_cell::sync::Lazy;
use regex::Regex;

/// Masks PII (Personally Identifiable Information) in the given text.
///
/// Detects and replaces common PII patterns with placeholder tokens:
/// - Email addresses → `[EMAIL]`
/// - US Social Security Numbers → `[SSN]`
/// - Phone numbers (various formats) → `[PHONE]`
/// - Credit card numbers → `[CREDIT_CARD]`
/// - IPv4 addresses → `[IP_ADDRESS]`
/// - API keys/tokens (common patterns) → `[API_KEY]`
///
/// # Example
///
/// ```
/// use memvid_core::pii::mask_pii;
///
/// let text = "Contact me at john@example.com or call 555-123-4567";
/// let masked = mask_pii(text);
/// assert_eq!(masked, "Contact me at [EMAIL] or call [PHONE]");
/// ```
pub fn mask_pii(text: &str) -> String {
    let mut masked = text.to_string();

    // Email addresses
    // Matches: john@example.com, user+tag@domain.co.uk
    masked = EMAIL_REGEX.replace_all(&masked, "[EMAIL]").to_string();

    // US Social Security Numbers
    // Matches: 123-45-6789, 123 45 6789, 123456789
    masked = SSN_REGEX.replace_all(&masked, "[SSN]").to_string();

    // Credit card numbers - MUST come before phone numbers!
    // Matches: 1234-5678-9012-3456, 1234 5678 9012 3456, 1234567890123456
    // Covers Visa (16 digits), Mastercard (16), Amex (15), Discover (16)
    masked = CREDIT_CARD_REGEX
        .replace_all(&masked, "[CREDIT_CARD]")
        .to_string();

    // Phone numbers (various formats)
    // Matches: (555) 123-4567, 555-123-4567, +1-555-123-4567, 555.123.4567
    masked = PHONE_REGEX.replace_all(&masked, "[PHONE]").to_string();

    // IPv4 addresses
    // Matches: 192.168.1.1, 10.0.0.1
    masked = IPV4_REGEX.replace_all(&masked, "[IP_ADDRESS]").to_string();

    // API keys and tokens (common patterns)
    // Matches: sk_live_..., pk_test_..., ghp_..., AKIA... (AWS), etc.
    masked = API_KEY_REGEX.replace_all(&masked, "[API_KEY]").to_string();

    // Generic token patterns (long alphanumeric strings that look like secrets)
    // Matches: bearer tokens, JWT-like strings, etc.
    masked = TOKEN_REGEX.replace_all(&masked, "[TOKEN]").to_string();

    masked
}

/// Checks if the given text contains any detectable PII.
///
/// Returns `true` if any PII pattern is found, `false` otherwise.
/// Useful for checking whether masking is needed.
pub fn contains_pii(text: &str) -> bool {
    EMAIL_REGEX.is_match(text)
        || SSN_REGEX.is_match(text)
        || PHONE_REGEX.is_match(text)
        || CREDIT_CARD_REGEX.is_match(text)
        || IPV4_REGEX.is_match(text)
        || API_KEY_REGEX.is_match(text)
        || TOKEN_REGEX.is_match(text)
}

// Regex patterns for PII detection
// Using Lazy to compile regexes once at first use

static EMAIL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").expect("invalid email regex")
});

static SSN_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Matches: 123-45-6789, 123 45 6789, or 123456789
    // Uses word boundaries to avoid false positives
    Regex::new(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b").expect("invalid SSN regex")
});

static PHONE_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Matches various phone formats:
    // (555) 123-4567, 555-123-4567, +1-555-123-4567, 555.123.4567, 5551234567, 555-1234
    Regex::new(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3}[-.\s]\d{4}\b|\b\d{10}\b",
    )
    .expect("invalid phone regex")
});

static CREDIT_CARD_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Matches credit card numbers:
    // - Standard 16 digits: 4532-1234-5678-9010, 4532 1234 5678 9010
    // - Amex 15 digits: 3782-822463-10005, 3782 822463 10005
    // - No separator: 4532123456789010, 378282246310005
    Regex::new(
        r"\b(?:\d{4}[-\s]?\d{6}[-\s]?\d{5}|\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|\d{15,16})\b",
    )
    .expect("invalid credit card regex")
});

static IPV4_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Matches IPv4 addresses: 192.168.1.1
    Regex::new(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")
        .expect("invalid IPv4 regex")
});

static API_KEY_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Matches common API key patterns:
    // - Stripe: sk_live_..., pk_test_...
    // - GitHub: ghp_..., gho_...
    // - AWS: AKIA...
    // - Generic: api_key=..., apikey=..., token=...
    Regex::new(
        r#"(?i)\b(?:sk_live_[a-zA-Z0-9]{24,}|pk_test_[a-zA-Z0-9]{24,}|ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|AKIA[A-Z0-9]{16}|api[-_]?key[:=]\s*['"]?[a-zA-Z0-9_\-]{20,}['"]?)\b"#
    )
    .expect("invalid API key regex")
});

static TOKEN_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Matches bearer tokens and JWT-like patterns
    // Long base64-like strings that look like tokens (40+ chars)
    Regex::new(r"\b[A-Za-z0-9_\-]{40,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\b")
        .expect("invalid token regex")
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_email() {
        let text = "Contact john.doe+tag@example.com for details";
        let masked = mask_pii(text);
        assert_eq!(masked, "Contact [EMAIL] for details");
    }

    #[test]
    fn test_mask_multiple_emails() {
        let text = "Email alice@test.com or bob@example.org";
        let masked = mask_pii(text);
        assert_eq!(masked, "Email [EMAIL] or [EMAIL]");
    }

    #[test]
    fn test_mask_ssn() {
        let text = "SSN: 123-45-6789";
        let masked = mask_pii(text);
        assert_eq!(masked, "SSN: [SSN]");
    }

    #[test]
    fn test_mask_ssn_variations() {
        assert_eq!(mask_pii("123-45-6789"), "[SSN]");
        assert_eq!(mask_pii("123 45 6789"), "[SSN]");
        assert_eq!(mask_pii("123456789"), "[SSN]");
    }

    #[test]
    fn test_mask_phone() {
        let text = "Call me at (555) 123-4567";
        let masked = mask_pii(text);
        assert_eq!(masked, "Call me at [PHONE]");
    }

    #[test]
    fn test_mask_phone_variations() {
        assert_eq!(mask_pii("555-123-4567"), "[PHONE]");
        assert_eq!(mask_pii("+1-555-123-4567"), "[PHONE]");
        assert_eq!(mask_pii("555.123.4567"), "[PHONE]");
        assert_eq!(mask_pii("5551234567"), "[PHONE]");
    }

    #[test]
    fn test_mask_credit_card() {
        let text = "Card: 4532-1234-5678-9010";
        let masked = mask_pii(text);
        assert_eq!(masked, "Card: [CREDIT_CARD]");
    }

    #[test]
    fn test_mask_credit_card_variations() {
        assert_eq!(mask_pii("4532 1234 5678 9010"), "[CREDIT_CARD]");
        assert_eq!(mask_pii("4532123456789010"), "[CREDIT_CARD]");
        // Amex (15 digits)
        assert_eq!(mask_pii("3782-822463-10005"), "[CREDIT_CARD]");
    }

    #[test]
    fn test_mask_ip_address() {
        let text = "Server at 192.168.1.1";
        let masked = mask_pii(text);
        assert_eq!(masked, "Server at [IP_ADDRESS]");
    }

    #[test]
    fn test_mask_api_key() {
        // Use generic api_key= pattern instead of Stripe pattern to avoid GitHub secret scanning
        let text = "Use key: api_key=abcdefghij1234567890xyz";
        let masked = mask_pii(text);
        assert_eq!(masked, "Use key: [API_KEY]");
    }

    #[test]
    fn test_mask_multiple_pii_types() {
        let text = "Contact john@example.com at 555-123-4567. SSN: 123-45-6789";
        let masked = mask_pii(text);
        assert_eq!(masked, "Contact [EMAIL] at [PHONE]. SSN: [SSN]");
    }

    #[test]
    fn test_no_false_positives_on_normal_text() {
        let text = "The year 2024 has 365 days.";
        let masked = mask_pii(text);
        assert_eq!(masked, text); // Should not change
    }

    #[test]
    fn test_contains_pii() {
        assert!(contains_pii("Email: john@example.com"));
        assert!(contains_pii("SSN: 123-45-6789"));
        assert!(contains_pii("Call 555-1234"));
        assert!(!contains_pii("No PII here"));
        assert!(!contains_pii("Just numbers: 12345"));
    }

    #[test]
    fn test_preserves_non_pii_numbers() {
        let text = "Invoice #12345 for $100.00";
        let masked = mask_pii(text);
        assert_eq!(masked, text); // Should not mask invoice numbers or prices
    }

    #[test]
    fn test_preserves_dates() {
        let text = "Meeting on 2024-01-15";
        let masked = mask_pii(text);
        assert_eq!(masked, text); // Should not mask dates as SSN
    }
}
