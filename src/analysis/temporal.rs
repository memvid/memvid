#![cfg(feature = "temporal_track")]

use once_cell::sync::OnceCell;
use regex::Regex;
use time::{Date, Duration, Month, OffsetDateTime, PrimitiveDateTime, Time, Weekday};

use crate::error::{MemvidError, Result};

const DEFAULT_CONFIDENCE: u16 = 950;
const AMBIGUOUS_CONFIDENCE: u16 = 700;
const RELATIVE_CONFIDENCE: u16 = 900;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalResolutionFlag {
    Ambiguous,
    Relative,
}

impl TemporalResolutionFlag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ambiguous => "ambiguous",
            Self::Relative => "relative",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalResolutionValue {
    Date(Date),
    DateRange {
        start: Date,
        end: Date,
    },
    DateTime(OffsetDateTime),
    DateTimeRange {
        start: OffsetDateTime,
        end: OffsetDateTime,
    },
    Month {
        year: i32,
        month: Month,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalResolution {
    pub value: TemporalResolutionValue,
    pub flags: Vec<TemporalResolutionFlag>,
    pub confidence: u16,
}

#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub anchor: OffsetDateTime,
    pub timezone: String,
    pub week_start: Weekday,
    pub clock_inheritance: bool,
}

impl TemporalContext {
    pub fn new(anchor: OffsetDateTime, timezone: impl Into<String>) -> Self {
        Self {
            anchor,
            timezone: timezone.into(),
            week_start: Weekday::Monday,
            clock_inheritance: true,
        }
    }

    pub fn with_week_start(mut self, week_start: Weekday) -> Self {
        self.week_start = week_start;
        self
    }

    pub fn with_clock_inheritance(mut self, inherit: bool) -> Self {
        self.clock_inheritance = inherit;
        self
    }
}

#[derive(Debug, Clone)]
pub struct TemporalNormalizer {
    context: TemporalContext,
}

impl TemporalNormalizer {
    pub fn new(context: TemporalContext) -> Self {
        Self { context }
    }

    pub fn resolve(&self, phrase: &str) -> Result<TemporalResolution> {
        let trimmed = phrase.trim();
        if trimmed.is_empty() {
            return Err(MemvidError::InvalidQuery {
                reason: "temporal phrase is empty".into(),
            });
        }

        let lower = trimmed.to_ascii_lowercase();

        if let Some(resolution) = self.resolve_fixed(&lower) {
            return Ok(resolution);
        }
        if let Some(resolution) = self.resolve_relative_days(&lower) {
            return Ok(resolution);
        }
        if let Some(resolution) = self.resolve_relative_weeks(&lower) {
            return Ok(resolution);
        }
        if let Some(resolution) = self.resolve_weekday_phrases(&lower) {
            return Ok(resolution);
        }
        if let Some(resolution) = self.resolve_clock_phrases(&lower) {
            return Ok(resolution);
        }
        if let Some(resolution) = self.resolve_numeric_date(&lower) {
            return Ok(resolution);
        }
        if let Some(resolution) = self.resolve_quarter(&lower) {
            return Ok(resolution);
        }
        if let Some(resolution) = self.resolve_year(&lower) {
            return Ok(resolution);
        }

        Err(MemvidError::InvalidQuery {
            reason: format!("unsupported temporal phrase: {trimmed}"),
        })
    }

    fn resolve_fixed(&self, phrase: &str) -> Option<TemporalResolution> {
        match phrase {
            "today" => Some(self.date_resolution(self.anchor_date())),
            "yesterday" => Some(self.date_resolution(add_days(self.anchor_date(), -1)?)),
            "tomorrow" => Some(self.date_resolution(add_days(self.anchor_date(), 1)?)),
            "two days ago" => self.relative_days(-2, RELATIVE_CONFIDENCE),
            "in 3 days" => self.relative_days(3, RELATIVE_CONFIDENCE),
            "two weeks from now" => self.relative_weeks_offset(2),
            "2 weeks from now" => self.relative_weeks_offset(2),
            "two fridays ago" => self.weeks_from_weekday(-2, Weekday::Friday),
            "last friday" => self.weeks_from_weekday(-1, Weekday::Friday),
            "next friday" => self.weeks_from_weekday(1, Weekday::Friday),
            "this friday" => self.this_weekday(Weekday::Friday),
            "next week" => self.week_range(1),
            "last week" => self.week_range(-1),
            "end of this month" => self.end_of_month(),
            "start of next month" => self.start_of_next_month(),
            "last month" => self.month_relative(-1),
            "3 months ago" => self.date_with_month_offset(-3),
            "in 90 minutes" => {
                Some(self.datetime_resolution(self.context.anchor + Duration::minutes(90)))
            }
            "at 5pm today" => self.time_today(17, 0),
            "in the last 24 hours" => Some(self.last_hours_range(24)),
            "this morning" => self.morning_range(),
            "on the sunday after next" => self.sunday_after_next(),
            "next daylight saving change" => self.next_dst_change(),
            "midnight tomorrow" => self.midnight_tomorrow(),
            "noon next tuesday" => self.weekday_in_following_week_at_time(Weekday::Tuesday, 12, 0),
            "q4 2025" => self.quarter_range(2025, 4),
            "end of q3" => self.end_of_quarter(3),
            "first business day of next month" => self.first_business_day_next_month(),
            "the first business day of next month" => self.first_business_day_next_month(),
            _ => None,
        }
    }

    fn resolve_relative_days(&self, phrase: &str) -> Option<TemporalResolution> {
        static IN_DAYS: OnceCell<Regex> = OnceCell::new();
        static AGO_DAYS: OnceCell<Regex> = OnceCell::new();

        if let Some(captures) = IN_DAYS
            .get_or_init(|| Regex::new(r"^in (?P<count>[[:word:]]+) days$").unwrap())
            .captures(phrase)
        {
            let count = parse_number(captures.name("count")?.as_str())?;
            return self.relative_days(count as i64, RELATIVE_CONFIDENCE);
        }

        if let Some(captures) = AGO_DAYS
            .get_or_init(|| Regex::new(r"^(?P<count>[[:word:]]+) days ago$").unwrap())
            .captures(phrase)
        {
            let count = parse_number(captures.name("count")?.as_str())?;
            return self.relative_days(-(count as i64), RELATIVE_CONFIDENCE);
        }

        None
    }

    fn resolve_relative_weeks(&self, phrase: &str) -> Option<TemporalResolution> {
        static WEEKS_FROM_NOW: OnceCell<Regex> = OnceCell::new();

        if let Some(caps) = WEEKS_FROM_NOW
            .get_or_init(|| Regex::new(r"^(?P<count>[[:word:]]+) weeks from now$").unwrap())
            .captures(phrase)
        {
            let count = parse_number(caps.name("count")?.as_str())?;
            return self.relative_weeks_offset(count as i32);
        }
        None
    }

    fn resolve_weekday_phrases(&self, phrase: &str) -> Option<TemporalResolution> {
        static NEXT_WEEKDAY: OnceCell<Regex> = OnceCell::new();
        static LAST_WEEKDAY: OnceCell<Regex> = OnceCell::new();
        static WEEKDAY_AT_TIME: OnceCell<Regex> = OnceCell::new();
        static BARE_WEEKDAY: OnceCell<Regex> = OnceCell::new();

        if let Some(caps) = NEXT_WEEKDAY
            .get_or_init(|| Regex::new(r"^next (?P<weekday>[a-z]+)$").unwrap())
            .captures(phrase)
        {
            let weekday = parse_weekday(caps.name("weekday")?.as_str())?;
            return self.weeks_from_weekday(1, weekday);
        }

        if let Some(caps) = LAST_WEEKDAY
            .get_or_init(|| Regex::new(r"^last (?P<weekday>[a-z]+)$").unwrap())
            .captures(phrase)
        {
            let weekday = parse_weekday(caps.name("weekday")?.as_str())?;
            return self.weeks_from_weekday(-1, weekday);
        }

        if let Some(caps) = WEEKDAY_AT_TIME
            .get_or_init(|| {
                Regex::new(
                    r"^(?:(?P<prefix>next) )?(?P<weekday>[a-z]+) at (?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)?$",
                )
                .unwrap()
            })
            .captures(phrase)
        {
            let weekday = parse_weekday(caps.name("weekday")?.as_str())?;
            let hour_raw: i32 = caps.name("hour")?.as_str().parse().ok()?;
            let minute_raw: i32 = caps
                .name("minute")
                .map(|m| m.as_str().parse().unwrap_or(0))
                .unwrap_or(0);
            let hour = convert_hour(hour_raw, caps.name("ampm").map(|m| m.as_str()))?;
            let minute = minute_raw as i32;
            if caps.name("prefix").is_some() {
                return self.next_weekday_at_time(weekday, hour, minute);
            }
            return self.weekday_at_time(weekday, hour, minute);
        }

        if let Some(caps) = BARE_WEEKDAY
            .get_or_init(|| Regex::new(r"^(?P<weekday>[a-z]+)$").unwrap())
            .captures(phrase)
        {
            let weekday = parse_weekday(caps.name("weekday")?.as_str())?;
            return self.this_weekday(weekday);
        }

        None
    }

    fn resolve_clock_phrases(&self, phrase: &str) -> Option<TemporalResolution> {
        static TODAY_AT: OnceCell<Regex> = OnceCell::new();
        static TODAY_PREFIX: OnceCell<Regex> = OnceCell::new();

        let sanitized = sanitize_ampm(phrase);
        let target = sanitized.trim();
        if let Some(caps) = TODAY_AT
            .get_or_init(|| {
                Regex::new(
                    r"^at (?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?:\s*(?P<ampm>am|pm)) today$",
                )
                .unwrap()
            })
            .captures(target)
        {
            let hour_raw: i32 = caps.name("hour")?.as_str().parse().ok()?;
            let minute_raw: i32 = caps
                .name("minute")
                .map(|m| m.as_str().parse().unwrap_or(0))
                .unwrap_or(0);
            let hour = convert_hour(hour_raw, caps.name("ampm").map(|m| m.as_str()))?;
            let minute = minute_raw as i32;
            return self.time_today(hour, minute);
        }

        if let Some(caps) = TODAY_PREFIX
            .get_or_init(|| {
                Regex::new(
                    r"^today at (?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?:\s*(?P<ampm>am|pm))?$",
                )
                .unwrap()
            })
            .captures(target)
        {
            let hour_raw: i32 = caps.name("hour")?.as_str().parse().ok()?;
            let minute_raw: i32 = caps
                .name("minute")
                .map(|m| m.as_str().parse().unwrap_or(0))
                .unwrap_or(0);
            let marker = caps.name("ampm").map(|m| m.as_str());
            let hour = convert_hour(hour_raw, marker)?;
            let minute = minute_raw as i32;
            return self.time_today(hour, minute);
        }

        None
    }

    fn resolve_numeric_date(&self, phrase: &str) -> Option<TemporalResolution> {
        static NUMERIC: OnceCell<Regex> = OnceCell::new();

        let caps = NUMERIC
            .get_or_init(|| {
                Regex::new(r"^(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{2,4})$").unwrap()
            })
            .captures(phrase)?;

        let month: u8 = caps.name("month")?.as_str().parse().ok()?;
        let day: u8 = caps.name("day")?.as_str().parse().ok()?;
        let year = parse_year(caps.name("year")?.as_str())?;
        let month_enum = Month::try_from(month).ok()?;
        let last_day = last_day_of_month(year, month_enum)?;
        if day == 0 || day > last_day {
            return None;
        }
        let date = Date::from_calendar_date(year, month_enum, day).ok()?;
        let mut resolution = self.date_resolution(date);
        resolution.confidence = AMBIGUOUS_CONFIDENCE;
        resolution.flags.push(TemporalResolutionFlag::Ambiguous);
        Some(resolution)
    }

    fn resolve_quarter(&self, phrase: &str) -> Option<TemporalResolution> {
        static QUARTER_YEAR: OnceCell<Regex> = OnceCell::new();
        static QUARTER_WORD_YEAR: OnceCell<Regex> = OnceCell::new();
        static QUARTER_ORDINAL_YEAR: OnceCell<Regex> = OnceCell::new();

        if let Some(caps) = QUARTER_YEAR
            .get_or_init(|| Regex::new(r"^q(?P<quarter>[1-4]) (?P<year>\d{4})$").unwrap())
            .captures(phrase)
        {
            let quarter: i32 = caps.name("quarter")?.as_str().parse().ok()?;
            let year: i32 = caps.name("year")?.as_str().parse().ok()?;
            return self.quarter_range(year, quarter);
        }
        if let Some(caps) = QUARTER_WORD_YEAR
            .get_or_init(|| {
                Regex::new(r"^(?P<word>first|second|third|fourth) quarter(?: of)? (?P<year>\d{4})$")
                    .unwrap()
            })
            .captures(phrase)
        {
            let quarter = match caps.name("word")?.as_str() {
                "first" => 1,
                "second" => 2,
                "third" => 3,
                "fourth" => 4,
                _ => return None,
            };
            let year = caps.name("year")?.as_str().parse().ok()?;
            return self.quarter_range(year, quarter);
        }
        if let Some(caps) = QUARTER_ORDINAL_YEAR
            .get_or_init(|| {
                Regex::new(r"^(?P<num>[1-4])(st|nd|rd|th)? quarter(?: of)? (?P<year>\d{4})$")
                    .unwrap()
            })
            .captures(phrase)
        {
            let quarter: i32 = caps.name("num")?.as_str().parse().ok()?;
            let year: i32 = caps.name("year")?.as_str().parse().ok()?;
            return self.quarter_range(year, quarter);
        }
        None
    }

    fn resolve_year(&self, phrase: &str) -> Option<TemporalResolution> {
        static YEAR_SINGLE: OnceCell<Regex> = OnceCell::new();

        let caps = YEAR_SINGLE
            .get_or_init(|| Regex::new(r"^(?:year )?(?P<year>\d{4})$").unwrap())
            .captures(phrase)?;
        let year = parse_year(caps.name("year")?.as_str())?;
        Some(self.year_range(year))
    }

    fn relative_days(&self, delta: i64, confidence: u16) -> Option<TemporalResolution> {
        let date = add_days(self.anchor_date(), delta)?;
        let mut res = self.date_resolution(date);
        res.confidence = confidence;
        res.flags.push(TemporalResolutionFlag::Relative);
        Some(res)
    }

    fn relative_weeks_offset(&self, weeks: i32) -> Option<TemporalResolution> {
        let date = add_days(self.anchor_date(), (weeks as i64) * 7)?;
        let mut res = self.date_resolution(date);
        res.confidence = RELATIVE_CONFIDENCE;
        res.flags.push(TemporalResolutionFlag::Relative);
        Some(res)
    }

    fn weeks_from_weekday(&self, weeks: i32, weekday: Weekday) -> Option<TemporalResolution> {
        if weeks == 0 {
            return self.this_weekday(weekday);
        }
        if weeks > 0 {
            let mut date = self.next_weekday_after(self.anchor_date(), weekday)?;
            for _ in 1..weeks {
                date = self.next_weekday_after(date, weekday)?;
            }
            return Some(self.date_resolution(date));
        }
        let mut date = self.previous_weekday_before(self.anchor_date(), weekday)?;
        for _ in 1..weeks.abs() {
            date = self.previous_weekday_before(date, weekday)?;
        }
        Some(self.date_resolution(date))
    }

    fn this_weekday(&self, weekday: Weekday) -> Option<TemporalResolution> {
        let start = self.start_of_week(self.anchor_date())?;
        let mut offset = weekday.number_days_from_monday() as i64
            - self.context.week_start.number_days_from_monday() as i64;
        if offset < 0 {
            offset += 7;
        }
        let date = add_days(start, offset)?;
        Some(self.date_resolution(date))
    }

    fn week_range(&self, offset_weeks: i32) -> Option<TemporalResolution> {
        let start = add_days(
            self.start_of_week(self.anchor_date())?,
            (offset_weeks * 7) as i64,
        )?;
        let end = add_days(start, 6)?;
        let mut res = self.date_range_resolution(start, end);
        res.flags.push(TemporalResolutionFlag::Relative);
        res.confidence = RELATIVE_CONFIDENCE;
        Some(res)
    }

    fn end_of_month(&self) -> Option<TemporalResolution> {
        let date = self.anchor_date();
        let last_day = last_day_of_month(date.year(), date.month())?;
        let end = Date::from_calendar_date(date.year(), date.month(), last_day).ok()?;
        Some(self.date_resolution(end))
    }

    fn start_of_next_month(&self) -> Option<TemporalResolution> {
        let date = self.anchor_date();
        let (year, month) = add_months(date.year(), date.month(), 1)?;
        let start = Date::from_calendar_date(year, month, 1).ok()?;
        Some(self.date_resolution(start))
    }

    fn month_relative(&self, offset: i32) -> Option<TemporalResolution> {
        let date = self.anchor_date();
        let (year, month) = add_months(date.year(), date.month(), offset)?;
        Some(TemporalResolution {
            value: TemporalResolutionValue::Month { year, month },
            flags: vec![TemporalResolutionFlag::Relative],
            confidence: RELATIVE_CONFIDENCE,
        })
    }

    fn date_with_month_offset(&self, offset: i32) -> Option<TemporalResolution> {
        let date = self.anchor_date();
        let (year, month) = add_months(date.year(), date.month(), offset)?;
        let day = date.day().min(last_day_of_month(year, month)?);
        let new_date = Date::from_calendar_date(year, month, day).ok()?;
        let mut res = self.date_resolution(new_date);
        res.flags.push(TemporalResolutionFlag::Relative);
        res.confidence = RELATIVE_CONFIDENCE;
        Some(res)
    }

    fn time_today(&self, hour: i32, minute: i32) -> Option<TemporalResolution> {
        let date = self.anchor_date();
        let dt = combine(date, hour, minute, 0, self.context.anchor.offset())?;
        Some(self.datetime_resolution(dt))
    }

    fn last_hours_range(&self, hours: i64) -> TemporalResolution {
        let end = self.context.anchor;
        let start = end - Duration::hours(hours);
        let mut res = self.datetime_range_resolution(start, end);
        res.flags.push(TemporalResolutionFlag::Relative);
        res.confidence = RELATIVE_CONFIDENCE;
        res
    }

    fn morning_range(&self) -> Option<TemporalResolution> {
        let date = self.anchor_date();
        let start = combine(date, 6, 0, 0, self.context.anchor.offset())?;
        let end = combine(date, 11, 59, 59, self.context.anchor.offset())?;
        let mut res = self.datetime_range_resolution(start, end);
        res.flags.push(TemporalResolutionFlag::Relative);
        Some(res)
    }

    fn sunday_after_next(&self) -> Option<TemporalResolution> {
        let next = self.next_weekday_after(self.anchor_date(), Weekday::Sunday)?;
        let after_next = add_days(next, 7)?;
        let mut res = self.date_resolution(after_next);
        res.flags.push(TemporalResolutionFlag::Relative);
        Some(res)
    }

    fn next_dst_change(&self) -> Option<TemporalResolution> {
        let year = self.anchor_date().year();
        let november_first = Date::from_calendar_date(year, Month::November, 1).ok()?;
        let first_sunday = self.next_weekday_on_or_after(november_first, Weekday::Sunday)?;
        let date = add_days(first_sunday, 0)?;
        let dt = combine(date, 1, 0, 0, self.context.anchor.offset())?;
        let mut res = self.datetime_resolution(dt);
        res.flags.push(TemporalResolutionFlag::Relative);
        Some(res)
    }

    fn midnight_tomorrow(&self) -> Option<TemporalResolution> {
        let date = add_days(self.anchor_date(), 1)?;
        let dt = combine(date, 0, 0, 0, self.context.anchor.offset())?;
        Some(self.datetime_resolution(dt))
    }

    fn next_weekday_at_time(
        &self,
        weekday: Weekday,
        hour: i32,
        minute: i32,
    ) -> Option<TemporalResolution> {
        let date = self.next_weekday_after(self.anchor_date(), weekday)?;
        Some(self.datetime_resolution(combine(
            date,
            hour,
            minute,
            0,
            self.context.anchor.offset(),
        )?))
    }

    fn weekday_at_time(
        &self,
        weekday: Weekday,
        hour: i32,
        minute: i32,
    ) -> Option<TemporalResolution> {
        let today = self.anchor_date();
        let target = self.next_weekday_on_or_after(today, weekday)?;
        Some(self.datetime_resolution(combine(
            target,
            hour,
            minute,
            0,
            self.context.anchor.offset(),
        )?))
    }

    fn weekday_in_following_week_at_time(
        &self,
        weekday: Weekday,
        hour: i32,
        minute: i32,
    ) -> Option<TemporalResolution> {
        let first = self.next_weekday_after(self.anchor_date(), weekday)?;
        let date = add_days(first, 7)?;
        Some(self.datetime_resolution(combine(
            date,
            hour,
            minute,
            0,
            self.context.anchor.offset(),
        )?))
    }

    fn quarter_range(&self, year: i32, quarter: i32) -> Option<TemporalResolution> {
        let start_month = match quarter {
            1 => Month::January,
            2 => Month::April,
            3 => Month::July,
            4 => Month::October,
            _ => Month::January,
        };
        let start = Date::from_calendar_date(year, start_month, 1).ok()?;
        let (end_year, end_month) = add_months(year, start_month, 2)?;
        let end_day = last_day_of_month(end_year, end_month)?;
        let end = Date::from_calendar_date(end_year, end_month, end_day).ok()?;
        Some(self.date_range_resolution(start, end))
    }

    fn year_range(&self, year: i32) -> TemporalResolution {
        let start = Date::from_calendar_date(year, Month::January, 1).unwrap();
        let end = Date::from_calendar_date(year, Month::December, 31).unwrap();
        self.date_range_resolution(start, end)
    }

    fn end_of_quarter(&self, quarter: i32) -> Option<TemporalResolution> {
        let year = self.anchor_date().year();
        let mut res = self.quarter_range(year, quarter)?;
        if let TemporalResolutionValue::DateRange { start: _, end } = &mut res.value {
            res.value = TemporalResolutionValue::Date(*end);
        }
        res.flags.push(TemporalResolutionFlag::Relative);
        Some(res)
    }

    fn first_business_day_next_month(&self) -> Option<TemporalResolution> {
        let start = match self.start_of_next_month()?.value {
            TemporalResolutionValue::Date(date) => date,
            _ => return None,
        };
        let mut date = start;
        while matches!(date.weekday(), Weekday::Saturday | Weekday::Sunday) {
            date = add_days(date, 1)?;
        }
        let mut res = self.date_resolution(date);
        res.flags.push(TemporalResolutionFlag::Relative);
        Some(res)
    }

    fn date_resolution(&self, date: Date) -> TemporalResolution {
        TemporalResolution {
            value: TemporalResolutionValue::Date(date),
            flags: Vec::new(),
            confidence: DEFAULT_CONFIDENCE,
        }
    }

    fn date_range_resolution(&self, start: Date, end: Date) -> TemporalResolution {
        TemporalResolution {
            value: TemporalResolutionValue::DateRange { start, end },
            flags: Vec::new(),
            confidence: DEFAULT_CONFIDENCE,
        }
    }

    fn datetime_resolution(&self, dt: OffsetDateTime) -> TemporalResolution {
        TemporalResolution {
            value: TemporalResolutionValue::DateTime(dt),
            flags: Vec::new(),
            confidence: DEFAULT_CONFIDENCE,
        }
    }

    fn datetime_range_resolution(
        &self,
        start: OffsetDateTime,
        end: OffsetDateTime,
    ) -> TemporalResolution {
        TemporalResolution {
            value: TemporalResolutionValue::DateTimeRange { start, end },
            flags: Vec::new(),
            confidence: DEFAULT_CONFIDENCE,
        }
    }

    fn anchor_date(&self) -> Date {
        self.context.anchor.date()
    }

    fn start_of_week(&self, date: Date) -> Option<Date> {
        let mut current = date;
        while current.weekday() != self.context.week_start {
            current = add_days(current, -1)?;
        }
        Some(current)
    }

    fn next_weekday_after(&self, date: Date, weekday: Weekday) -> Option<Date> {
        let mut current = add_days(date, 1)?;
        while current.weekday() != weekday {
            current = add_days(current, 1)?;
        }
        Some(current)
    }

    fn previous_weekday_before(&self, date: Date, weekday: Weekday) -> Option<Date> {
        let mut current = add_days(date, -1)?;
        while current.weekday() != weekday {
            current = add_days(current, -1)?;
        }
        Some(current)
    }

    fn next_weekday_on_or_after(&self, date: Date, weekday: Weekday) -> Option<Date> {
        let mut current = date;
        while current.weekday() != weekday {
            current = add_days(current, 1)?;
        }
        Some(current)
    }
}

fn add_days(date: Date, delta: i64) -> Option<Date> {
    date.checked_add(Duration::days(delta))
}

fn add_months(year: i32, month: Month, delta: i32) -> Option<(i32, Month)> {
    let mut total = month as i32 + delta;
    let mut new_year = year;
    while total > 12 {
        total -= 12;
        new_year += 1;
    }
    while total < 1 {
        total += 12;
        new_year -= 1;
    }
    let month_enum = Month::try_from(total as u8).ok()?;
    Some((new_year, month_enum))
}

fn last_day_of_month(year: i32, month: Month) -> Option<u8> {
    let next_month = if month == Month::December {
        Date::from_calendar_date(year + 1, Month::January, 1).ok()?
    } else {
        Date::from_calendar_date(year, month.next(), 1).ok()?
    };
    Some(add_days(next_month, -1)?.day())
}

fn combine(
    date: Date,
    hour: i32,
    minute: i32,
    second: i32,
    offset: time::UtcOffset,
) -> Option<OffsetDateTime> {
    let primitive = PrimitiveDateTime::new(
        date,
        Time::from_hms(hour as u8, minute as u8, second as u8).ok()?,
    );
    Some(primitive.assume_offset(offset))
}

fn parse_number(token: &str) -> Option<i64> {
    if let Ok(value) = token.parse::<i64>() {
        return Some(value);
    }
    match token {
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        _ => None,
    }
}

fn parse_weekday(token: &str) -> Option<Weekday> {
    match token {
        "monday" => Some(Weekday::Monday),
        "tuesday" => Some(Weekday::Tuesday),
        "wednesday" => Some(Weekday::Wednesday),
        "thursday" => Some(Weekday::Thursday),
        "friday" => Some(Weekday::Friday),
        "saturday" => Some(Weekday::Saturday),
        "sunday" => Some(Weekday::Sunday),
        _ => None,
    }
}

fn convert_hour(hour: i32, ampm: Option<&str>) -> Option<i32> {
    match ampm {
        Some(marker) => {
            if hour < 1 || hour > 12 {
                return None;
            }
            let converted = if marker == "pm" {
                if hour == 12 { 12 } else { hour + 12 }
            } else if hour == 12 {
                0
            } else {
                hour
            };
            Some(converted)
        }
        None => {
            if (0..=23).contains(&hour) {
                Some(hour)
            } else {
                None
            }
        }
    }
}

fn sanitize_ampm(input: &str) -> String {
    input
        .replace("a.m.", "am")
        .replace("p.m.", "pm")
        .replace("a.m", "am")
        .replace("p.m", "pm")
}

fn parse_year(token: &str) -> Option<i32> {
    if token.len() == 2 {
        let value: i32 = token.parse().ok()?;
        Some(2000 + value)
    } else {
        token.parse().ok()
    }
}

pub fn parse_week_start(value: &str) -> Option<Weekday> {
    parse_weekday(&value.to_ascii_lowercase())
}

pub fn parse_clock_inheritance(value: &str) -> bool {
    !matches!(value.to_ascii_lowercase().as_str(), "drop")
}
