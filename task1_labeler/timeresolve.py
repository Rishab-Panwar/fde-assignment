"""Deterministic resolution of structured day/time expressions (rules F1, F4-F6, F10, F11).

The LLM never computes calendar dates. It only reports *what was said* as a
structured DaySpec/TimeSpec; this module turns that into an absolute
Asia/Kolkata timestamp relative to call_started_at.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30))

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

PART_OF_DAY = {"morning": (10, 0), "afternoon": (15, 0), "evening": (18, 0), "lunch": (13, 0)}
DEFAULT_TIME = (11, 0)


def parse_call_time(call_started_at: str) -> datetime:
    dt = datetime.fromisoformat(call_started_at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def fmt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y += d.year
    m += 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last))


def resolve_clock(t: Optional[dict]) -> Optional[tuple]:
    """TimeSpec -> (hour, minute) or None when no time was stated (F5)."""
    if not t:
        return None
    kind = t.get("kind") or "none"
    if kind == "none":
        return None
    if kind in PART_OF_DAY:
        return PART_OF_DAY[kind]
    if kind == "clock":
        h = t.get("hour")
        if h is None:
            return None
        mi = t.get("minute") or 0
        ampm = (t.get("ampm") or "unspecified").lower()
        if ampm == "am":
            h = 0 if h == 12 else h
        elif ampm == "pm":
            h = h if h == 12 else h + 12 if h < 12 else h
        else:  # F5: 1-7 -> PM, 8-11 -> AM, 12 -> noon, 13-23 already 24h
            if 1 <= h <= 7:
                h += 12
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            return None
        return (h, mi)
    return None


def resolve_day(day: Optional[dict], call_dt: datetime, clock: Optional[tuple]) -> Optional[date]:
    """DaySpec -> calendar date, or None if vague/unresolvable/past (F4, F6, F11)."""
    if not day:
        return None
    kind = day.get("kind") or "none"
    d0 = call_dt.date()
    hm = clock or DEFAULT_TIME

    def future_on(d: date) -> bool:
        return datetime.combine(d, time(*hm), IST) > call_dt

    if kind == "today":
        return d0
    if kind == "tomorrow":
        return d0 + timedelta(days=1)
    if kind == "day_after_tomorrow":
        return d0 + timedelta(days=2)
    if kind == "in_n_days":
        n = day.get("n")
        return d0 + timedelta(days=n) if isinstance(n, int) and n >= 0 else None
    if kind in ("weekday", "this_weekday", "next_weekday"):
        wd_name = (day.get("weekday") or "").lower()
        if wd_name not in WEEKDAYS:
            return None
        wd = WEEKDAYS.index(wd_name)
        delta = (wd - d0.weekday()) % 7
        if kind == "weekday":  # nearest strictly future occurrence
            return d0 + timedelta(days=delta or 7)
        if kind == "this_weekday":  # today or later, with a future time
            cand = d0 + timedelta(days=delta)
            return cand if future_on(cand) else cand + timedelta(days=7)
        next_monday = d0 + timedelta(days=7 - d0.weekday())
        return next_monday + timedelta(days=wd)
    if kind == "next_week":
        return d0 + timedelta(days=7 - d0.weekday())
    if kind == "absolute_date":
        dom = day.get("day_of_month")
        month = day.get("month")
        year = day.get("year")
        if not isinstance(dom, int):
            return None
        try:
            if isinstance(year, int) and isinstance(month, int):
                cand = date(year, month, dom)
                return cand  # explicit full date; past check happens in caller
            if isinstance(month, int):
                for y in (d0.year, d0.year + 1):
                    cand = date(y, month, dom)
                    if cand > d0 or (cand == d0 and future_on(cand)):
                        return cand
                return None
            # day-of-month only: next future occurrence of that day number
            y, m = d0.year, d0.month
            for _ in range(13):
                if dom <= calendar.monthrange(y, m)[1]:
                    cand = date(y, m, dom)
                    if cand > d0 or (cand == d0 and future_on(cand)):
                        return cand
                m += 1
                if m == 13:
                    y, m = y + 1, 1
        except ValueError:
            return None
        return None
    if kind == "in_n_months":
        n = day.get("n")
        return _add_months(d0, n) if isinstance(n, int) and n > 0 else None
    if kind == "next_month":
        return _add_months(d0.replace(day=1), 1)
    # "none", "this_week", "vague" and anything unknown are not resolvable days
    return None


def resolve_callback(day: Optional[dict], t: Optional[dict], call_started_at: str) -> Optional[str]:
    """Resolve a callback commitment to an ISO timestamp (or None if invalid/vague/past)."""
    call_dt = parse_call_time(call_started_at)
    day = day or {}
    kind = day.get("kind") or "none"

    # F10: relative intervals keep seconds from call_started_at
    if kind in ("in_n_minutes", "in_n_hours"):
        n = day.get("n")
        if not isinstance(n, (int, float)) or n <= 0:
            return None
        delta = timedelta(minutes=n) if kind == "in_n_minutes" else timedelta(hours=n)
        return fmt(call_dt + delta)

    clock = resolve_clock(t)
    if kind == "none":
        if clock is None:
            return None
        # F5: time only -> today if still future, else tomorrow
        cand = datetime.combine(call_dt.date(), time(*clock), IST)
        if cand <= call_dt:
            cand += timedelta(days=1)
        return fmt(cand)

    d = resolve_day(day, call_dt, clock)
    if d is None:
        return None
    cand = datetime.combine(d, time(*(clock or DEFAULT_TIME)), IST)
    if cand <= call_dt:  # F6/F9: never repair a past commitment
        return None
    return fmt(cand)


def day_offset(day: Optional[dict], t: Optional[dict], call_started_at: str) -> Optional[int]:
    """Calendar-date offset of a (consultation) day from the call date; None if unresolvable."""
    call_dt = parse_call_time(call_started_at)
    day = day or {}
    if (day.get("kind") or "none") == "none":
        return None
    d = resolve_day(day, call_dt, resolve_clock(t))
    if d is None:
        return None
    return (d - call_dt.date()).days


def plus_days_11am(call_started_at: str, n: int) -> str:
    call_dt = parse_call_time(call_started_at)
    return fmt(datetime.combine(call_dt.date() + timedelta(days=n), time(*DEFAULT_TIME), IST))
