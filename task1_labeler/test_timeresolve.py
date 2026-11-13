"""Unit tests for timeresolve.py. Run: python -m unittest test_timeresolve -v"""
import unittest

from timeresolve import day_offset, plus_days_11am, resolve_callback

MON = "2026-11-09T14:00:00+05:30"  # rule-sheet boundary example call time (Monday)


def cb(day=None, t=None, call=MON):
    return resolve_callback(day, t, call)


class TestCallback(unittest.TestCase):
    def test_ex1_tomorrow_at_4(self):
        self.assertEqual(cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 4, "ampm": "unspecified"}),
                         "2026-11-10T16:00:00+05:30")

    def test_ex2_vague_fallback(self):
        self.assertEqual(plus_days_11am(MON, 3), "2026-11-12T11:00:00+05:30")

    def test_ex4_default_plus1(self):
        self.assertEqual(plus_days_11am(MON, 1), "2026-11-10T11:00:00+05:30")

    def test_hours_8_to_11_am(self):
        self.assertEqual(cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 9}), "2026-11-10T09:00:00+05:30")

    def test_noon(self):
        self.assertEqual(cb({"kind": "tomorrow"}, {"kind": "clock", "hour": 12}), "2026-11-10T12:00:00+05:30")

    def test_day_only_11am(self):
        self.assertEqual(cb({"kind": "day_after_tomorrow"}), "2026-11-11T11:00:00+05:30")

    def test_time_only_future_today(self):
        self.assertEqual(cb(None, {"kind": "evening"}), "2026-11-09T18:00:00+05:30")

    def test_time_only_past_rolls_to_tomorrow(self):
        self.assertEqual(cb(None, {"kind": "morning"}), "2026-11-10T10:00:00+05:30")

    def test_today_past_time_rejected(self):
        self.assertIsNone(cb({"kind": "today"}, {"kind": "morning"}))

    def test_bare_weekday_strictly_future(self):
        self.assertEqual(cb({"kind": "weekday", "weekday": "monday"}), "2026-11-16T11:00:00+05:30")
        self.assertEqual(cb({"kind": "weekday", "weekday": "wednesday"}), "2026-11-11T11:00:00+05:30")

    def test_this_weekday_today_future_time(self):
        self.assertEqual(cb({"kind": "this_weekday", "weekday": "monday"}, {"kind": "evening"}),
                         "2026-11-09T18:00:00+05:30")

    def test_next_weekday_following_week(self):
        self.assertEqual(cb({"kind": "next_weekday", "weekday": "wednesday"}), "2026-11-18T11:00:00+05:30")

    def test_next_week_monday(self):
        self.assertEqual(cb({"kind": "next_week"}), "2026-11-16T11:00:00+05:30")

    def test_this_week_vague(self):
        self.assertIsNone(cb({"kind": "this_week"}))

    def test_in_15_minutes_keeps_seconds(self):  # AVT-047
        self.assertEqual(cb({"kind": "in_n_minutes", "n": 15}, call="2026-10-15T14:11:01+05:30"),
                         "2026-10-15T14:26:01+05:30")

    def test_day_of_month_only(self):  # AVT-017 "call you on the 26th"
        self.assertEqual(cb({"kind": "absolute_date", "day_of_month": 26}, call="2026-10-15T18:28:03+05:30"),
                         "2026-10-26T11:00:00+05:30")

    def test_month_day(self):  # AVT-029 "4th or 5th of November" -> earliest
        self.assertEqual(cb({"kind": "absolute_date", "day_of_month": 4, "month": 11},
                            call="2026-10-17T18:05:01+05:30"), "2026-11-04T11:00:00+05:30")

    def test_explicit_past_full_date_rejected(self):
        self.assertIsNone(cb({"kind": "absolute_date", "day_of_month": 1, "month": 11, "year": 2026}))

    def test_in_one_month_clamped(self):
        self.assertEqual(cb({"kind": "in_n_months", "n": 1}, call="2026-01-31T10:00:00+05:30"),
                         "2026-02-28T11:00:00+05:30")

    def test_next_month_first(self):
        self.assertEqual(cb({"kind": "next_month"}), "2026-12-01T11:00:00+05:30")

    def test_tomorrow_evening(self):  # AVT-004
        self.assertEqual(cb({"kind": "tomorrow"}, {"kind": "evening"}, call="2026-10-14T13:19:21+05:30"),
                         "2026-10-15T18:00:00+05:30")


class TestOffset(unittest.TestCase):
    def test_offsets(self):
        self.assertEqual(day_offset({"kind": "today"}, None, MON), 0)
        self.assertEqual(day_offset({"kind": "in_n_days", "n": 7}, None, MON), 7)
        self.assertEqual(day_offset({"kind": "weekday", "weekday": "sunday"}, None, MON), 6)
        self.assertIsNone(day_offset({"kind": "none"}, None, MON))


if __name__ == "__main__":
    unittest.main()
