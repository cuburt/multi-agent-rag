"""Pure-Python unit tests for slot-discovery math.

These functions don't touch the DB, so they're cheap to run on every CI build
and lock in the trickier rounding behavior that's easy to regress on.
"""
from datetime import datetime

from src.tools.scheduler import _round_up_to_slot, _hhmm_on, _WEEKDAY_KEYS


class TestRoundUpToSlot:
    def test_rounds_up_to_next_30min_boundary(self):
        assert _round_up_to_slot(datetime(2026, 5, 4, 8, 55), 30) == datetime(2026, 5, 4, 9, 0)

    def test_on_boundary_stays_put(self):
        assert _round_up_to_slot(datetime(2026, 5, 4, 8, 30), 30) == datetime(2026, 5, 4, 8, 30)

    def test_one_minute_past_boundary_rounds_to_next(self):
        assert _round_up_to_slot(datetime(2026, 5, 4, 8, 31), 30) == datetime(2026, 5, 4, 9, 0)

    def test_rolls_over_to_next_hour_when_round_lands_on_60(self):
        assert _round_up_to_slot(datetime(2026, 5, 4, 9, 59), 30) == datetime(2026, 5, 4, 10, 0)

    def test_zeroes_seconds_and_microseconds(self):
        assert _round_up_to_slot(
            datetime(2026, 5, 4, 9, 0, 15, 7), 30
        ) == datetime(2026, 5, 4, 9, 0)

    def test_15_minute_grid(self):
        # 15-min grid: 09:08 → 09:15 (not 09:30)
        assert _round_up_to_slot(datetime(2026, 5, 4, 9, 8), 15) == datetime(2026, 5, 4, 9, 15)

    def test_60_minute_grid_rolls_at_top_of_hour(self):
        assert _round_up_to_slot(datetime(2026, 5, 4, 9, 1), 60) == datetime(2026, 5, 4, 10, 0)


class TestHhmmOn:
    def test_replaces_hour_minute_zeroes_sec_usec(self):
        anchor = datetime(2026, 5, 4, 8, 55, 30, 999)
        assert _hhmm_on(anchor, "13:00") == datetime(2026, 5, 4, 13, 0)

    def test_preserves_date(self):
        anchor = datetime(2026, 12, 31, 23, 59)
        assert _hhmm_on(anchor, "09:30") == datetime(2026, 12, 31, 9, 30)


class TestWeekdayKeys:
    def test_monday_is_index_zero(self):
        # 2026-05-04 is a Monday.
        assert _WEEKDAY_KEYS[datetime(2026, 5, 4).weekday()] == "mon"

    def test_sunday_is_index_six(self):
        # 2026-05-10 is a Sunday.
        assert _WEEKDAY_KEYS[datetime(2026, 5, 10).weekday()] == "sun"

    def test_full_week_round_trip(self):
        # Mon=0 .. Sun=6 alignment with our keys list.
        assert _WEEKDAY_KEYS == ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
