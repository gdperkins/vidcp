import pytest

from vidcp.util import format_duration, parse_timestamp


def test_format_duration_mm_ss():
    assert format_duration(0) == "00:00"
    assert format_duration(5) == "00:05"
    assert format_duration(65) == "01:05"
    assert format_duration(600) == "10:00"


def test_format_duration_hours():
    assert format_duration(3661) == "1:01:01"


def test_format_duration_none():
    assert format_duration(None) == "-"


def test_parse_timestamp_plain_seconds():
    assert parse_timestamp("83") == 83.0
    assert parse_timestamp("83.5") == 83.5


def test_parse_timestamp_mm_ss():
    assert parse_timestamp("1:23") == 83.0
    assert parse_timestamp("01:23.5") == 83.5


def test_parse_timestamp_h_mm_ss():
    assert parse_timestamp("1:02:03") == 3723.0


def test_parse_timestamp_invalid():
    for bad in ("", "abc", "1:2:3:4", "1::3", "-5", "1:-2"):
        with pytest.raises(ValueError):
            parse_timestamp(bad)
