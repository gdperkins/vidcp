from vidcp.util import format_duration


def test_format_duration_mm_ss():
    assert format_duration(0) == "00:00"
    assert format_duration(5) == "00:05"
    assert format_duration(65) == "01:05"
    assert format_duration(600) == "10:00"


def test_format_duration_hours():
    assert format_duration(3661) == "1:01:01"


def test_format_duration_none():
    assert format_duration(None) == "-"
