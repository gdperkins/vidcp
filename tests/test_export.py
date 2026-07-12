import re

from vidcp.export.srt import to_srt
from vidcp.export.vtt import to_vtt


class Seg:
    def __init__(self, start_s, end_s, text):
        self.start_s = start_s
        self.end_s = end_s
        self.text = text


SEGS = [Seg(0.0, 2.5, "Hello world"), Seg(2.5, 5.0, "Second line")]

SRT_TS = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")
VTT_TS = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$")


def test_srt_structure_and_sequential_indices():
    blocks = [b for b in to_srt(SEGS).strip().split("\n\n") if b.strip()]
    assert len(blocks) == 2
    for i, block in enumerate(blocks, 1):
        lines = block.splitlines()
        assert lines[0] == str(i)  # sequential index
        assert SRT_TS.match(lines[1])
        assert lines[2].strip()  # text present


def test_srt_timestamp_values():
    out = to_srt([Seg(3661.5, 3662.0, "x")])
    assert "01:01:01,500 --> 01:01:02,000" in out


def test_vtt_header_and_timestamps():
    out = to_vtt(SEGS)
    assert out.startswith("WEBVTT")
    ts_lines = [ln for ln in out.splitlines() if VTT_TS.match(ln)]
    assert len(ts_lines) == 2
