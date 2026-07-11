import json

from vidcp.models import OcrBlock, SceneRow, Segment, StageState, Video


def test_video_short_id():
    v = Video(id="a" * 64, path="/x/movie.mp4", ingested_at="2026-07-11T00:00:00")
    assert v.short_id == "a" * 8
    assert len(v.short_id) == 8


def test_video_from_row_parses_meta_and_has_audio():
    row = {
        "id": "b" * 64,
        "path": "/x/m.mp4",
        "title": "m",
        "duration_s": 8.0,
        "width": 320,
        "height": 240,
        "fps": 15.0,
        "vcodec": "h264",
        "acodec": None,
        "size_bytes": 1234,
        "has_audio": 0,
        "created_at": None,
        "ingested_at": "2026-07-11T00:00:00",
        "meta": json.dumps({"format": {"x": 1}}),
    }
    v = Video.from_row(row)
    assert v.has_audio is False
    assert v.meta == {"format": {"x": 1}}
    assert v.width == 320


def test_video_json_roundtrip():
    v = Video(
        id="c" * 64,
        path="/x/m.mp4",
        ingested_at="2026-07-11T00:00:00",
        has_audio=True,
    )
    back = json.loads(json.dumps(v.model_dump(mode="json")))
    assert back["id"] == "c" * 64
    assert back["has_audio"] is True


def test_stage_state_from_row():
    row = {
        "video_id": "d" * 64,
        "stage": "probe",
        "status": "done",
        "started_at": "t0",
        "finished_at": "t1",
        "error": None,
        "config_hash": "h",
    }
    s = StageState.from_row(row)
    assert s.stage == "probe"
    assert s.status == "done"


def test_other_models_construct():
    assert SceneRow(id=1, video_id="e" * 64, idx=0, start_s=0.0, end_s=2.0).idx == 0
    assert Segment(id=1, video_id="e" * 64, start_s=0.0, end_s=1.0, text="hi").text == "hi"
    assert OcrBlock(id=1, video_id="e" * 64, start_s=0.0, end_s=1.0, text="T").text == "T"
