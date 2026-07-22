from vidcp.config import Settings, get_settings

# Note: VIDCP_HOME is intentionally NOT cleared here — the autouse fixture keeps
# it pointed at an isolated empty temp dir so no stray config.toml can perturb
# the default-value assertions.
_ENV_VARS = [
    "VIDCP_WHISPER_MODEL",
    "VIDCP_SCENE_THRESHOLD",
    "VIDCP_KEYFRAME_MIN_INTERVAL_S",
    "VIDCP_PHASH_MAX_DISTANCE",
    "VIDCP_OCR_ENABLED",
    "VIDCP_EMBED_MODEL",
    "VIDCP_LINK_MODE",
    "VIDCP_CLIP_MODEL",
    "VIDCP_CLIP_ENABLED",
]


def _clear_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults(monkeypatch):
    _clear_env(monkeypatch)
    s = Settings()
    assert s.whisper_model == "small"
    assert s.scene_threshold == 27.0
    assert s.keyframe_min_interval_s == 10.0
    assert s.phash_max_distance == 6
    assert s.ocr_enabled is True
    assert s.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.link_mode == "copy"
    assert s.clip_model == "sentence-transformers/clip-ViT-B-32"
    assert s.clip_enabled is True
    # home default is expanded (no literal tilde left behind)
    assert "~" not in str(s.home)


def test_home_and_derived_paths(monkeypatch, tmp_path):
    home = tmp_path / "h"
    monkeypatch.setenv("VIDCP_HOME", str(home))
    s = Settings()
    assert s.home == home
    assert s.db_path == home / "library.db"
    assert s.store_path == home / "store"


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("VIDCP_WHISPER_MODEL", "tiny")
    assert Settings().whisper_model == "tiny"


def test_toml_overrides_default(monkeypatch, tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.toml").write_text('whisper_model = "base"\n', encoding="utf-8")
    monkeypatch.setenv("VIDCP_HOME", str(home))
    monkeypatch.delenv("VIDCP_WHISPER_MODEL", raising=False)
    assert Settings().whisper_model == "base"


def test_env_beats_toml(monkeypatch, tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.toml").write_text('whisper_model = "base"\n', encoding="utf-8")
    monkeypatch.setenv("VIDCP_HOME", str(home))
    monkeypatch.setenv("VIDCP_WHISPER_MODEL", "tiny")
    assert Settings().whisper_model == "tiny"


def test_get_settings_is_cached():
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_clip_settings_defaults_and_env(monkeypatch):
    from vidcp.config import get_settings

    # conftest sets VIDCP_CLIP_ENABLED=false for the whole suite
    assert get_settings().clip_enabled is False
    assert get_settings().clip_model == "sentence-transformers/clip-ViT-B-32"

    monkeypatch.setenv("VIDCP_CLIP_ENABLED", "true")
    monkeypatch.setenv("VIDCP_CLIP_MODEL", "sentence-transformers/clip-ViT-B-16")
    get_settings.cache_clear()
    assert get_settings().clip_enabled is True
    assert get_settings().clip_model == "sentence-transformers/clip-ViT-B-16"
