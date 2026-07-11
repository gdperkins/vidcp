"""Shared pytest fixtures.

The autouse ``_isolated_home`` fixture points ``VIDCP_HOME`` at a per-test
temporary directory so nothing ever touches the real ``~/.vidcp`` and each test
gets a clean database/store. It also clears the cached settings accessor before
and after each test so env changes take effect.

The ``fixtures`` session fixture (re)generates the synthetic sample videos;
``speech_fixture`` returns the committed ``speech.mp4``.
"""

import importlib.util
from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    from vidcp.config import get_settings

    home = tmp_path / "vidcp_home"
    monkeypatch.setenv("VIDCP_HOME", str(home))
    get_settings.cache_clear()
    yield home
    get_settings.cache_clear()


def _load_fixture_generator():
    path = _FIXTURES_DIR / "generate.py"
    spec = importlib.util.spec_from_file_location("vidcp_fixtures_generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def fixtures() -> dict[str, Path]:
    """Ensure the synthetic fixtures exist; return ``{name: Path}``."""
    return _load_fixture_generator().ensure_fixtures()


@pytest.fixture(scope="session")
def speech_fixture() -> Path:
    """Path to the committed ``speech.mp4`` fixture."""
    path = _FIXTURES_DIR / "speech.mp4"
    if not path.exists():
        pytest.skip("speech.mp4 fixture is missing")
    return path
