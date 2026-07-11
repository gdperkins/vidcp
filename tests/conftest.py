"""Shared pytest fixtures.

The autouse ``_isolated_home`` fixture points ``VIDCP_HOME`` at a per-test
temporary directory so nothing ever touches the real ``~/.vidcp`` and each test
gets a clean database/store. It also clears the cached settings accessor before
and after each test so env changes take effect.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    from vidcp.config import get_settings

    home = tmp_path / "vidcp_home"
    monkeypatch.setenv("VIDCP_HOME", str(home))
    get_settings.cache_clear()
    yield home
    get_settings.cache_clear()
