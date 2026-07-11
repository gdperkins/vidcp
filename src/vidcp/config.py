"""Application settings.

Resolution order (lowest to highest precedence): field defaults → the TOML file
at ``<VIDCP_HOME>/config.toml`` → ``VIDCP_*`` environment variables.

Note on the TOML path: the plan sketched ``toml_file="~/.vidcp/config.toml"``,
but the config file logically belongs inside the (overridable) home directory.
We therefore compute the TOML path from ``VIDCP_HOME`` in
``settings_customise_sources`` so it stays consistent with ``home`` and is
testable via a temporary ``VIDCP_HOME``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _home_from_env() -> Path:
    return Path(os.environ.get("VIDCP_HOME", "~/.vidcp")).expanduser()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIDCP_", extra="ignore")

    home: Path = Path("~/.vidcp").expanduser()
    whisper_model: str = "small"  # tiny|base|small|medium
    scene_threshold: float = 27.0  # PySceneDetect ContentDetector
    keyframe_min_interval_s: float = 10.0  # floor for sparse-cut videos
    phash_max_distance: int = 6
    ocr_enabled: bool = True
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    link_mode: str = "copy"  # copy|hardlink

    @field_validator("home", mode="after")
    @classmethod
    def _expand_home(cls, value: Path) -> Path:
        return value.expanduser()

    @property
    def db_path(self) -> Path:
        return self.home / "library.db"

    @property
    def store_path(self) -> Path:
        return self.home / "store"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_source = TomlConfigSettingsSource(
            settings_cls, toml_file=_home_from_env() / "config.toml"
        )
        # First source wins. Env beats TOML beats field defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            toml_source,
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Tests that manipulate ``VIDCP_*`` env vars should call
    ``get_settings.cache_clear()`` (the autouse test fixture does this).
    """
    return Settings()
