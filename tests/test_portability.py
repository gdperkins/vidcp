"""Windows/Unix portability guards.

Windows defaults text IO to the legacy ANSI code page (cp1252) rather than
UTF-8, silently ignores ``start_new_session``, and boots consoles in a legacy
code page. These tests pin the cross-platform behavior so it can be developed
and kept honest from any platform.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import vidcp

SRC_DIR = Path(vidcp.__file__).parent
TESTS_DIR = Path(__file__).parent

# Win32 process-creation flags (stable Win32 API values; the subprocess module
# only defines the named constants on Windows).
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


# --------------------------------------------------------------------------- #
# convention: all text IO declares an explicit encoding
# --------------------------------------------------------------------------- #


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _open_mode(node: ast.Call) -> str:
    mode = "r"
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = str(node.args[1].value)
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = str(kw.value.value)
    return mode


def test_source_text_io_declares_encoding():
    """Every text-mode IO call in src/ and tests/ must pass encoding= explicitly.

    Without it, Windows uses the ANSI code page (cp1252), which crashes or
    mangles non-ASCII transcript text, titles, and ffprobe/yt-dlp output.
    """
    offenders: list[str] = []
    files = sorted(SRC_DIR.rglob("*.py")) + sorted(TESTS_DIR.rglob("*.py"))
    for py in files:
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kwarg_names = {kw.arg for kw in node.keywords}
            name = _call_name(node)
            where = f"{py.relative_to(SRC_DIR.parents[1])}:{node.lineno}"
            if name in {"write_text", "read_text"} and "encoding" not in kwarg_names:
                offenders.append(f"{where} {name}() without encoding=")
            elif name == "open" and isinstance(node.func, ast.Name):
                if "b" not in _open_mode(node) and "encoding" not in kwarg_names:
                    offenders.append(f"{where} text-mode open() without encoding=")
            elif name in {"run", "Popen", "check_output"}:
                text_mode = any(
                    kw.arg in {"text", "universal_newlines"}
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if text_mode and "encoding" not in kwarg_names:
                    offenders.append(f"{where} {name}(text=True) without encoding=")
    assert not offenders, (
        "text IO without explicit encoding (defaults to cp1252 on Windows):\n"
        + "\n".join(offenders)
    )


# --------------------------------------------------------------------------- #
# detached background ingest
# --------------------------------------------------------------------------- #


def _capture_spawn(monkeypatch, platform: str) -> dict:
    from vidcp import mcp_server

    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mcp_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mcp_server.sys, "platform", platform)
    mcp_server._spawn_ingest(Path("video.mp4"), force=False)
    return captured


def test_spawn_ingest_detaches_via_creationflags_on_windows(monkeypatch):
    kwargs = _capture_spawn(monkeypatch, "win32")["kwargs"]
    assert "start_new_session" not in kwargs  # silently ignored on Windows
    assert kwargs["creationflags"] & DETACHED_PROCESS
    assert kwargs["creationflags"] & CREATE_NEW_PROCESS_GROUP


def test_spawn_ingest_starts_new_session_on_posix(monkeypatch):
    kwargs = _capture_spawn(monkeypatch, "linux")["kwargs"]
    assert kwargs.get("start_new_session") is True
    assert "creationflags" not in kwargs


# --------------------------------------------------------------------------- #
# console output on legacy Windows code pages
# --------------------------------------------------------------------------- #


class _FakeStream:
    def __init__(self):
        self.reconfigured = None

    def reconfigure(self, **kwargs):
        self.reconfigured = kwargs


def test_stdio_forced_to_utf8_on_windows(monkeypatch):
    from vidcp import cli

    out, err = _FakeStream(), _FakeStream()
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    cli._configure_windows_stdio()
    assert out.reconfigured == {"encoding": "utf-8", "errors": "replace"}
    assert err.reconfigured == {"encoding": "utf-8", "errors": "replace"}


def test_stdio_untouched_on_posix(monkeypatch):
    from vidcp import cli

    out = _FakeStream()
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli.sys, "stdout", out)
    cli._configure_windows_stdio()
    assert out.reconfigured is None


# --------------------------------------------------------------------------- #
# end-to-end: export must write UTF-8 regardless of the platform default
# --------------------------------------------------------------------------- #

NON_ASCII_TEXT = "café 日本語 naïve"


def _seed_video_with_segment(text: str) -> str:
    from vidcp.db import connect

    vid = "ab" + "0" * 62
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO videos (id, path, title, ingested_at) VALUES (?, ?, ?, ?)",
            (vid, "seeded.mp4", "seeded", "2026-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO segments (video_id, start_s, end_s, text) VALUES (?, 0.0, 2.0, ?)",
            (vid, text),
        )
        conn.commit()
    finally:
        conn.close()
    return vid


def test_export_writes_utf8_even_under_ascii_locale(tmp_path):
    """Reproduces the Windows cp1252 failure class on any POSIX host.

    LC_ALL=C with UTF-8 mode and locale coercion disabled gives the child
    Python an ASCII default encoding — the same trap as cp1252 on Windows.
    The export must come out as valid UTF-8 anyway.
    """
    vid = _seed_video_with_segment(NON_ASCII_TEXT)
    out = tmp_path / "out.srt"
    env = os.environ | {
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
    }
    result = subprocess.run(
        [sys.executable, "-m", "vidcp", "export", vid[:8], "--format", "srt", "-o", str(out)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, f"export failed:\n{result.stderr}"
    assert NON_ASCII_TEXT in out.read_bytes().decode("utf-8")
