"""Generate synthetic video fixtures with ffmpeg + Pillow.

These files are gitignored and (re)generated on demand by the ``fixtures``
pytest session fixture. ``speech.mp4`` is NOT produced here — it is a committed
fixture created once with macOS ``say`` (real speech that whisper can
transcribe).

This build of ffmpeg has no ``drawtext`` filter (no libfreetype), so text is
rendered to a PNG with Pillow and then encoded to video.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = Path(__file__).resolve().parent

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    # Pillow >= 10.1 bundles a scalable font (Aileron), so text fixtures work
    # on hosts with no system fonts at all (e.g. minimal CI runners).
    return ImageFont.load_default(size=size)


def _make_color(path: Path) -> None:
    # testsrc2 animates continuously (no hard cuts) and carries no audio track.
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=15:duration=8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )


def _make_cuts(path: Path) -> None:
    # Four maximally-distinct solid-colour 2s segments -> hard cuts at 2s, 4s,
    # 6s that PySceneDetect reliably detects at the default threshold.
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:r=15:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x240:r=15:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:r=15:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=320x240:r=15:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
            "-filter_complex",
            "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "4:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )


def _make_text(path: Path) -> None:
    width, height, text = 720, 480, "HELLO VIDCP 42"
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    font = _load_font(72)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = (width - (right - left)) / 2 - left
    y = (height - (bottom - top)) / 2 - top
    draw.text((x, y), text, fill="white", font=font)

    png = path.with_suffix(".png")
    image.save(png)
    try:
        _ffmpeg(
            [
                "-loop",
                "1",
                "-i",
                str(png),
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=8",
                "-t",
                "8",
                "-r",
                "15",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                str(path),
            ]
        )
    finally:
        png.unlink(missing_ok=True)


_GENERATORS = {
    "color.mp4": _make_color,
    "cuts.mp4": _make_cuts,
    "text.mp4": _make_text,
}


def ensure_fixtures(directory: Path = FIXTURES_DIR) -> dict[str, Path]:
    """Create any missing synthetic fixtures and return {name: path}."""
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name, generator in _GENERATORS.items():
        path = directory / name
        if not path.exists():
            generator(path)
        result[name] = path
    return result


if __name__ == "__main__":
    for name, path in ensure_fixtures().items():
        print(f"{name}: {path}")
