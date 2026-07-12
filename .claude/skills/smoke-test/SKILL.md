---
name: smoke-test
description: End-to-end sanity check of the vidcp pipeline against a throwaway library. Use after changing pipeline stages, the CLI, search, or the DB layer to verify the real flow works beyond unit tests — ingests a fixture video into a temp VIDCP_HOME and exercises search, transcript, and export.
---

Run the full vidcp flow against an isolated, disposable library. Never touch the user's real `~/.vidcp`.

## Steps

1. **Isolate**: create a temp home and use it for every command:
   ```bash
   SMOKE_HOME=$(mktemp -d)/vidcp-smoke
   export VIDCP_HOME="$SMOKE_HOME" VIDCP_WHISPER_MODEL=tiny
   ```
   The `tiny` model keeps transcription fast (CPU-only; first run may download it).

2. **Doctor**: `uv run vidcp doctor` — must pass (ffmpeg/ffprobe present). If it fails, stop and report; nothing else will work.

3. **Ingest** the committed fixture (contains real speech):
   ```bash
   uv run vidcp ingest tests/fixtures/speech.mp4
   ```
   Must exit 0. Capture the video ID from `uv run vidcp list --json`.

4. **Inspect stages**: `uv run vidcp inspect <id> --stages` — all stages should be complete (ocr may be skipped/empty for this fixture; that's fine).

5. **Exercise outputs**, checking each is non-empty and exits 0:
   - `uv run vidcp transcript <id>` — should contain recognizable words; pick one for the next step.
   - `uv run vidcp search "<word from transcript>" --json` — must return at least one hit for this video.
   - `uv run vidcp export <id> --format markdown -o "$SMOKE_HOME/export.md"` — file must exist and be non-empty.

6. **Cleanup**: `rm -rf "$(dirname "$SMOKE_HOME")"` — always, even on failure.

## Reporting

Report pass/fail per step with the actual command output on failure. A search that returns zero hits for a word visibly present in the transcript is a FAILURE even if the command exited 0.
