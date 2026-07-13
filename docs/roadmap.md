# Roadmap

Ranked candidate features and improvements, ordered by value relative to effort
for a local-first, CPU-only, agent-friendly tool. (Snapshot as of 2026-07-13;
vidcp v0.2.)

## Ranked list

1. **CLIP image embeddings for keyframes (visual semantic search).** The
   biggest capability gap: today search only finds what was *said* or *shown
   as text*. Embedding keyframes with a CLIP-style model (e.g.
   `clip-ViT-B-32` via sentence-transformers, already a dependency) enables
   searches like "whiteboard diagram" or "person at a podium". Slots into the
   existing `embed` stage and sqlite-vec table; RRF can fuse a third leg.
   Medium effort, transforms what the tool is.

2. **Clip extraction — `vidcp clip <id> --from --to -o out.mp4`, plus an MCP
   `get_clip` tool.** Search currently ends at a timestamp; this makes hits
   actionable (share the moment, feed it to another tool). ffmpeg
   stream-copy makes it nearly free. Low effort, high payoff.

3. **Batch/directory sync — `vidcp sync ~/Videos`.** Scan a directory, ingest
   anything whose hash isn't in the library, skip the rest. Turns the library
   from "things I remembered to ingest" into "everything I have". Low effort
   since hashing and resume already exist; a `--watch` daemon can come later.

4. **Tags and search filters.** `vidcp tag <id> +conference`,
   `search --tag conference`, plus a tag filter on the MCP `search` tool.
   Once the library grows past a few dozen videos, organization becomes the
   bottleneck. Small schema migration + query filters.

5. **`vidcp play <id> --at 12:30`.** Open the source file in
   mpv/IINA/QuickTime at a timestamp (mpv's `--start` makes this trivial).
   Closes the loop from search hit to actually watching the moment.

6. **Word-level timestamps from faster-whisper.** Sharpens search-hit
   precision (jump to the word, not the 10-second segment) and enables better
   snippets and karaoke-style VTT. Mostly a transcribe-stage config change
   plus schema for word offsets.

7. **Speaker diarization.** "Who said what" matters for meetings, interviews,
   and podcasts. Good local diarization (pyannote, speechbrain clustering)
   brings heavy dependencies and is slow on CPU, so gate it behind an
   optional extra and a `--diarize` flag.

8. **URL ingest via yt-dlp (optional extra).** `vidcp ingest <url>` downloads
   then runs the normal pipeline. Keeping it an optional dependency preserves
   the lean core.

9. **Whisper language options.** Expose `--language` and `--translate`
   (whisper natively translates to English). Cheap win for non-English
   libraries.

10. **Local web UI — `vidcp serve`.** Browse the library, keyframe strips per
    video, click a search hit to play at that timestamp in-browser. The
    biggest UX jump available, but also the biggest scope jump — ranks higher
    once items 1–5 exist to power it.

11. **Faster transcription backends.** Config passthrough for CUDA on Linux
    is nearly free; a whisper.cpp backend would unlock Metal on Apple Silicon
    but is a bigger integration.

12. **More MCP surface.** A `get_contact_sheet` tool (grid of keyframes for a
    whole video) would let agents "skim" a video in one image; also
    `delete_video` / `reindex` / `get_stats` for full agent-side library
    management.

13. **Optional local-LLM enrichment.** Summaries, chapter titles, and topic
    tags via an Ollama/OpenAI-compatible endpoint. Ranked last because the
    MCP server already lets an agent do this on demand — baking it in mostly
    adds precomputation, not new capability.

## v0.2 milestone (shipped)

Items **1 + 2 + 3** shipped together as v0.2: the library became visually
searchable (CLIP keyframe embeddings), hits became extractable clips
(`vidcp clip` / `get_clip`), and filling the library became frictionless
(`vidcp sync`). The remaining ranked list (4–13) should be re-evaluated and
re-ranked for v0.3 now that those three are done.
