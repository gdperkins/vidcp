"""Shared constants for the test suite.

Known fixture properties live here so tests assert against named values instead
of scattered magic numbers.
"""

# color.mp4 — silent testsrc2 pattern, no hard cuts.
COLOR_DURATION_S = 8.0
COLOR_WIDTH = 320
COLOR_HEIGHT = 240
COLOR_FPS = 15.0

# cuts.mp4 — four 2s segments concatenated (hard cuts) + sine audio.
CUTS_DURATION_S = 8.0
CUTS_CUT_POINTS_S = (2.0, 4.0, 6.0)
CUTS_SCENE_COUNT = 4

# text.mp4 — on-screen text on black + sine audio (used for OCR in Step 5).
TEXT_OVERLAY = "HELLO VIDCP 42"

# speech.mp4 — a committed fixture generated once with macOS `say`.
SPEECH_PHRASE = "Machine learning models can understand natural language."
# Distinctive substring whisper should transcribe reliably (even the tiny model).
SPEECH_EXPECTED_SUBSTRING = "natural language"
# Exact keyword expected to rank the speech segment #1 in keyword search (Step 6).
SPEECH_KEYWORD = "language"
# Semantic paraphrase sharing no distinctive keywords, for vector search (Step 6).
SPEECH_PARAPHRASE = "artificial intelligence understands human speech"
