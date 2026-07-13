"""Processing pipeline: stages, context, and the runner."""

from __future__ import annotations


def transitive_dependents(stages, target_name: str) -> set[str]:
    """Return ``target_name`` plus every stage that (transitively) depends on it."""
    result = {target_name}
    changed = True
    while changed:
        changed = False
        for stage in stages:
            if stage.name in result:
                continue
            if any(dep in result for dep in stage.depends_on):
                result.add(stage.name)
                changed = True
    return result


def default_stages():
    """Return the ordered default ingest pipeline.

    Heavy imports (scenedetect, imagehash) happen here rather than at module
    import time so lightweight commands like ``vidcp list`` start fast.
    """
    from vidcp.pipeline.stages.audio import AudioStage
    from vidcp.pipeline.stages.embed import EmbedStage
    from vidcp.pipeline.stages.embed_frames import EmbedFramesStage
    from vidcp.pipeline.stages.keyframes import KeyframesStage
    from vidcp.pipeline.stages.ocr import OcrStage
    from vidcp.pipeline.stages.probe import ProbeStage
    from vidcp.pipeline.stages.scenes import ScenesStage
    from vidcp.pipeline.stages.transcribe import TranscribeStage

    return [
        ProbeStage(),
        AudioStage(),
        TranscribeStage(),
        ScenesStage(),
        KeyframesStage(),
        OcrStage(),
        EmbedStage(),
        EmbedFramesStage(),
    ]
