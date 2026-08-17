from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path) -> Path:
    """
    Resolve a path relative to the FootballAnalysisAI repository root.

    Absolute paths are preserved. This makes command-line scripts portable
    across Windows machines without hard-coded drive letters.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def default_input_video() -> Path:
    return PROJECT_ROOT / "input" / "input.mp4"


def default_detection_model() -> Path:
    return PROJECT_ROOT / "models" / "football-player-detection.pt"


def default_output(name: str) -> Path:
    return PROJECT_ROOT / "output" / name
