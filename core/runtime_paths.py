from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINES_ROOT = PROJECT_ROOT.parent / "CalibrationEngines"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_engine_path(value: str | Path | None, engine_name: str) -> Path:
    if value is None or not str(value).strip():
        return (ENGINES_ROOT / engine_name).resolve()

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
