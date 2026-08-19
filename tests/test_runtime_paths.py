from core.runtime_paths import (
    ENGINES_ROOT,
    PROJECT_ROOT,
    resolve_engine_path,
    resolve_project_path,
)


def test_relative_path_resolves_from_project_root():
    result = resolve_project_path(r"output\example.json")
    assert result == (PROJECT_ROOT / "output" / "example.json").resolve()


def test_absolute_path_is_preserved(tmp_path):
    target = tmp_path / "video.mp4"
    assert resolve_project_path(target) == target.resolve()


def test_empty_engine_path_uses_sibling_calibration_engines():
    result = resolve_engine_path("", "PnLCalib")
    assert result == (ENGINES_ROOT / "PnLCalib").resolve()


def test_absolute_engine_path_is_preserved(tmp_path):
    target = tmp_path / "PnLCalib"
    assert resolve_engine_path(target, "PnLCalib") == target.resolve()
