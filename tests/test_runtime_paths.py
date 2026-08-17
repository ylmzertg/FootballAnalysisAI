from pathlib import Path

from core.runtime_paths import PROJECT_ROOT, resolve_project_path


def test_relative_path_resolves_from_project_root():
    result = resolve_project_path(r"output\example.json")
    assert result == (PROJECT_ROOT / "output" / "example.json").resolve()


def test_absolute_path_is_preserved(tmp_path):
    target = tmp_path / "video.mp4"
    assert resolve_project_path(target) == target.resolve()
