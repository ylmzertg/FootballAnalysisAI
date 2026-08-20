
from pathlib import Path

from core.analysis_pipeline import (
    build_paths,
    build_steps,
    outputs_ready,
)


def test_pipeline_paths_are_run_scoped(tmp_path):
    p = build_paths(tmp_path, "goal_test")

    assert p.run_dir == tmp_path / "output" / "pipeline_v1" / "goal_test"
    assert p.tracking_jsonl.parent == p.run_dir
    assert p.analysis_video.parent == p.run_dir


def test_pipeline_step_order(tmp_path):
    paths = build_paths(tmp_path, "goal_test")

    steps = build_steps(
        project_root=tmp_path,
        source=tmp_path / "input" / "goal.mp4",
        paths=paths,
        device="auto",
        max_frames=250,
        imgsz=640,
        calibration_stride=15,
        max_calibration_gap=8,
        max_bridge_gap=75,
        max_unresolved_flight=50,
    )

    assert [s.name for s in steps] == [
        "tracking",
        "team_v25",
        "ball",
        "possession_v11",
        "possession_v12",
        "direction",
        "tactical",
        "shape_space",
        "shot_v16",
    ]


def test_outputs_ready_requires_nonempty_files(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"

    assert not outputs_ready([a, b])

    a.write_text("a", encoding="utf-8")
    b.write_text("", encoding="utf-8")
    assert not outputs_ready([a, b])

    b.write_text("b", encoding="utf-8")
    assert outputs_ready([a, b])
