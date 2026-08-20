
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PipelinePaths:
    run_dir: Path
    tracking_video: Path
    tracking_jsonl: Path
    team_video: Path
    team_jsonl: Path
    calibration_json: Path
    calibration_frames_dir: Path
    ball_video: Path
    ball_jsonl: Path
    possession_v11_video: Path
    possession_v11_jsonl: Path
    possession_v12_video: Path
    possession_v12_jsonl: Path
    direction_video: Path
    direction_jsonl: Path
    tactical_video: Path
    tactical_jsonl: Path
    shape_video: Path
    shape_jsonl: Path
    shot_video: Path
    shot_jsonl: Path
    analysis_jsonl: Path
    analysis_video: Path
    manifest_json: Path


@dataclass(frozen=True)
class PipelineStep:
    name: str
    module: str
    args: tuple[str, ...]
    outputs: tuple[Path, ...]


def build_paths(project_root: Path, run_name: str) -> PipelinePaths:
    run_dir = project_root / "output" / "pipeline_v1" / run_name

    return PipelinePaths(
        run_dir=run_dir,
        tracking_video=run_dir / "01_tracking.mp4",
        tracking_jsonl=run_dir / "01_tracking.jsonl",
        team_video=run_dir / "02_team_v25.mp4",
        team_jsonl=run_dir / "02_team_v25.jsonl",
        calibration_json=run_dir / "02_team_v25_calibration.json",
        calibration_frames_dir=run_dir / "02_pnl_frames",
        ball_video=run_dir / "03_ball_v1.mp4",
        ball_jsonl=run_dir / "03_ball_v1.jsonl",
        possession_v11_video=run_dir / "04_possession_v11.mp4",
        possession_v11_jsonl=run_dir / "04_possession_v11.jsonl",
        possession_v12_video=run_dir / "05_possession_v12.mp4",
        possession_v12_jsonl=run_dir / "05_possession_v12.jsonl",
        direction_video=run_dir / "06_direction_v11.mp4",
        direction_jsonl=run_dir / "06_direction_v11.jsonl",
        tactical_video=run_dir / "07_tactical_v11.mp4",
        tactical_jsonl=run_dir / "07_tactical_v11.jsonl",
        shape_video=run_dir / "08_shape_space_v1.mp4",
        shape_jsonl=run_dir / "08_shape_space_v1.jsonl",
        shot_video=run_dir / "09_shot_v16.mp4",
        shot_jsonl=run_dir / "09_shot_v16.jsonl",
        analysis_jsonl=run_dir / "analysis_v1.jsonl",
        analysis_video=run_dir / "analysis_v1.mp4",
        manifest_json=run_dir / "manifest.json",
    )


def project_relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_steps(
    *,
    project_root: Path,
    source: Path,
    paths: PipelinePaths,
    device: str,
    max_frames: int,
    imgsz: int,
    calibration_stride: int,
    max_calibration_gap: int,
    max_bridge_gap: int,
    max_unresolved_flight: int,
) -> list[PipelineStep]:
    src = project_relative(project_root, source)

    def p(path: Path) -> str:
        return project_relative(project_root, path)

    common_max = str(max_frames)

    return [
        PipelineStep(
            name="tracking",
            module="scripts.track_players",
            args=(
                "--source", src,
                "--imgsz", str(imgsz),
                "--max-frames", common_max,
                "--output", p(paths.tracking_video),
                "--jsonl", p(paths.tracking_jsonl),
            ),
            outputs=(paths.tracking_video, paths.tracking_jsonl),
        ),
        PipelineStep(
            name="team_v25",
            module="scripts.classify_teams_v25_pnl_exact",
            args=(
                "--source", src,
                "--tracking", p(paths.tracking_jsonl),
                "--device", device,
                "--max-frames", common_max,
                "--calibration-stride", str(calibration_stride),
                "--max-calibration-gap", str(max_calibration_gap),
                "--output", p(paths.team_video),
                "--jsonl", p(paths.team_jsonl),
                "--calibration-json", p(paths.calibration_json),
                "--calibration-frames-dir", p(paths.calibration_frames_dir),
            ),
            outputs=(
                paths.team_video,
                paths.team_jsonl,
                paths.calibration_json,
            ),
        ),
        PipelineStep(
            name="ball",
            module="scripts.track_ball",
            args=(
                "--source", src,
                "--device", device,
                "--max-frames", common_max,
                "--output", p(paths.ball_video),
                "--jsonl", p(paths.ball_jsonl),
            ),
            outputs=(paths.ball_video, paths.ball_jsonl),
        ),
        PipelineStep(
            name="possession_v11",
            module="scripts.estimate_possession_v11",
            args=(
                "--source", src,
                "--ball-jsonl", p(paths.ball_jsonl),
                "--team-jsonl", p(paths.team_jsonl),
                "--calibration-json", p(paths.calibration_json),
                "--output", p(paths.possession_v11_video),
                "--jsonl", p(paths.possession_v11_jsonl),
            ),
            outputs=(paths.possession_v11_video, paths.possession_v11_jsonl),
        ),
        PipelineStep(
            name="possession_v12",
            module="scripts.estimate_possession_v12_events",
            args=(
                "--source", src,
                "--possession-jsonl", p(paths.possession_v11_jsonl),
                "--max-bridge-gap", str(max_bridge_gap),
                "--max-unresolved-flight", str(max_unresolved_flight),
                "--min-ball-motion-px", "8",
                "--max-missing-ball-ratio", "0.45",
                "--output", p(paths.possession_v12_video),
                "--jsonl", p(paths.possession_v12_jsonl),
            ),
            outputs=(paths.possession_v12_video, paths.possession_v12_jsonl),
        ),
        PipelineStep(
            name="direction",
            module="scripts.attack_direction_defline_v11",
            args=(
                "--source", src,
                "--team-jsonl", p(paths.team_jsonl),
                "--output", p(paths.direction_video),
                "--jsonl", p(paths.direction_jsonl),
                "--max-frames", common_max,
            ),
            outputs=(paths.direction_video, paths.direction_jsonl),
        ),
        PipelineStep(
            name="tactical",
            module="scripts.tactical_engine_v1",
            args=(
                "--source", src,
                "--team-jsonl", p(paths.team_jsonl),
                "--possession-jsonl", p(paths.possession_v11_jsonl),
                "--output", p(paths.tactical_video),
                "--jsonl", p(paths.tactical_jsonl),
            ),
            outputs=(paths.tactical_video, paths.tactical_jsonl),
        ),
        PipelineStep(
            name="shape_space",
            module="scripts.team_shape_space_v1",
            args=(
                "--source", src,
                "--team-jsonl", p(paths.team_jsonl),
                "--possession-jsonl", p(paths.possession_v11_jsonl),
                "--output", p(paths.shape_video),
                "--jsonl", p(paths.shape_jsonl),
            ),
            outputs=(paths.shape_video, paths.shape_jsonl),
        ),
        PipelineStep(
            name="shot_v16",
            module="scripts.estimate_possession_v16_contested_shot",
            args=(
                "--source", src,
                "--possession-jsonl", p(paths.possession_v12_jsonl),
                "--direction-jsonl", p(paths.direction_jsonl),
                "--team-jsonl", p(paths.team_jsonl),
                "--calibration-json", p(paths.calibration_json),
                "--output", p(paths.shot_video),
                "--jsonl", p(paths.shot_jsonl),
            ),
            outputs=(paths.shot_video, paths.shot_jsonl),
        ),
    ]


def outputs_ready(paths: Iterable[Path]) -> bool:
    paths = list(paths)
    return bool(paths) and all(
        p.exists() and p.is_file() and p.stat().st_size > 0
        for p in paths
    )
