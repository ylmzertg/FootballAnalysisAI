
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge FootballAnalysisAI pipeline outputs into one JSONL."
    )
    p.add_argument("--team-jsonl", required=True)
    p.add_argument("--ball-jsonl", required=True)
    p.add_argument("--possession-control-jsonl", required=True)
    p.add_argument("--possession-event-jsonl", required=True)
    p.add_argument("--direction-jsonl", required=True)
    p.add_argument("--tactical-jsonl", required=True)
    p.add_argument("--shape-jsonl", required=True)
    p.add_argument("--shot-jsonl", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def read_map(path: Path):
    result = {}
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            result[int(row["frame_index"])] = row

    return result


def main():
    args = parse_args()

    paths = {
        "team": resolve_project_path(args.team_jsonl),
        "ball": resolve_project_path(args.ball_jsonl),
        "possession_control": resolve_project_path(args.possession_control_jsonl),
        "possession_event": resolve_project_path(args.possession_event_jsonl),
        "direction": resolve_project_path(args.direction_jsonl),
        "tactical": resolve_project_path(args.tactical_jsonl),
        "shape_space": resolve_project_path(args.shape_jsonl),
        "shot": resolve_project_path(args.shot_jsonl),
    }

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        name: read_map(path)
        for name, path in paths.items()
    }

    all_frames = sorted(
        set().union(
            *(set(rows.keys()) for rows in data.values())
        )
    )

    if not all_frames:
        raise RuntimeError("No frames found in pipeline outputs.")

    with output.open("w", encoding="utf-8") as out:
        for frame_index in all_frames:
            team = data["team"].get(frame_index) or {}
            ball = data["ball"].get(frame_index) or {}
            control = data["possession_control"].get(frame_index) or {}
            event = data["possession_event"].get(frame_index) or {}
            direction = data["direction"].get(frame_index) or {}
            tactical = data["tactical"].get(frame_index) or {}
            shape = data["shape_space"].get(frame_index) or {}
            shot = data["shot"].get(frame_index) or {}

            timestamp = (
                team.get("timestamp_seconds")
                or ball.get("timestamp_seconds")
                or control.get("timestamp_seconds")
                or event.get("timestamp_seconds")
                or frame_index
            )

            payload = {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "tracks": team.get("tracks", []),
                "calibration_frame": team.get("pnl_calibration_frame"),
                "ball": {
                    "center_xy": ball.get("center_xy"),
                    "bbox_xyxy": ball.get("bbox_xyxy"),
                    "confidence": ball.get("confidence"),
                    "detected": ball.get("detected"),
                    "predicted": ball.get("predicted"),
                    "gap_frames": ball.get("gap_frames"),
                    "candidate_count": ball.get("candidate_count"),
                },
                "possession_control": control.get("possession"),
                "possession_event": {
                    "team_state": event.get("team_state"),
                    "phase": event.get("phase"),
                    "possessor_track_id": event.get("possessor_track_id"),
                    "source_owner_track_id": event.get("source_owner_track_id"),
                    "target_owner_track_id": event.get("target_owner_track_id"),
                    "source_team": event.get("source_team"),
                    "target_team": event.get("target_team"),
                    "confidence": event.get("confidence"),
                    "reason": event.get("reason"),
                },
                "attack_direction": direction.get("attack_direction"),
                "defensive_line": direction.get("defensive_line"),
                "tactical": {
                    "possession_state": tactical.get("possession_state"),
                    "possessor_track_id": tactical.get("possessor_track_id"),
                    "pressure": tactical.get("pressure"),
                    "passing_lanes": tactical.get("passing_lanes", []),
                    "best_open_receiver_ids": tactical.get(
                        "best_open_receiver_ids", []
                    ),
                },
                "shape_space": {
                    "possession_team": shape.get("possession_team"),
                    "possessor_track_id": shape.get("possessor_track_id"),
                    "team_shapes": shape.get("team_shapes"),
                    "spaces": shape.get("spaces", []),
                },
                "shot": {
                    "phase": shot.get("phase_v16", shot.get("phase")),
                    "team_state": shot.get(
                        "team_state_v16", shot.get("team_state")
                    ),
                    "local_window": shot.get("local_shot_window_v16"),
                    "source_phase": shot.get("shot_source_phase_v16"),
                },
            }

            out.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Merged frames : {len(all_frames)}")
    print(f"Output        : {output}")


if __name__ == "__main__":
    main()
