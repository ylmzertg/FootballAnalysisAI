
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from core.runtime_paths import resolve_project_path


TEAM_COLORS = {
    "TEAM_A": (255, 90, 40),
    "TEAM_B": (40, 70, 255),
    "UNKNOWN": (150, 150, 150),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Render a compact combined FootballAnalysisAI analysis video."
    )
    p.add_argument("--source", required=True)
    p.add_argument("--analysis-jsonl", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def read_map(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["frame_index"])] = row
    return rows


def assignment(track):
    return track.get("team_v25") or track.get("team_v24") or {}


def foot_point(track):
    bbox = track.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return int(round((x1 + x2) / 2)), int(round(y2))


def draw_team_boxes(frame, tracks, possessor_id):
    track_map = {}

    for track in tracks:
        tid = int(track.get("track_id", -1))
        bbox = track.get("bbox_xyxy")
        a = assignment(track)

        if tid < 0 or not bbox or len(bbox) != 4:
            continue

        team = str(a.get("team", "UNKNOWN"))
        role = str(a.get("role", "PLAYER")).upper()

        if role in {"REFEREE", "OUTSIDE_PITCH"}:
            continue

        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        color = TEAM_COLORS.get(team, TEAM_COLORS["UNKNOWN"])
        thickness = 3 if tid == possessor_id else 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )

        if tid == possessor_id:
            cv2.putText(
                frame,
                f"OWNER {tid}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

        track_map[tid] = track

    return track_map


def draw_pass_lanes(frame, tactical, track_map):
    possessor_id = tactical.get("possessor_track_id")
    if possessor_id is None:
        return

    possessor = track_map.get(int(possessor_id))
    start = foot_point(possessor) if possessor else None

    if start is None:
        return

    open_count = 0

    for lane in tactical.get("passing_lanes", []):
        status = lane.get("status")
        if status == "OPEN" and open_count >= 4:
            continue

        receiver = track_map.get(int(lane.get("receiver_track_id", -1)))
        end = foot_point(receiver) if receiver else None
        if end is None:
            continue

        if status == "OPEN":
            color = (70, 230, 70)
            thickness = 2
            open_count += 1
        else:
            color = (70, 70, 220)
            thickness = 1

        cv2.line(
            frame,
            start,
            end,
            color,
            thickness,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    analysis_path = resolve_project_path(args.analysis_jsonl)
    output = resolve_project_path(args.output)

    for path in (source, analysis_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = read_map(analysis_path)
    if not rows:
        raise RuntimeError("Analysis JSONL is empty.")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    last = max(rows)
    frame_index = 0

    try:
        while frame_index <= last:
            ok, frame = cap.read()
            if not ok:
                break

            row = rows.get(frame_index)

            if row is None:
                writer.write(frame)
                frame_index += 1
                continue

            control = row.get("possession_control") or {}
            event = row.get("possession_event") or {}
            tactical = row.get("tactical") or {}
            shape = row.get("shape_space") or {}
            shot = row.get("shot") or {}
            ball = row.get("ball") or {}

            possessor_id = control.get("possessor_track_id")

            track_map = draw_team_boxes(
                frame,
                row.get("tracks", []),
                possessor_id,
            )
            draw_pass_lanes(frame, tactical, track_map)

            center = ball.get("center_xy")
            if center is not None and len(center) >= 2:
                bx, by = int(round(center[0])), int(round(center[1]))
                shot_phase = str(shot.get("phase") or "")
                color = (
                    (0, 80, 255)
                    if shot_phase == "SHOT_FLIGHT"
                    else (0, 255, 255)
                )
                cv2.circle(
                    frame,
                    (bx, by),
                    10,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            header_h = 142
            cv2.rectangle(
                frame,
                (0, 0),
                (width, header_h),
                (18, 18, 18),
                -1,
            )

            team_state = (
                shot.get("team_state")
                or event.get("team_state")
                or control.get("state")
                or "UNKNOWN"
            )
            phase = (
                shot.get("phase")
                or event.get("phase")
                or "UNKNOWN"
            )

            cv2.putText(
                frame,
                f"Analysis v1 | {team_state} | {phase}",
                (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TEAM_COLORS.get(team_state, (230, 230, 230)),
                2,
                cv2.LINE_AA,
            )

            pressure = tactical.get("pressure") or {}
            pressure_level = pressure.get("level", "N/A")
            nearest_pressure = pressure.get("nearest_opponent_distance_m")
            nearest_text = (
                f"{float(nearest_pressure):.2f}m"
                if nearest_pressure is not None
                else "N/A"
            )

            lanes = tactical.get("passing_lanes", [])
            open_lanes = sum(1 for x in lanes if x.get("status") == "OPEN")
            blocked_lanes = sum(
                1 for x in lanes if x.get("status") == "BLOCKED"
            )

            cv2.putText(
                frame,
                (
                    f"Owner={possessor_id} | Pressure={pressure_level} "
                    f"nearest={nearest_text} | Pass lanes "
                    f"open={open_lanes} blocked={blocked_lanes}"
                ),
                (18, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            shapes = shape.get("team_shapes") or {}
            a = shapes.get("TEAM_A") or {}
            b = shapes.get("TEAM_B") or {}

            cv2.putText(
                frame,
                (
                    f"Shape A: W={float(a.get('width_m', 0)):.1f} "
                    f"D={float(a.get('depth_m', 0)):.1f} "
                    f"C={float(a.get('compactness_m', 0)):.1f} | "
                    f"B: W={float(b.get('width_m', 0)):.1f} "
                    f"D={float(b.get('depth_m', 0)):.1f} "
                    f"C={float(b.get('compactness_m', 0)):.1f}"
                ),
                (18, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (190, 210, 190),
                1,
                cv2.LINE_AA,
            )

            direction = row.get("attack_direction") or {}
            da = (direction.get("TEAM_A") or {}).get("direction", "UNKNOWN")
            db = (direction.get("TEAM_B") or {}).get("direction", "UNKNOWN")

            shot_window = shot.get("local_window") or {}
            shot_text = ""
            if shot_window:
                shot_text = (
                    f" | ShotWin={shot_window.get('start_frame')}.."
                    f"{shot_window.get('end_frame')} "
                    f"closing={float(shot_window.get('closing', 0)):.2f}"
                )

            cv2.putText(
                frame,
                f"Directions A={da} B={db}{shot_text}",
                (18, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Frame {frame_index}",
                (18, 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    print(f"Rendered frames : {frame_index}")
    print(f"Output          : {output}")


if __name__ == "__main__":
    main()
