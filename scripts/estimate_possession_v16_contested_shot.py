from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from core.runtime_paths import resolve_project_path
from core.shot_context_image_goal import (
    UNKNOWN,
    PASS_FLIGHT,
    TEAM_FLIGHT,
    goal_image_geometry,
    normalized_ball_goal_distance,
)
from core.shot_window import (
    SHOT_FLIGHT,
    GOAL_ATTEMPT,
    ATTACKING_FLIGHT,
    LocalGoalSample,
    LocalShotWindowDetector,
)


CONTESTED_FLIGHT = "CONTESTED_FLIGHT"

# v1.6:
# Shot search is anchored to event-structured ball flights.
# RAW_LOOSE is intentionally excluded by default because the gsGol1 validation
# showed a post-event false positive at frames 201..205.
SHOT_CANDIDATE_PHASES = {
    PASS_FLIGHT,
    TEAM_FLIGHT,
    CONTESTED_FLIGHT,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "FootballAnalysisAI - Shot Context v1.6 "
            "Contested Flight Guard"
        )
    )
    p.add_argument("--source", default=r"input\input.mp4")
    p.add_argument("--possession-jsonl", required=True)
    p.add_argument("--direction-jsonl", required=True)
    p.add_argument("--team-jsonl", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--jsonl", required=True)
    return p.parse_args()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def read_jsonl_map(path: Path):
    return {
        int(r["frame_index"]): r
        for r in read_jsonl(path)
    }


def read_calibration(path: Path):
    rows = json.loads(
        path.read_text(encoding="utf-8")
    )
    return {
        int(r["frame_index"]): np.asarray(
            r["homography_image_to_pitch"],
            dtype=np.float64,
        )
        for r in rows
        if r.get("status") == "ok"
        and r.get("homography_image_to_pitch") is not None
    }


def resolved_directions(rows):
    for row in rows:
        ad = row.get("attack_direction") or {}
        a = (ad.get("TEAM_A") or {}).get("direction", UNKNOWN)
        b = (ad.get("TEAM_B") or {}).get("direction", UNKNOWN)
        if a != UNKNOWN or b != UNKNOWN:
            return {"TEAM_A": a, "TEAM_B": b}
    return {"TEAM_A": UNKNOWN, "TEAM_B": UNKNOWN}


def phase(row):
    return str(row.get("phase", "RAW_LOOSE"))


def source_team(row):
    team = row.get("source_team")
    if team in {"TEAM_A", "TEAM_B"}:
        return team

    state = row.get("team_state")
    if state in {"TEAM_A", "TEAM_B"}:
        return state

    return None


def ball_xy(row):
    xy = (row.get("ball") or {}).get("image_xy")
    if xy is None or len(xy) < 2:
        return None
    return float(xy[0]), float(xy[1])


def build_candidate_runs(rows):
    """
    Preserve event-run boundaries.

    For CONTESTED_FLIGHT, source_team is deliberately authoritative for shot
    direction. The target team represents the next controller, not the attacking
    team during the flight.
    """
    runs = []
    i = 0

    while i < len(rows):
        ph = phase(rows[i])

        if ph not in SHOT_CANDIDATE_PHASES:
            i += 1
            continue

        start = i
        team = source_team(rows[i])
        run_phase = ph

        i += 1

        while i < len(rows):
            if phase(rows[i]) != run_phase:
                break

            current_team = source_team(rows[i])

            if (
                team is not None
                and current_team is not None
                and current_team != team
            ):
                break

            if team is None and current_team is not None:
                team = current_team

            i += 1

        runs.append(
            (start, i - 1, team, run_phase)
        )

    return runs


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    possession_path = resolve_project_path(args.possession_jsonl)
    direction_path = resolve_project_path(args.direction_jsonl)
    team_path = resolve_project_path(args.team_jsonl)
    calibration_path = resolve_project_path(args.calibration_json)
    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)

    for path in (
        source,
        possession_path,
        direction_path,
        team_path,
        calibration_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = read_jsonl(possession_path)
    team_rows = read_jsonl_map(team_path)
    calibrations = read_calibration(calibration_path)
    directions = resolved_directions(read_jsonl(direction_path))

    detector = LocalShotWindowDetector()
    enhanced = [dict(r) for r in rows]

    detected = []
    run_debug = []

    for start, end, team, run_phase in build_candidate_runs(rows):
        if team not in {"TEAM_A", "TEAM_B"}:
            continue

        direction = directions.get(team, UNKNOWN)
        if direction == UNKNOWN:
            continue

        samples = []

        for row in rows[start:end + 1]:
            frame_index = int(row["frame_index"])
            bxy = ball_xy(row)
            team_row = team_rows.get(frame_index)

            if bxy is None or team_row is None:
                continue

            calibration_frame = team_row.get("pnl_calibration_frame")
            if calibration_frame is None:
                continue

            H = calibrations.get(int(calibration_frame))
            if H is None:
                continue

            geometry = goal_image_geometry(H, direction)
            if geometry is None:
                continue

            normalized = normalized_ball_goal_distance(
                bxy,
                geometry,
                6.0,
            )
            if normalized is None:
                continue

            samples.append(
                LocalGoalSample(
                    frame_index=frame_index,
                    normalized_goal_distance=float(normalized),
                )
            )

        best = detector.best_window(samples)

        run_debug.append(
            {
                "run_start": int(rows[start]["frame_index"]),
                "run_end": int(rows[end]["frame_index"]),
                "run_phase": run_phase,
                "source_team": team,
                "attack_direction": direction,
                "valid_samples": len(samples),
                "best": best.__dict__ if best is not None else None,
            }
        )

        if best is None:
            continue

        detected.append(
            (team, run_phase, best)
        )

        # Only the local window is relabeled.
        for row in enhanced:
            frame_index = int(row["frame_index"])
            if best.start_frame <= frame_index <= best.end_frame:
                row["phase_v16"] = best.classification
                row["team_state_v16"] = team
                row["local_shot_window_v16"] = best.__dict__
                row["shot_source_phase_v16"] = run_phase

    for row in enhanced:
        row.setdefault("phase_v16", phase(row))
        row.setdefault("team_state_v16", row.get("team_state", "UNKNOWN"))
        row.setdefault("local_shot_window_v16", None)
        row.setdefault("shot_source_phase_v16", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    by_frame = {
        int(row["frame_index"]): row
        for row in enhanced
    }

    last_frame = max(by_frame)
    frame_index = 0
    stats = Counter()

    colors = {
        "TEAM_A": (255, 90, 40),
        "TEAM_B": (40, 70, 255),
        "LOOSE": (0, 215, 255),
        "UNKNOWN": (150, 150, 150),
    }

    with jsonl_out.open("w", encoding="utf-8") as out:
        try:
            while frame_index <= last_frame:
                ok, frame = cap.read()
                if not ok:
                    break

                row = by_frame.get(frame_index)
                if row is None:
                    frame_index += 1
                    continue

                team = row.get("team_state_v16", "UNKNOWN")
                current_phase = row.get("phase_v16", "RAW_LOOSE")

                stats[f"team:{team}"] += 1
                stats[f"phase:{current_phase}"] += 1

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, 126),
                    (18, 18, 18),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Shot Context v1.6 | {team} | {current_phase}",
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    colors.get(team, (200, 200, 200)),
                    2,
                    cv2.LINE_AA,
                )

                window = row.get("local_shot_window_v16") or {}

                if window:
                    cv2.putText(
                        frame,
                        (
                            f"Source phase={row.get('shot_source_phase_v16')} | "
                            f"window={window['start_frame']}..{window['end_frame']} "
                            f"W={window['window_size']}"
                        ),
                        (18, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.47,
                        (235, 235, 235),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"closing={window['closing']:.2f} "
                            f"closest={window['closest_distance']:.2f} "
                            f"approach={window['approach_fraction']:.2f}"
                        ),
                        (18, 88),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (180, 180, 180),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    (
                        f"Directions: A={directions['TEAM_A']} "
                        f"B={directions['TEAM_B']} | "
                        f"RAW_LOOSE shot search disabled"
                    ),
                    (18, 116),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(frame)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                frame_index += 1

        finally:
            cap.release()
            writer.release()

    print("=" * 94)
    print("DONE - Shot Context v1.6 CONTESTED FLIGHT GUARD")
    print(
        f"Directions: A={directions['TEAM_A']} "
        f"B={directions['TEAM_B']}"
    )
    print("Detected local windows:")

    if not detected:
        print("  NONE")
    else:
        for team, run_phase, window in detected:
            print(
                f"  {team} | {window.classification:<16} "
                f"| source_phase={run_phase:<18} "
                f"| frames={window.start_frame}..{window.end_frame} "
                f"| W={window.window_size} "
                f"| closing={window.closing:.3f} "
                f"| approach={window.approach_fraction:.2f} "
                f"| closest={window.closest_distance:.3f}"
            )

    print()
    print("Candidate run diagnostics:")
    for item in run_debug:
        best = item["best"]
        if best is None:
            best_text = "NONE"
        else:
            best_text = (
                f"{best['classification']} "
                f"{best['start_frame']}..{best['end_frame']}"
            )

        print(
            f"  {item['run_start']}..{item['run_end']} "
            f"| phase={item['run_phase']:<18} "
            f"| team={str(item['source_team']):<7} "
            f"| dir={item['attack_direction']:<7} "
            f"| samples={item['valid_samples']:<3} "
            f"| best={best_text}"
        )

    print(f"Video output: {output}")
    print(f"JSONL output: {jsonl_out}")
    print("=" * 94)


if __name__ == "__main__":
    main()
