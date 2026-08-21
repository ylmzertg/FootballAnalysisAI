from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from core.runtime_paths import resolve_project_path
from core.shot_context import (
    PLUS_X,
    MINUS_X,
    UNKNOWN,
    RAW_LOOSE,
    TEAM_FLIGHT,
    PASS_FLIGHT,
    SHOT_FLIGHT,
    GOAL_ATTEMPT,
    ATTACKING_FLIGHT,
    ShotFrame,
    ShotContextClassifier,
    ShotContextConfig,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Possession v1.3 Shot Context"
    )
    p.add_argument("--source", default=r"input\input.mp4")
    p.add_argument(
        "--possession-jsonl",
        default=r"output\possession_v12_events.jsonl",
    )
    p.add_argument(
        "--direction-jsonl",
        default=r"output\attack_direction_defline_v11.jsonl",
    )
    p.add_argument(
        "--output",
        default=r"output\possession_v13_shot_context.mp4",
    )
    p.add_argument(
        "--jsonl",
        default=r"output\possession_v13_shot_context.jsonl",
    )
    p.add_argument("--min-forward-progress-m", type=float, default=8.0)
    p.add_argument("--shot-goal-distance-m", type=float, default=25.0)
    p.add_argument("--goal-attempt-distance-m", type=float, default=12.0)
    return p.parse_args()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def resolved_directions(rows):
    for row in rows:
        ad = row.get("attack_direction") or {}
        a = (ad.get("TEAM_A") or {}).get("direction", UNKNOWN)
        b = (ad.get("TEAM_B") or {}).get("direction", UNKNOWN)
        if a != UNKNOWN or b != UNKNOWN:
            return {"TEAM_A": a, "TEAM_B": b}
    return {"TEAM_A": UNKNOWN, "TEAM_B": UNKNOWN}


def is_candidate_phase(row):
    return str(row.get("phase")) in {
        RAW_LOOSE,
        TEAM_FLIGHT,
        PASS_FLIGHT,
    }


def source_team(row):
    team = row.get("source_team")
    if team in {"TEAM_A", "TEAM_B"}:
        return team
    state = row.get("team_state")
    if state in {"TEAM_A", "TEAM_B"}:
        return state
    return None


def ball_pitch(row):
    ball = row.get("ball") or {}
    xy = ball.get("pitch_xy")
    if xy is None or len(xy) < 2:
        return None
    return float(xy[0]), float(xy[1])


def build_runs(rows):
    runs = []
    i = 0
    while i < len(rows):
        if not is_candidate_phase(rows[i]):
            i += 1
            continue

        start = i
        team = source_team(rows[i])

        i += 1
        while i < len(rows):
            if not is_candidate_phase(rows[i]):
                break
            current_team = source_team(rows[i])
            if team is not None and current_team is not None and current_team != team:
                break
            if team is None and current_team is not None:
                team = current_team
            i += 1

        runs.append((start, i - 1))
    return runs


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    possession_path = resolve_project_path(args.possession_jsonl)
    direction_path = resolve_project_path(args.direction_jsonl)
    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)

    for p in (source, possession_path, direction_path):
        if not p.exists():
            raise FileNotFoundError(p)

    rows = read_jsonl(possession_path)
    direction_rows = read_jsonl(direction_path)
    directions = resolved_directions(direction_rows)

    classifier = ShotContextClassifier(
        ShotContextConfig(
            min_forward_progress_m=max(1.0, args.min_forward_progress_m),
            shot_goal_distance_m=max(5.0, args.shot_goal_distance_m),
            goal_attempt_distance_m=max(2.0, args.goal_attempt_distance_m),
        )
    )

    enhanced = [dict(r) for r in rows]
    run_results = []

    for start, end in build_runs(rows):
        run_rows = rows[start:end + 1]
        team = source_team(run_rows[0])

        direction = (
            directions.get(team, UNKNOWN)
            if team is not None
            else UNKNOWN
        )

        frames = [
            ShotFrame(
                frame_index=int(r["frame_index"]),
                phase=str(r.get("phase", RAW_LOOSE)),
                team_state=str(r.get("team_state", "LOOSE")),
                source_team=source_team(r),
                source_owner_track_id=(
                    int(r["source_owner_track_id"])
                    if r.get("source_owner_track_id") is not None
                    else None
                ),
                ball_pitch_xy=ball_pitch(r),
            )
            for r in run_rows
        ]

        result = classifier.classify_run(
            frames,
            direction,
        )
        run_results.append(result)

        if result.classification in {
            SHOT_FLIGHT,
            GOAL_ATTEMPT,
            ATTACKING_FLIGHT,
        }:
            for idx in range(start, end + 1):
                enhanced[idx]["phase_v13"] = result.classification
                if result.team in {"TEAM_A", "TEAM_B"}:
                    enhanced[idx]["team_state_v13"] = result.team
                else:
                    enhanced[idx]["team_state_v13"] = enhanced[idx].get(
                        "team_state",
                        "LOOSE",
                    )
                enhanced[idx]["shot_context"] = result.__dict__
        else:
            for idx in range(start, end + 1):
                enhanced[idx]["phase_v13"] = enhanced[idx].get("phase")
                enhanced[idx]["team_state_v13"] = enhanced[idx].get("team_state")
                enhanced[idx]["shot_context"] = result.__dict__

    # Non-run rows preserve original semantics.
    for row in enhanced:
        row.setdefault("phase_v13", row.get("phase"))
        row.setdefault("team_state_v13", row.get("team_state"))
        row.setdefault("shot_context", None)

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
        int(r["frame_index"]): r
        for r in enhanced
    }
    last_frame = max(by_frame)
    stats = Counter()
    idx = 0

    colors = {
        "TEAM_A": (255, 90, 40),
        "TEAM_B": (40, 70, 255),
        "LOOSE": (0, 215, 255),
        "UNKNOWN": (150, 150, 150),
    }

    with jsonl_out.open("w", encoding="utf-8") as out:
        try:
            while idx <= last_frame:
                ok, frame = cap.read()
                if not ok:
                    break

                row = by_frame.get(idx)
                if row is None:
                    idx += 1
                    continue

                team = row.get("team_state_v13", row.get("team_state", "UNKNOWN"))
                phase = row.get("phase_v13", row.get("phase"))
                stats[f"team:{team}"] += 1
                stats[f"phase:{phase}"] += 1

                color = colors.get(team, (200, 200, 200))

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, 118),
                    (18, 18, 18),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Possession v1.3 | {team} | {phase}",
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                context = row.get("shot_context") or {}
                if context:
                    progress = float(context.get("max_forward_progress_m", 0.0))
                    goal_d = context.get("end_goal_distance_m")
                    goal_text = (
                        f"{float(goal_d):.1f}m"
                        if goal_d is not None
                        else "N/A"
                    )
                    cv2.putText(
                        frame,
                        (
                            f"Attack dir={directions.get(team, UNKNOWN)} | "
                            f"max progress={progress:.1f}m | "
                            f"end goal dist={goal_text}"
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
                        str(context.get("reason", "")),
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
                        f"B={directions['TEAM_B']}"
                    ),
                    (18, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(frame)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                idx += 1
        finally:
            cap.release()
            writer.release()

    print("=" * 88)
    print("DONE - Possession v1.3 SHOT CONTEXT")
    print(f"TEAM_A team-state : {stats['team:TEAM_A']}")
    print(f"TEAM_B team-state : {stats['team:TEAM_B']}")
    print(f"LOOSE             : {stats['team:LOOSE']}")
    print(f"UNKNOWN           : {stats['team:UNKNOWN']}")
    print("Phases:")
    for phase in (
        "CONTROL",
        "CONTROL_GAP",
        "PASS_FLIGHT",
        "TEAM_FLIGHT",
        "CONTESTED_FLIGHT",
        "ATTACKING_FLIGHT",
        "SHOT_FLIGHT",
        "GOAL_ATTEMPT",
        "RAW_LOOSE",
        "RAW_UNKNOWN",
    ):
        print(f"  {phase:<20} {stats[f'phase:{phase}']}")
    print("Run classifications:")
    rc = Counter(r.classification for r in run_results)
    for key, value in rc.most_common():
        print(f"  {key:<20} {value}")
    print(f"Video output       : {output}")
    print(f"JSONL output       : {jsonl_out}")
    print("=" * 88)


if __name__ == "__main__":
    main()
