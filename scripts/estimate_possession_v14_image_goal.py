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
    RAW_LOOSE,
    TEAM_FLIGHT,
    PASS_FLIGHT,
    ATTACKING_FLIGHT,
    SHOT_FLIGHT,
    GOAL_ATTEMPT,
    GoalApproachSample,
    ImageGoalShotClassifier,
    ImageGoalShotConfig,
    goal_image_geometry,
    normalized_ball_goal_distance,
)


CANDIDATE_PHASES = {
    RAW_LOOSE,
    TEAM_FLIGHT,
    PASS_FLIGHT,
    ATTACKING_FLIGHT,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "FootballAnalysisAI - Shot Context v1.4 "
            "Image Goal Geometry"
        )
    )

    p.add_argument(
        "--source",
        default=r"input\input.mp4",
    )
    p.add_argument(
        "--possession-jsonl",
        default=r"output\possession_v12_events.jsonl",
    )
    p.add_argument(
        "--direction-jsonl",
        default=r"output\attack_direction_defline_v11.jsonl",
    )
    p.add_argument(
        "--team-jsonl",
        default=r"output\team_classification_v25_pnl_exact.jsonl",
    )
    p.add_argument(
        "--calibration-json",
        default=r"output\team_classification_v25_pnl_exact_calibration.json",
    )
    p.add_argument(
        "--output",
        default=r"output\possession_v14_image_goal.mp4",
    )
    p.add_argument(
        "--jsonl",
        default=r"output\possession_v14_image_goal.jsonl",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=3,
    )
    p.add_argument(
        "--min-goal-mouth-px",
        type=float,
        default=6.0,
    )
    return p.parse_args()


def read_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    rows.sort(
        key=lambda r: int(
            r["frame_index"]
        )
    )
    return rows


def read_jsonl_map(path: Path):
    return {
        int(row["frame_index"]): row
        for row in read_jsonl(path)
    }


def read_calibration(path: Path):
    rows = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    result = {}

    for item in rows:
        if item.get("status") != "ok":
            continue

        H = item.get(
            "homography_image_to_pitch"
        )

        if H is None:
            continue

        result[
            int(item["frame_index"])
        ] = np.asarray(
            H,
            dtype=np.float64,
        )

    return result


def resolved_directions(rows):
    for row in rows:
        ad = (
            row.get("attack_direction")
            or {}
        )

        a = (
            ad.get("TEAM_A")
            or {}
        ).get(
            "direction",
            UNKNOWN,
        )

        b = (
            ad.get("TEAM_B")
            or {}
        ).get(
            "direction",
            UNKNOWN,
        )

        if (
            a != UNKNOWN
            or b != UNKNOWN
        ):
            return {
                "TEAM_A": a,
                "TEAM_B": b,
            }

    return {
        "TEAM_A": UNKNOWN,
        "TEAM_B": UNKNOWN,
    }


def source_team(row):
    team = row.get(
        "source_team"
    )

    if team in {
        "TEAM_A",
        "TEAM_B",
    }:
        return team

    state = row.get(
        "team_state"
    )

    if state in {
        "TEAM_A",
        "TEAM_B",
    }:
        return state

    return None


def phase(row):
    return str(
        row.get(
            "phase_v13",
            row.get(
                "phase",
                RAW_LOOSE,
            ),
        )
    )


def is_candidate(row):
    return phase(row) in CANDIDATE_PHASES


def build_runs(rows):
    runs = []
    i = 0

    while i < len(rows):
        if not is_candidate(
            rows[i]
        ):
            i += 1
            continue

        start = i
        team = source_team(
            rows[i]
        )

        i += 1

        while i < len(rows):
            if not is_candidate(
                rows[i]
            ):
                break

            current_team = source_team(
                rows[i]
            )

            if (
                team is not None
                and current_team is not None
                and current_team != team
            ):
                break

            if (
                team is None
                and current_team is not None
            ):
                team = current_team

            i += 1

        runs.append(
            (start, i - 1)
        )

    return runs


def ball_image_xy(row):
    ball = (
        row.get("ball")
        or {}
    )

    xy = ball.get(
        "image_xy"
    )

    if (
        xy is None
        or len(xy) < 2
    ):
        return None

    return (
        float(xy[0]),
        float(xy[1]),
    )


def main():
    args = parse_args()

    source = resolve_project_path(
        args.source
    )
    possession_path = resolve_project_path(
        args.possession_jsonl
    )
    direction_path = resolve_project_path(
        args.direction_jsonl
    )
    team_path = resolve_project_path(
        args.team_jsonl
    )
    calibration_path = resolve_project_path(
        args.calibration_json
    )
    output = resolve_project_path(
        args.output
    )
    jsonl_out = resolve_project_path(
        args.jsonl
    )

    for path in (
        source,
        possession_path,
        direction_path,
        team_path,
        calibration_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    rows = read_jsonl(
        possession_path
    )
    direction_rows = read_jsonl(
        direction_path
    )
    team_rows = read_jsonl_map(
        team_path
    )
    calibrations = read_calibration(
        calibration_path
    )

    directions = resolved_directions(
        direction_rows
    )

    classifier = ImageGoalShotClassifier(
        ImageGoalShotConfig(
            min_samples=max(
                2,
                args.min_samples,
            ),
            min_goal_mouth_px=max(
                2.0,
                args.min_goal_mouth_px,
            ),
        )
    )

    enhanced = [
        dict(row)
        for row in rows
    ]

    run_results = []

    for start, end in build_runs(
        rows
    ):
        run_rows = rows[
            start:end + 1
        ]

        team = source_team(
            run_rows[0]
        )

        attack_direction = (
            directions.get(
                team,
                UNKNOWN,
            )
            if team is not None
            else UNKNOWN
        )

        samples = []

        for row in run_rows:
            frame_index = int(
                row["frame_index"]
            )

            ball_xy = ball_image_xy(
                row
            )

            if ball_xy is None:
                continue

            team_row = team_rows.get(
                frame_index
            )

            if team_row is None:
                continue

            calibration_frame = team_row.get(
                "pnl_calibration_frame"
            )

            if calibration_frame is None:
                continue

            H = calibrations.get(
                int(
                    calibration_frame
                )
            )

            if H is None:
                continue

            geometry = goal_image_geometry(
                H,
                attack_direction,
            )

            if geometry is None:
                continue

            normalized = normalized_ball_goal_distance(
                ball_xy,
                geometry,
                classifier.config.min_goal_mouth_px,
            )

            if normalized is None:
                continue

            samples.append(
                GoalApproachSample(
                    frame_index=frame_index,
                    normalized_goal_distance=normalized,
                    ball_xy=ball_xy,
                    goal_center_xy=geometry.center_xy,
                    goal_mouth_width_px=geometry.mouth_width_px,
                )
            )

        fallback = phase(
            run_rows[0]
        )

        result = classifier.classify(
            start_frame=int(
                run_rows[0]["frame_index"]
            ),
            end_frame=int(
                run_rows[-1]["frame_index"]
            ),
            team=team,
            samples=samples,
            fallback_phase=fallback,
        )

        run_results.append(
            result
        )

        for index in range(
            start,
            end + 1,
        ):
            enhanced[index][
                "phase_v14"
            ] = result.classification

            if (
                result.team
                in {
                    "TEAM_A",
                    "TEAM_B",
                }
                and result.classification
                in {
                    ATTACKING_FLIGHT,
                    SHOT_FLIGHT,
                    GOAL_ATTEMPT,
                }
            ):
                enhanced[index][
                    "team_state_v14"
                ] = result.team
            else:
                enhanced[index][
                    "team_state_v14"
                ] = enhanced[index].get(
                    "team_state_v13",
                    enhanced[index].get(
                        "team_state"
                    ),
                )

            enhanced[index][
                "image_goal_context"
            ] = result.__dict__

    for row in enhanced:
        row.setdefault(
            "phase_v14",
            row.get(
                "phase_v13",
                row.get("phase"),
            ),
        )

        row.setdefault(
            "team_state_v14",
            row.get(
                "team_state_v13",
                row.get(
                    "team_state"
                ),
            ),
        )

        row.setdefault(
            "image_goal_context",
            None,
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    jsonl_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(source)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    ) or 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    by_frame = {
        int(row["frame_index"]): row
        for row in enhanced
    }

    last_frame = max(
        by_frame
    )

    stats = Counter()
    frame_index = 0

    colors = {
        "TEAM_A": (
            255,
            90,
            40,
        ),
        "TEAM_B": (
            40,
            70,
            255,
        ),
        "LOOSE": (
            0,
            215,
            255,
        ),
        "UNKNOWN": (
            150,
            150,
            150,
        ),
    }

    with jsonl_out.open(
        "w",
        encoding="utf-8",
    ) as out:
        try:
            while frame_index <= last_frame:
                ok, frame = cap.read()

                if not ok:
                    break

                row = by_frame.get(
                    frame_index
                )

                if row is None:
                    frame_index += 1
                    continue

                team = row.get(
                    "team_state_v14",
                    row.get(
                        "team_state",
                        "UNKNOWN",
                    ),
                )

                current_phase = row.get(
                    "phase_v14",
                    row.get(
                        "phase",
                        RAW_LOOSE,
                    ),
                )

                stats[
                    f"team:{team}"
                ] += 1
                stats[
                    f"phase:{current_phase}"
                ] += 1

                color = colors.get(
                    team,
                    (
                        200,
                        200,
                        200,
                    ),
                )

                cv2.rectangle(
                    frame,
                    (
                        0,
                        0,
                    ),
                    (
                        width,
                        125,
                    ),
                    (
                        18,
                        18,
                        18,
                    ),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        f"Shot Context v1.4 | "
                        f"{team} | "
                        f"{current_phase}"
                    ),
                    (
                        18,
                        30,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                context = (
                    row.get(
                        "image_goal_context"
                    )
                    or {}
                )

                if context:
                    start_d = context.get(
                        "start_goal_distance_units"
                    )
                    closest_d = context.get(
                        "closest_goal_distance_units"
                    )
                    closing = float(
                        context.get(
                            "closing_progress_units",
                            0.0,
                        )
                    )
                    approach = float(
                        context.get(
                            "approach_fraction",
                            0.0,
                        )
                    )

                    start_text = (
                        f"{float(start_d):.2f}"
                        if start_d is not None
                        else "N/A"
                    )
                    close_text = (
                        f"{float(closest_d):.2f}"
                        if closest_d is not None
                        else "N/A"
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Goal units: "
                            f"start={start_text} "
                            f"closest={close_text} "
                            f"closing={closing:.2f}"
                        ),
                        (
                            18,
                            60,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.47,
                        (
                            235,
                            235,
                            235,
                        ),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Approach fraction="
                            f"{approach:.2f} | "
                            f"{context.get('reason', '')}"
                        ),
                        (
                            18,
                            88,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (
                            180,
                            180,
                            180,
                        ),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    (
                        f"Directions: "
                        f"A={directions['TEAM_A']} "
                        f"B={directions['TEAM_B']} | "
                        f"airborne ball not ground-projected"
                    ),
                    (
                        18,
                        116,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (
                        150,
                        150,
                        150,
                    ),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(
                    frame
                )

                out.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                frame_index += 1

        finally:
            cap.release()
            writer.release()

    print("=" * 92)
    print(
        "DONE - Shot Context v1.4 "
        "IMAGE GOAL GEOMETRY"
    )
    print(
        f"Directions         : "
        f"A={directions['TEAM_A']} "
        f"B={directions['TEAM_B']}"
    )
    print(
        f"TEAM_A team-state  : "
        f"{stats['team:TEAM_A']}"
    )
    print(
        f"TEAM_B team-state  : "
        f"{stats['team:TEAM_B']}"
    )
    print(
        f"LOOSE              : "
        f"{stats['team:LOOSE']}"
    )
    print(
        f"UNKNOWN            : "
        f"{stats['team:UNKNOWN']}"
    )
    print("Phases:")

    for name in (
        "CONTROL",
        "CONTROL_GAP",
        PASS_FLIGHT,
        TEAM_FLIGHT,
        "CONTESTED_FLIGHT",
        ATTACKING_FLIGHT,
        SHOT_FLIGHT,
        GOAL_ATTEMPT,
        RAW_LOOSE,
        "RAW_UNKNOWN",
    ):
        print(
            f"  {name:<20} "
            f"{stats[f'phase:{name}']}"
        )

    print("Run classifications:")
    rc = Counter(
        result.classification
        for result in run_results
    )

    for name, count in rc.most_common():
        print(
            f"  {name:<20} "
            f"{count}"
        )

    print(
        f"Video output       : "
        f"{output}"
    )
    print(
        f"JSONL output       : "
        f"{jsonl_out}"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
