from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.possession import (
    LOOSE,
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    PlayerPossessionCandidate,
    PossessionConfig,
    PossessionEstimator,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Ball-to-pitch + Possession v1.1"
    )
    p.add_argument("--source", default=r"input\input.mp4")
    p.add_argument("--ball-jsonl", default=r"output\ball_tracking.jsonl")
    p.add_argument(
        "--team-jsonl",
        default=r"output\team_classification_v24_pnl_exact.jsonl",
    )
    p.add_argument(
        "--calibration-json",
        default=r"output\team_classification_v24_pnl_exact_calibration.json",
    )
    p.add_argument("--output", default=r"output\possession_v11.mp4")
    p.add_argument("--jsonl", default=r"output\possession_v11.jsonl")
    p.add_argument("--max-frames", type=int, default=-1)

    p.add_argument("--acquire-distance", type=float, default=3.2)
    p.add_argument("--release-distance", type=float, default=5.0)
    p.add_argument("--acquire-image-ratio", type=float, default=0.75)
    p.add_argument("--release-image-ratio", type=float, default=1.20)
    p.add_argument("--switch-margin", type=float, default=0.75)
    p.add_argument("--confirm-frames", type=int, default=2)
    p.add_argument("--hold-missing-ball", type=int, default=4)
    return p.parse_args()


def read_jsonl(path: Path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                out[int(item["frame_index"])] = item
    return out


def read_calibration(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in rows:
        if item.get("status") != "ok":
            continue
        if "homography_image_to_pitch" not in item:
            continue
        result[int(item["frame_index"])] = item
    return result


def transform_point(H: np.ndarray, xy):
    p = H @ np.array(
        [float(xy[0]), float(xy[1]), 1.0],
        dtype=np.float64,
    )
    if abs(float(p[2])) < 1e-12:
        return None
    p = p / p[2]
    if not np.isfinite(p).all():
        return None
    return float(p[0]), float(p[1])


def inside_pitch(xy):
    if xy is None:
        return False
    x, y = xy
    return 0.0 <= x <= 105.0 and 0.0 <= y <= 68.0


def collect_players(team_row):
    players = []

    for tr in team_row.get("tracks", []):
        pitch = tr.get("pitch_xy")
        if pitch is None or len(pitch) < 2:
            continue

        assignment = tr.get("team_v24") or {}
        team = str(assignment.get("team", "UNKNOWN"))
        role = str(assignment.get("role", "PLAYER")).upper()
        source_class = str(
            tr.get("class_name", "")
        ).strip().lower()

        if team not in {TEAM_A, TEAM_B}:
            continue

        # Hard referee veto.
        if (
            role in {"REFEREE", "OUTSIDE_PITCH"}
            or source_class == "referee"
        ):
            continue

        foot = tr.get("foot_point")
        image_foot_xy = None
        if foot is not None and len(foot) >= 2:
            image_foot_xy = (
                float(foot[0]),
                float(foot[1]),
            )

        bbox = tr.get("bbox_xyxy")
        bbox_height = None
        if bbox is not None and len(bbox) == 4:
            bbox_height = max(
                0.0,
                float(bbox[3]) - float(bbox[1]),
            )

        players.append(
            PlayerPossessionCandidate(
                track_id=int(tr.get("track_id", -1)),
                team=team,
                role=role,
                pitch_xy=(
                    float(pitch[0]),
                    float(pitch[1]),
                ),
                image_foot_xy=image_foot_xy,
                bbox_height_px=bbox_height,
            )
        )

    return players


def draw_pitch_panel(frame, ball_xy, players, result):
    panel_w = min(360, max(260, frame.shape[1] // 4))
    panel_h = int(round(panel_w * 68.0 / 105.0))
    margin = 12

    panel = np.zeros(
        (panel_h, panel_w, 3),
        dtype=np.uint8,
    )
    panel[:] = (38, 105, 45)

    def p(x, y):
        return (
            margin + int(
                round(
                    (x / 105.0)
                    * (panel_w - 2 * margin)
                )
            ),
            margin + int(
                round(
                    (y / 68.0)
                    * (panel_h - 2 * margin)
                )
            ),
        )

    white = (230, 230, 230)

    cv2.rectangle(
        panel,
        p(0, 0),
        p(105, 68),
        white,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        panel,
        p(52.5, 0),
        p(52.5, 68),
        white,
        1,
        cv2.LINE_AA,
    )

    for pl in players:
        color = (
            (255, 90, 40)
            if pl.team == TEAM_A
            else (40, 70, 255)
        )
        radius = (
            6
            if pl.track_id == result.possessor_track_id
            else 4
        )
        cv2.circle(
            panel,
            p(*pl.pitch_xy),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )

    if ball_xy is not None and inside_pitch(ball_xy):
        cv2.circle(
            panel,
            p(*ball_xy),
            5,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            panel,
            p(*ball_xy),
            8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    x0 = frame.shape[1] - panel_w - 8
    y0 = frame.shape[0] - panel_h - 8

    if x0 >= 0 and y0 >= 0:
        roi = frame[
            y0:y0 + panel_h,
            x0:x0 + panel_w,
        ]
        cv2.addWeighted(
            panel,
            0.90,
            roi,
            0.10,
            0,
            roi,
        )


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    ball_path = resolve_project_path(args.ball_jsonl)
    team_path = resolve_project_path(args.team_jsonl)
    calib_path = resolve_project_path(args.calibration_json)
    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)

    for path in (
        source,
        ball_path,
        team_path,
        calib_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    jsonl_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ball_rows = read_jsonl(ball_path)
    team_rows = read_jsonl(team_path)
    calibrations = read_calibration(calib_path)

    common_frames = sorted(
        set(ball_rows) & set(team_rows)
    )

    if args.max_frames >= 0:
        common_frames = common_frames[:args.max_frames]

    if not common_frames:
        raise RuntimeError(
            "No common frames between ball and team JSONL."
        )

    estimator = PossessionEstimator(
        PossessionConfig(
            acquire_distance_m=args.acquire_distance,
            release_distance_m=args.release_distance,
            acquire_image_ratio=args.acquire_image_ratio,
            release_image_ratio=args.release_image_ratio,
            switch_margin_m=args.switch_margin,
            confirm_frames=max(
                1,
                args.confirm_frames,
            ),
            hold_missing_ball_frames=max(
                0,
                args.hold_missing_ball,
            ),
        )
    )

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    ) or 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output: {output}"
        )

    stats = Counter()
    frame_set = set(common_frames)
    last_needed = max(common_frames)
    frame_index = 0

    print("=" * 84)
    print(
        "FootballAnalysisAI - Ball to Pitch + "
        "Possession v1.1 HYBRID"
    )
    print(f"Frames              : {len(common_frames)}")
    print(
        f"Pitch acquire/release: "
        f"{args.acquire_distance:.2f}/"
        f"{args.release_distance:.2f} m"
    )
    print(
        f"Image acquire/release: "
        f"{args.acquire_image_ratio:.2f}/"
        f"{args.release_image_ratio:.2f} bbox-heights"
    )
    print("=" * 84)

    with jsonl_out.open(
        "w",
        encoding="utf-8",
    ) as out:
        try:
            while frame_index <= last_needed:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_index not in frame_set:
                    frame_index += 1
                    continue

                ball_row = ball_rows[frame_index]
                team_row = team_rows[frame_index]

                calibration_frame = team_row.get(
                    "pnl_calibration_frame"
                )

                calib = (
                    calibrations.get(
                        int(calibration_frame)
                    )
                    if calibration_frame is not None
                    else None
                )

                ball_image_xy = ball_row.get(
                    "center_xy"
                )

                ball_pitch_xy: Optional[
                    tuple[float, float]
                ] = None

                if (
                    calib is not None
                    and ball_image_xy is not None
                ):
                    H = np.asarray(
                        calib[
                            "homography_image_to_pitch"
                        ],
                        dtype=np.float64,
                    )

                    projected = transform_point(
                        H,
                        ball_image_xy,
                    )

                    if inside_pitch(projected):
                        ball_pitch_xy = projected

                players = collect_players(
                    team_row
                )

                image_xy_tuple = (
                    (
                        float(ball_image_xy[0]),
                        float(ball_image_xy[1]),
                    )
                    if ball_image_xy is not None
                    else None
                )

                result = estimator.update(
                    ball_pitch_xy,
                    players,
                    ball_detected=bool(
                        ball_row.get(
                            "detected",
                            False,
                        )
                    ),
                    ball_predicted=bool(
                        ball_row.get(
                            "predicted",
                            False,
                        )
                    ),
                    ball_image_xy=image_xy_tuple,
                )

                stats[result.state] += 1
                stats[
                    f"reason:{result.reason}"
                ] += 1
                stats[
                    f"source:{result.control_source}"
                ] += 1

                if (
                    result.state in {TEAM_A, TEAM_B}
                    and ball_pitch_xy is None
                    and result.control_source in {
                        "IMAGE",
                        "BOTH",
                    }
                ):
                    stats[
                        "image_fallback_possession"
                    ] += 1

                if ball_pitch_xy is not None:
                    stats["ball_pitch_ok"] += 1
                else:
                    stats[
                        "ball_pitch_missing"
                    ] += 1

                annotated = frame.copy()

                if ball_image_xy is not None:
                    bx, by = map(
                        int,
                        map(
                            round,
                            ball_image_xy,
                        ),
                    )
                    cv2.circle(
                        annotated,
                        (bx, by),
                        9,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                state_color = {
                    TEAM_A: (255, 90, 40),
                    TEAM_B: (40, 70, 255),
                    LOOSE: (0, 215, 255),
                    UNKNOWN: (160, 160, 160),
                }.get(
                    result.state,
                    (200, 200, 200),
                )

                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (width, 118),
                    (18, 18, 18),
                    -1,
                )

                cv2.putText(
                    annotated,
                    (
                        f"Possession v1.1: "
                        f"{result.state}"
                    ),
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    state_color,
                    2,
                    cv2.LINE_AA,
                )

                owner = (
                    f"ID {result.possessor_track_id}"
                    if result.possessor_track_id
                    is not None
                    else "NONE"
                )

                if result.distance_m is not None:
                    owner += (
                        f" | pitch="
                        f"{result.distance_m:.2f}m"
                    )

                if (
                    result.image_distance_ratio
                    is not None
                ):
                    owner += (
                        f" | img="
                        f"{result.image_distance_ratio:.2f}h"
                    )

                cv2.putText(
                    annotated,
                    (
                        f"Owner: {owner} | "
                        f"conf={result.confidence:.2f}"
                    ),
                    (18, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    annotated,
                    (
                        f"Signal={result.control_source} "
                        f"| {result.reason}"
                    ),
                    (18, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (180, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    annotated,
                    (
                        "Hybrid control: "
                        "pitch meters + "
                        "perspective-normalized foot distance"
                    ),
                    (18, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )

                draw_pitch_panel(
                    annotated,
                    ball_pitch_xy,
                    players,
                    result,
                )

                writer.write(annotated)

                payload = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(
                        frame_index / fps,
                        5,
                    ),
                    "calibration_frame": (
                        calibration_frame
                    ),
                    "ball": {
                        "image_xy": ball_image_xy,
                        "pitch_xy": (
                            [
                                round(
                                    ball_pitch_xy[0],
                                    4,
                                ),
                                round(
                                    ball_pitch_xy[1],
                                    4,
                                ),
                            ]
                            if ball_pitch_xy
                            is not None
                            else None
                        ),
                        "detected": bool(
                            ball_row.get(
                                "detected",
                                False,
                            )
                        ),
                        "predicted": bool(
                            ball_row.get(
                                "predicted",
                                False,
                            )
                        ),
                    },
                    "possession": {
                        "state": result.state,
                        "possessor_track_id": (
                            result.possessor_track_id
                        ),
                        "possessor_team": (
                            result.possessor_team
                        ),
                        "distance_m": (
                            round(
                                result.distance_m,
                                4,
                            )
                            if result.distance_m
                            is not None
                            else None
                        ),
                        "image_distance_px": (
                            round(
                                result.image_distance_px,
                                3,
                            )
                            if result.image_distance_px
                            is not None
                            else None
                        ),
                        "image_distance_ratio": (
                            round(
                                result.image_distance_ratio,
                                4,
                            )
                            if result.image_distance_ratio
                            is not None
                            else None
                        ),
                        "control_source": (
                            result.control_source
                        ),
                        "confidence": round(
                            result.confidence,
                            5,
                        ),
                        "nearest_track_id": (
                            result.nearest_track_id
                        ),
                        "nearest_team": (
                            result.nearest_team
                        ),
                        "nearest_distance_m": (
                            round(
                                result.nearest_distance_m,
                                4,
                            )
                            if result.nearest_distance_m
                            is not None
                            else None
                        ),
                        "reason": result.reason,
                    },
                }

                out.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                frame_index += 1

        finally:
            cap.release()
            writer.release()

    total = sum(
        stats[s]
        for s in (
            TEAM_A,
            TEAM_B,
            LOOSE,
            UNKNOWN,
        )
    )

    print("=" * 84)
    print("DONE - Possession v1.1 HYBRID")
    print(f"Frames processed           : {total}")
    print(f"TEAM_A possession          : {stats[TEAM_A]}")
    print(f"TEAM_B possession          : {stats[TEAM_B]}")
    print(f"LOOSE                      : {stats[LOOSE]}")
    print(f"UNKNOWN                    : {stats[UNKNOWN]}")
    print(f"Ball pitch OK              : {stats['ball_pitch_ok']}")
    print(f"Ball pitch missing         : {stats['ball_pitch_missing']}")
    print(
        f"Image-fallback possession  : "
        f"{stats['image_fallback_possession']}"
    )

    print("Control sources:")
    for source in (
        "BOTH",
        "PITCH",
        "IMAGE",
        "NONE",
    ):
        print(
            f"  {source:<8} "
            f"{stats[f'source:{source}']}"
        )

    print("Top reasons:")
    reasons = sorted(
        (
            (
                k.split(":", 1)[1],
                v,
            )
            for k, v in stats.items()
            if k.startswith("reason:")
        ),
        key=lambda x: -x[1],
    )

    for reason, count in reasons[:8]:
        print(
            f"  {reason:<32} "
            f"{count}"
        )

    print(f"Video output               : {output}")
    print(f"JSONL output               : {jsonl_out}")
    print("=" * 84)


if __name__ == "__main__":
    main()
