from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
REFEREE = "REFEREE"
GOALKEEPER = "GOALKEEPER"

TEAM_COLORS = {
    TEAM_A: (235, 95, 50),
    TEAM_B: (55, 75, 235),
}
REF_COLOR = (0, 220, 255)
UNKNOWN_COLOR = (175, 175, 175)
WHITE = (238, 238, 238)


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Radar v4 (PnL primary + TVCalib fallback)"
    )
    p.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument(
        "--classification",
        default=r"output\team_classification_v24_pnl_exact.jsonl",
    )
    p.add_argument(
        "--fusion",
        default=r"output\calibration_fusion_v1.json",
    )
    p.add_argument(
        "--output",
        default=r"output\radar_v4.mp4",
    )
    p.add_argument(
        "--debug-output",
        default=r"output\radar_v4_debug.mp4",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument(
        "--position-alpha",
        type=float,
        default=0.65,
        help="EMA weight for current pitch position. 1 disables smoothing.",
    )
    p.add_argument(
        "--max-smoothing-gap",
        type=int,
        default=4,
    )
    return p.parse_args()


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rows[int(item["frame_index"])] = item
    return rows


def transform_point(H, x, y):
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(float(p[2])) < 1e-12:
        return None
    p = p / p[2]
    if not np.isfinite(p).all():
        return None
    return float(p[0]), float(p[1])


def inside_pitch(xy):
    return (
        xy is not None
        and 0.0 <= xy[0] <= PITCH_LENGTH_M
        and 0.0 <= xy[1] <= PITCH_WIDTH_M
    )


def draw_pitch(width: int):
    height = int(round(width * PITCH_WIDTH_M / PITCH_LENGTH_M))
    margin = max(24, int(round(width * 0.025)))

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (39, 105, 46)

    usable_w = width - 2 * margin
    stripe = usable_w / 12
    for i in range(12):
        if i % 2 == 0:
            x0 = int(round(margin + i * stripe))
            x1 = int(round(margin + (i + 1) * stripe))
            cv2.rectangle(
                img,
                (x0, margin),
                (x1, height - margin),
                (43, 114, 50),
                -1,
            )

    def p(x, y):
        px = margin + int(round((x / PITCH_LENGTH_M) * (width - 2 * margin)))
        py = margin + int(round((y / PITCH_WIDTH_M) * (height - 2 * margin)))
        return px, py

    t = max(1, width // 600)
    cv2.rectangle(img, p(0, 0), p(105, 68), WHITE, t, cv2.LINE_AA)
    cv2.line(img, p(52.5, 0), p(52.5, 68), WHITE, t, cv2.LINE_AA)

    cx, cy = p(52.5, 34)
    radius = int(round(9.15 / 105.0 * (width - 2 * margin)))
    cv2.circle(img, (cx, cy), radius, WHITE, t, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, t + 1), WHITE, -1, cv2.LINE_AA)

    for left in (True, False):
        if left:
            x_goal, x_pen, x_goalbox, x_spot = 0.0, 16.5, 5.5, 11.0
        else:
            x_goal, x_pen, x_goalbox, x_spot = 105.0, 88.5, 99.5, 94.0

        cv2.rectangle(
            img,
            p(min(x_goal, x_pen), 13.84),
            p(max(x_goal, x_pen), 54.16),
            WHITE,
            t,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            img,
            p(min(x_goal, x_goalbox), 24.84),
            p(max(x_goal, x_goalbox), 43.16),
            WHITE,
            t,
            cv2.LINE_AA,
        )
        cv2.circle(img, p(x_spot, 34.0), max(2, t + 1), WHITE, -1)

    return img, margin, p


def classification_payload(track: dict) -> dict:
    for key in ("team_v24", "team_v23", "team_v22", "team_v2"):
        if isinstance(track.get(key), dict):
            return track[key]
    return {}


def main():
    args = parse_args()

    source = Path(args.source)
    classification_path = resolve(args.classification)
    fusion_path = resolve(args.fusion)
    output_path = resolve(args.output)
    debug_output_path = resolve(args.debug_output)

    for p in (source, classification_path, fusion_path):
        if not p.exists():
            raise FileNotFoundError(p)

    classifications = read_jsonl(classification_path)
    fusion_payload = json.loads(fusion_path.read_text(encoding="utf-8"))
    fusion = {
        int(x["frame_index"]): x
        for x in fusion_payload["frames"]
    }

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pitch_template, margin, pitch_to_px = draw_pitch(args.width)
    radar_h, radar_w = pitch_template.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (radar_w, radar_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")

    debug_h = max(src_h, radar_h)
    left_w = int(round(src_w * debug_h / src_h))
    right_w = int(round(radar_w * debug_h / radar_h))
    debug_size = (left_w + right_w, debug_h)

    debug_writer = cv2.VideoWriter(
        str(debug_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        debug_size,
    )
    if not debug_writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create {debug_output_path}")

    alpha = float(np.clip(args.position_alpha, 0.0, 1.0))
    smooth_state = {}
    stats = Counter()

    frame_index = 0

    try:
        while True:
            ok, source_frame = cap.read()
            if not ok:
                break

            if frame_index not in classifications:
                frame_index += 1
                continue

            radar = pitch_template.copy()
            row = classifications[frame_index]
            geometry = fusion.get(frame_index)

            engine = None
            H = None

            if geometry and geometry.get("status") == "ok":
                engine = geometry.get("engine")
                H = np.asarray(
                    geometry["homography_image_to_pitch"],
                    dtype=np.float64,
                )
                stats[f"engine_{engine}"] += 1
            else:
                stats["missing_geometry"] += 1

            points = []

            if H is not None:
                for tr in row.get("tracks", []):
                    foot = tr.get("foot_point")
                    if not foot or len(foot) < 2:
                        continue

                    xy = transform_point(
                        H,
                        float(foot[0]),
                        float(foot[1]),
                    )
                    if not inside_pitch(xy):
                        stats["outside"] += 1
                        continue

                    payload = classification_payload(tr)
                    role = str(payload.get("role", "PLAYER"))
                    team = str(payload.get("team", "UNKNOWN"))

                    if role == "OUTSIDE_PITCH":
                        continue

                    tid = int(
                        payload.get(
                            "track_id",
                            tr.get("track_id", -1),
                        )
                    )
                    if tid < 0:
                        continue

                    previous = smooth_state.get(tid)
                    if (
                        previous is not None
                        and frame_index - previous[2] <= args.max_smoothing_gap
                    ):
                        x = alpha * xy[0] + (1.0 - alpha) * previous[0]
                        y = alpha * xy[1] + (1.0 - alpha) * previous[1]
                        xy = (x, y)

                    smooth_state[tid] = (xy[0], xy[1], frame_index)
                    points.append((tid, xy, team, role))

            for tid, xy, team, role in points:
                px, py = pitch_to_px(xy[0], xy[1])

                if role == REFEREE:
                    color = REF_COLOR
                    cv2.circle(radar, (px, py), 10, color, -1, cv2.LINE_AA)
                    cv2.circle(radar, (px, py), 12, WHITE, 2, cv2.LINE_AA)
                    label = f"R{tid}"
                    stats["referee"] += 1

                elif role == GOALKEEPER:
                    color = TEAM_COLORS.get(team, UNKNOWN_COLOR)
                    cv2.rectangle(
                        radar,
                        (px - 9, py - 9),
                        (px + 9, py + 9),
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.rectangle(
                        radar,
                        (px - 11, py - 11),
                        (px + 11, py + 11),
                        WHITE,
                        2,
                        cv2.LINE_AA,
                    )
                    label = f"G{tid}"
                    stats["goalkeeper"] += 1

                else:
                    color = TEAM_COLORS.get(team, UNKNOWN_COLOR)
                    cv2.circle(radar, (px, py), 9, color, -1, cv2.LINE_AA)
                    cv2.circle(radar, (px, py), 11, WHITE, 1, cv2.LINE_AA)
                    label = str(tid)

                cv2.putText(
                    radar,
                    label,
                    (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )

            cv2.rectangle(
                radar,
                (0, 0),
                (radar_w, 54),
                (18, 18, 18),
                -1,
            )
            cv2.putText(
                radar,
                f"RADAR v4 | Frame {frame_index} | Geometry: {engine or 'MISSING'}",
                (18, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

            if H is None:
                cv2.putText(
                    radar,
                    "NO CALIBRATION",
                    (radar_w // 2 - 120, radar_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (50, 50, 240),
                    3,
                    cv2.LINE_AA,
                )

            writer.write(radar)

            left = cv2.resize(
                source_frame,
                (left_w, debug_h),
                interpolation=cv2.INTER_AREA,
            )
            right = cv2.resize(
                radar,
                (right_w, debug_h),
                interpolation=cv2.INTER_AREA,
            )
            debug = np.hstack([left, right])
            debug_writer.write(debug)

            frame_index += 1
            stats["frames"] += 1

    finally:
        cap.release()
        writer.release()
        debug_writer.release()

    print("=" * 88)
    print("DONE - RADAR v4")
    print(f"Frames        : {stats['frames']}")
    print(f"PnL frames    : {stats['engine_PnLCalib']}")
    print(f"TV frames     : {stats['engine_TVCalib']}")
    print(f"Missing geom  : {stats['missing_geometry']}")
    print(f"Outside points: {stats['outside']}")
    print(f"Radar         : {output_path}")
    print(f"Debug         : {debug_output_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
