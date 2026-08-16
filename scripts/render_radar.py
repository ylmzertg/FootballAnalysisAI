from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


# Roboflow Sports SoccerPitchConfiguration, in centimeters.
PITCH_WIDTH_CM = 7000
PITCH_LENGTH_CM = 12000
PENALTY_BOX_WIDTH_CM = 4100
PENALTY_BOX_LENGTH_CM = 2015
GOAL_BOX_WIDTH_CM = 1832
GOAL_BOX_LENGTH_CM = 550
CENTRE_CIRCLE_RADIUS_CM = 915
PENALTY_SPOT_DISTANCE_CM = 1100


def pitch_vertices() -> np.ndarray:
    w = PITCH_WIDTH_CM
    l = PITCH_LENGTH_CM
    pbw = PENALTY_BOX_WIDTH_CM
    pbl = PENALTY_BOX_LENGTH_CM
    gbw = GOAL_BOX_WIDTH_CM
    gbl = GOAL_BOX_LENGTH_CM
    ccr = CENTRE_CIRCLE_RADIUS_CM
    psd = PENALTY_SPOT_DISTANCE_CM

    return np.array([
        (0, 0),
        (0, (w - pbw) / 2),
        (0, (w - gbw) / 2),
        (0, (w + gbw) / 2),
        (0, (w + pbw) / 2),
        (0, w),
        (gbl, (w - gbw) / 2),
        (gbl, (w + gbw) / 2),
        (psd, w / 2),
        (pbl, (w - pbw) / 2),
        (pbl, (w - gbw) / 2),
        (pbl, (w + gbw) / 2),
        (pbl, (w + pbw) / 2),
        (l / 2, 0),
        (l / 2, w / 2 - ccr),
        (l / 2, w / 2 + ccr),
        (l / 2, w),
        (l - pbl, (w - pbw) / 2),
        (l - pbl, (w - gbw) / 2),
        (l - pbl, (w + gbw) / 2),
        (l - pbl, (w + pbw) / 2),
        (l - psd, w / 2),
        (l - gbl, (w - gbw) / 2),
        (l - gbl, (w + gbw) / 2),
        (l, 0),
        (l, (w - pbw) / 2),
        (l, (w - gbw) / 2),
        (l, (w + gbw) / 2),
        (l, (w + pbw) / 2),
        (l, w),
        (l / 2 - ccr, w / 2),
        (l / 2 + ccr, w / 2),
    ], dtype=np.float32)


PITCH_VERTICES = pitch_vertices()

PITCH_EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
    (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
    (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
    (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
]

CLASS_COLORS_BGR = {
    "player": (0, 255, 255),
    "goalkeeper": (255, 0, 255),
    "referee": (0, 165, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FootballAnalysisAI - create 2D radar from tracking + pitch keypoints"
    )
    parser.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    parser.add_argument(
        "--tracking",
        default=r"output\player_tracking.jsonl",
    )
    parser.add_argument(
        "--pitch",
        default=r"output\pitch_keypoints.jsonl",
    )
    parser.add_argument(
        "--output",
        default=r"output\radar.mp4",
    )
    parser.add_argument(
        "--match-state",
        default=r"output\match_state.jsonl",
    )
    parser.add_argument(
        "--min-keypoints",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=35.0,
        help="Reject new homographies with median image reprojection error above this value (px).",
    )
    parser.add_argument(
        "--radar-width",
        type=int,
        default=720,
    )
    parser.add_argument(
        "--radar-opacity",
        type=float,
        default=0.82,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            result[int(item["frame_index"])] = item
    return result


def build_homography(keypoints: list[dict], min_keypoints: int):
    src = []
    dst = []
    ids = []

    for kp in keypoints:
        kp_id = int(kp["keypoint_id"])
        if 0 <= kp_id < len(PITCH_VERTICES):
            src.append([float(kp["x"]), float(kp["y"])])
            dst.append(PITCH_VERTICES[kp_id].tolist())
            ids.append(kp_id)

    if len(src) < min_keypoints:
        return None, None, None, ids

    src_arr = np.asarray(src, dtype=np.float32)
    dst_arr = np.asarray(dst, dtype=np.float32)

    H_img_to_pitch, inlier_mask = cv2.findHomography(
        src_arr,
        dst_arr,
        method=cv2.RANSAC,
        ransacReprojThreshold=500.0,
    )

    if H_img_to_pitch is None:
        return None, None, None, ids

    H_pitch_to_img = np.linalg.inv(H_img_to_pitch)

    # Reproject the known pitch coordinates back to the image,
    # so the error is expressed in pixels and is easy to interpret.
    dst_pitch = dst_arr.reshape(-1, 1, 2)
    reproj_img = cv2.perspectiveTransform(
        dst_pitch,
        H_pitch_to_img.astype(np.float64),
    ).reshape(-1, 2)

    errors_px = np.linalg.norm(reproj_img - src_arr, axis=1)
    median_error_px = float(np.median(errors_px))
    max_error_px = float(np.max(errors_px))

    inliers = (
        int(inlier_mask.sum())
        if inlier_mask is not None
        else len(src_arr)
    )

    metrics = {
        "num_keypoints": len(src_arr),
        "inliers": inliers,
        "median_reprojection_error_px": round(median_error_px, 3),
        "max_reprojection_error_px": round(max_error_px, 3),
    }

    return H_img_to_pitch, H_pitch_to_img, metrics, ids


def transform_point(H: np.ndarray, x: float, y: float):
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(
        point,
        H.astype(np.float64),
    )[0, 0]

    px = float(transformed[0])
    py = float(transformed[1])

    if not np.isfinite(px) or not np.isfinite(py):
        return None

    return px, py


def pitch_to_radar_xy(
    x_cm: float,
    y_cm: float,
    radar_w: int,
    radar_h: int,
    margin: int,
):
    usable_w = radar_w - 2 * margin
    usable_h = radar_h - 2 * margin

    x = margin + (x_cm / PITCH_LENGTH_CM) * usable_w
    y = margin + (y_cm / PITCH_WIDTH_CM) * usable_h

    return int(round(x)), int(round(y))


def draw_pitch(radar_w: int) -> np.ndarray:
    aspect = PITCH_WIDTH_CM / PITCH_LENGTH_CM
    radar_h = int(round(radar_w * aspect))
    margin = max(18, int(radar_w * 0.035))

    canvas = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)
    canvas[:] = (38, 105, 45)

    # Alternating mowing stripes.
    usable_w = radar_w - 2 * margin
    stripe_w = max(1, usable_w // 12)

    for i in range(12):
        if i % 2 == 0:
            x0 = margin + i * stripe_w
            x1 = margin + (i + 1) * stripe_w
            cv2.rectangle(
                canvas,
                (x0, margin),
                (x1, radar_h - margin),
                (42, 112, 49),
                -1,
            )

    def p(x_cm, y_cm):
        return pitch_to_radar_xy(
            x_cm, y_cm, radar_w, radar_h, margin
        )

    line_color = (235, 235, 235)
    thickness = max(1, radar_w // 360)

    cv2.rectangle(
        canvas,
        p(0, 0),
        p(PITCH_LENGTH_CM, PITCH_WIDTH_CM),
        line_color,
        thickness,
        cv2.LINE_AA,
    )

    # Centre line.
    cv2.line(
        canvas,
        p(PITCH_LENGTH_CM / 2, 0),
        p(PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM),
        line_color,
        thickness,
        cv2.LINE_AA,
    )

    # Centre circle.
    centre = p(PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2)
    radius_px = int(
        round(
            CENTRE_CIRCLE_RADIUS_CM
            / PITCH_LENGTH_CM
            * (radar_w - 2 * margin)
        )
    )
    cv2.circle(
        canvas,
        centre,
        radius_px,
        line_color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, centre, 3, line_color, -1, cv2.LINE_AA)

    # Penalty and goal boxes.
    for left in (True, False):
        if left:
            x_goal = 0
            x_pb = PENALTY_BOX_LENGTH_CM
            x_gb = GOAL_BOX_LENGTH_CM
            x_ps = PENALTY_SPOT_DISTANCE_CM
        else:
            x_goal = PITCH_LENGTH_CM
            x_pb = PITCH_LENGTH_CM - PENALTY_BOX_LENGTH_CM
            x_gb = PITCH_LENGTH_CM - GOAL_BOX_LENGTH_CM
            x_ps = PITCH_LENGTH_CM - PENALTY_SPOT_DISTANCE_CM

        pb_y0 = (PITCH_WIDTH_CM - PENALTY_BOX_WIDTH_CM) / 2
        pb_y1 = (PITCH_WIDTH_CM + PENALTY_BOX_WIDTH_CM) / 2
        gb_y0 = (PITCH_WIDTH_CM - GOAL_BOX_WIDTH_CM) / 2
        gb_y1 = (PITCH_WIDTH_CM + GOAL_BOX_WIDTH_CM) / 2

        cv2.rectangle(
            canvas,
            p(min(x_goal, x_pb), pb_y0),
            p(max(x_goal, x_pb), pb_y1),
            line_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            p(min(x_goal, x_gb), gb_y0),
            p(max(x_goal, x_gb), gb_y1),
            line_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            p(x_ps, PITCH_WIDTH_CM / 2),
            3,
            line_color,
            -1,
            cv2.LINE_AA,
        )

    return canvas


def overlay_radar(frame: np.ndarray, radar: np.ndarray, opacity: float):
    h, w = frame.shape[:2]

    radar_w = min(radar.shape[1], int(w * 0.58))
    scale = radar_w / radar.shape[1]
    radar_h = int(round(radar.shape[0] * scale))

    resized = cv2.resize(
        radar,
        (radar_w, radar_h),
        interpolation=cv2.INTER_AREA,
    )

    x0 = (w - radar_w) // 2
    y0 = h - radar_h - 18

    roi = frame[y0:y0 + radar_h, x0:x0 + radar_w]

    blended = cv2.addWeighted(
        roi,
        1.0 - opacity,
        resized,
        opacity,
        0,
    )

    frame[y0:y0 + radar_h, x0:x0 + radar_w] = blended

    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + radar_w, y0 + radar_h),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    source = Path(args.source)
    tracking_path = Path(args.tracking)
    pitch_path = Path(args.pitch)
    output_path = Path(args.output)
    match_state_path = Path(args.match_state)

    for p in (source, tracking_path, pitch_path):
        if not p.exists():
            raise FileNotFoundError(p)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_state_path.parent.mkdir(parents=True, exist_ok=True)

    tracking_by_frame = read_jsonl(tracking_path)
    pitch_by_frame = read_jsonl(pitch_path)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output: {output_path}")

    base_radar = draw_pitch(args.radar_width)
    radar_h, radar_w = base_radar.shape[:2]
    radar_margin = max(18, int(radar_w * 0.035))

    frame_index = 0
    valid_h_count = 0
    reused_h_count = 0
    rejected_h_count = 0
    projected_player_count = 0

    last_valid_H = None
    last_metrics = None

    started = time.perf_counter()

    print("=" * 78)
    print("FootballAnalysisAI - Homography + 2D Radar")
    print(f"Source      : {source}")
    print(f"Tracking    : {tracking_path}")
    print(f"Pitch       : {pitch_path}")
    print(f"Video out   : {output_path}")
    print(f"MatchState  : {match_state_path}")
    print("=" * 78)

    with match_state_path.open("w", encoding="utf-8") as state_file:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if args.max_frames >= 0 and frame_index >= args.max_frames:
                    break

                pitch_item = pitch_by_frame.get(
                    frame_index,
                    {"keypoints": []},
                )
                tracking_item = tracking_by_frame.get(
                    frame_index,
                    {"tracks": []},
                )

                new_H, _, metrics, ids = build_homography(
                    pitch_item.get("keypoints", []),
                    args.min_keypoints,
                )

                homography_source = "none"

                if new_H is not None and metrics is not None:
                    if (
                        metrics["median_reprojection_error_px"]
                        <= args.max_reprojection_error
                    ):
                        last_valid_H = new_H
                        last_metrics = metrics
                        valid_h_count += 1
                        homography_source = "current"
                    else:
                        rejected_h_count += 1

                H = last_valid_H

                if H is not None and homography_source != "current":
                    reused_h_count += 1
                    homography_source = "previous"

                radar = base_radar.copy()

                pitch_tracks = []

                if H is not None:
                    for track in tracking_item.get("tracks", []):
                        foot = track.get("foot_point")
                        if not foot or len(foot) != 2:
                            continue

                        transformed = transform_point(
                            H,
                            float(foot[0]),
                            float(foot[1]),
                        )

                        if transformed is None:
                            continue

                        x_cm, y_cm = transformed

                        # Small tolerance around the pitch.
                        if not (
                            -500 <= x_cm <= PITCH_LENGTH_CM + 500
                            and -500 <= y_cm <= PITCH_WIDTH_CM + 500
                        ):
                            continue

                        class_name = track.get(
                            "class_name",
                            "player",
                        )

                        track_id = int(
                            track.get("track_id", -1)
                        )

                        pitch_x_m = x_cm / 100.0
                        pitch_y_m = y_cm / 100.0

                        pitch_tracks.append({
                            **track,
                            "pitch_position_m": [
                                round(pitch_x_m, 3),
                                round(pitch_y_m, 3),
                            ],
                        })

                        projected_player_count += 1

                        rx, ry = pitch_to_radar_xy(
                            x_cm,
                            y_cm,
                            radar_w,
                            radar_h,
                            radar_margin,
                        )

                        color = CLASS_COLORS_BGR.get(
                            class_name,
                            (0, 255, 255),
                        )

                        cv2.circle(
                            radar,
                            (rx, ry),
                            7,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )

                        if track_id >= 0:
                            cv2.putText(
                                radar,
                                str(track_id),
                                (rx + 8, ry - 7),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.36,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                overlay_radar(
                    frame,
                    radar,
                    args.radar_opacity,
                )

                cv2.putText(
                    frame,
                    (
                        f"Frame {frame_index + 1} | "
                        f"H: {homography_source} | "
                        f"Pitch players: {len(pitch_tracks)}"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if last_metrics:
                    cv2.putText(
                        frame,
                        (
                            f"KP: {last_metrics['num_keypoints']} | "
                            f"inliers: {last_metrics['inliers']} | "
                            f"err: {last_metrics['median_reprojection_error_px']:.1f}px"
                        ),
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                writer.write(frame)

                state_payload = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(
                        frame_index / fps,
                        5,
                    ),
                    "camera": {
                        "homography_source": homography_source,
                        "homography_image_to_pitch": (
                            H.tolist()
                            if H is not None
                            else None
                        ),
                        "metrics": last_metrics,
                        "visible_pitch_keypoint_ids": ids,
                    },
                    "players": pitch_tracks,
                }

                state_file.write(
                    json.dumps(
                        state_payload,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                frame_index += 1

                if frame_index == 1 or frame_index % 25 == 0:
                    elapsed = time.perf_counter() - started
                    effective_fps = frame_index / max(
                        elapsed,
                        1e-6,
                    )

                    print(
                        f"Processed {frame_index}"
                        f"/{total_frames if total_frames > 0 else '?'}"
                        f" | Hcurrent={valid_h_count}"
                        f" | Hreused={reused_h_count}"
                        f" | projected={projected_player_count}"
                        f" | {effective_fps:.1f} FPS"
                    )

        finally:
            cap.release()
            writer.release()

    elapsed = time.perf_counter() - started

    print("=" * 78)
    print("DONE")
    print(f"Frames processed          : {frame_index}")
    print(f"New valid homographies    : {valid_h_count}")
    print(f"Reused homographies       : {reused_h_count}")
    print(f"Rejected homographies     : {rejected_h_count}")
    print(f"Projected player samples  : {projected_player_count}")
    print(f"Elapsed                   : {elapsed:.1f} s")
    print(
        f"Average FPS               : "
        f"{frame_index / max(elapsed, 1e-6):.1f}"
    )
    print(f"Radar video               : {output_path.resolve()}")
    print(f"MatchState JSONL          : {match_state_path.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
