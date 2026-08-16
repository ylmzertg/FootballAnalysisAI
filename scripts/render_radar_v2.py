from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


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


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - scene-aware homography + 2D radar"
    )
    p.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument("--tracking", default=r"output\player_tracking.jsonl")
    p.add_argument("--pitch", default=r"output\pitch_keypoints.jsonl")
    p.add_argument("--scenes", default=r"output\scene_labels.jsonl")
    p.add_argument("--output", default=r"output\radar_v2.mp4")
    p.add_argument("--match-state", default=r"output\match_state_v2.jsonl")
    p.add_argument("--min-keypoints", type=int, default=4)
    p.add_argument("--max-reprojection-error", type=float, default=35.0)
    p.add_argument("--radar-width", type=int, default=720)
    p.add_argument("--radar-opacity", type=float, default=0.82)
    return p.parse_args()


def read_jsonl(path: Path) -> dict[int, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            out[int(item["frame_index"])] = item
    return out


def build_homography(keypoints, min_keypoints):
    src, dst, ids = [], [], []

    for kp in keypoints:
        kp_id = int(kp["keypoint_id"])
        if 0 <= kp_id < len(PITCH_VERTICES):
            src.append([float(kp["x"]), float(kp["y"])])
            dst.append(PITCH_VERTICES[kp_id].tolist())
            ids.append(kp_id)

    if len(src) < min_keypoints:
        return None, None, ids

    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)

    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=500.0,
    )

    if H is None:
        return None, None, ids

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, None, ids

    reproj = cv2.perspectiveTransform(
        dst.reshape(-1, 1, 2),
        H_inv.astype(np.float64),
    ).reshape(-1, 2)

    errors = np.linalg.norm(reproj - src, axis=1)

    metrics = {
        "num_keypoints": len(src),
        "inliers": int(mask.sum()) if mask is not None else len(src),
        "median_reprojection_error_px": round(float(np.median(errors)), 3),
        "max_reprojection_error_px": round(float(np.max(errors)), 3),
    }

    return H, metrics, ids


def transform_point(H, x, y):
    point = np.array([[[x, y]]], dtype=np.float32)
    result = cv2.perspectiveTransform(
        point,
        H.astype(np.float64),
    )[0, 0]

    if not np.isfinite(result).all():
        return None

    return float(result[0]), float(result[1])


def pitch_to_radar_xy(x_cm, y_cm, radar_w, radar_h, margin):
    usable_w = radar_w - 2 * margin
    usable_h = radar_h - 2 * margin

    x = margin + (x_cm / PITCH_LENGTH_CM) * usable_w
    y = margin + (y_cm / PITCH_WIDTH_CM) * usable_h

    return int(round(x)), int(round(y))


def draw_pitch(radar_w):
    radar_h = int(round(radar_w * PITCH_WIDTH_CM / PITCH_LENGTH_CM))
    margin = max(18, int(radar_w * 0.035))

    canvas = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)
    canvas[:] = (36, 103, 43)

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
                (41, 112, 48),
                -1,
            )

    def p(x_cm, y_cm):
        return pitch_to_radar_xy(
            x_cm, y_cm, radar_w, radar_h, margin
        )

    white = (235, 235, 235)
    thickness = max(1, radar_w // 360)

    cv2.rectangle(
        canvas,
        p(0, 0),
        p(PITCH_LENGTH_CM, PITCH_WIDTH_CM),
        white,
        thickness,
        cv2.LINE_AA,
    )

    cv2.line(
        canvas,
        p(PITCH_LENGTH_CM / 2, 0),
        p(PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM),
        white,
        thickness,
        cv2.LINE_AA,
    )

    centre = p(PITCH_LENGTH_CM / 2, PITCH_WIDTH_CM / 2)
    radius_px = int(
        round(
            CENTRE_CIRCLE_RADIUS_CM
            / PITCH_LENGTH_CM
            * (radar_w - 2 * margin)
        )
    )
    cv2.circle(canvas, centre, radius_px, white, thickness, cv2.LINE_AA)
    cv2.circle(canvas, centre, 3, white, -1, cv2.LINE_AA)

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
            white,
            thickness,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            p(min(x_goal, x_gb), gb_y0),
            p(max(x_goal, x_gb), gb_y1),
            white,
            thickness,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            p(x_ps, PITCH_WIDTH_CM / 2),
            3,
            white,
            -1,
            cv2.LINE_AA,
        )

    return canvas


def overlay_radar(frame, radar, opacity):
    h, w = frame.shape[:2]

    radar_w = min(radar.shape[1], int(w * 0.58))
    scale = radar_w / radar.shape[1]
    radar_h = int(round(radar.shape[0] * scale))

    radar = cv2.resize(
        radar,
        (radar_w, radar_h),
        interpolation=cv2.INTER_AREA,
    )

    x0 = (w - radar_w) // 2
    y0 = h - radar_h - 18

    roi = frame[y0:y0 + radar_h, x0:x0 + radar_w]

    frame[y0:y0 + radar_h, x0:x0 + radar_w] = cv2.addWeighted(
        roi,
        1.0 - opacity,
        radar,
        opacity,
        0,
    )


def main():
    args = parse_args()

    source = Path(args.source)
    tracking_path = Path(args.tracking)
    pitch_path = Path(args.pitch)
    scenes_path = Path(args.scenes)
    output_path = Path(args.output)
    match_state_path = Path(args.match_state)

    for path in (
        source,
        tracking_path,
        pitch_path,
        scenes_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    tracking = read_jsonl(tracking_path)
    pitch = read_jsonl(pitch_path)
    scenes = read_jsonl(scenes_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_state_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(source)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    base_radar = draw_pitch(args.radar_width)
    rh, rw = base_radar.shape[:2]
    margin = max(18, int(rw * 0.035))

    frame_index = 0
    last_valid_H = None
    last_metrics = None

    stats = {
        "analysis_frames": 0,
        "paused_frames": 0,
        "new_h": 0,
        "reused_h": 0,
        "rejected_h": 0,
        "projected": 0,
    }

    started = time.perf_counter()

    with match_state_path.open("w", encoding="utf-8") as out:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            scene = scenes.get(
                frame_index,
                {
                    "label": "OTHER",
                    "analysis_enabled": False,
                },
            )

            scene_label = scene.get("label", "OTHER")
            analysis_enabled = bool(
                scene.get("analysis_enabled", False)
            )

            tracks = tracking.get(
                frame_index, {"tracks": []}
            ).get("tracks", [])

            keypoints = pitch.get(
                frame_index, {"keypoints": []}
            ).get("keypoints", [])

            homography_source = "none"
            pitch_tracks = []
            visible_ids = []

            if analysis_enabled:
                stats["analysis_frames"] += 1

                H_new, metrics, visible_ids = build_homography(
                    keypoints,
                    args.min_keypoints,
                )

                if H_new is not None and metrics is not None:
                    if (
                        metrics["median_reprojection_error_px"]
                        <= args.max_reprojection_error
                    ):
                        last_valid_H = H_new
                        last_metrics = metrics
                        homography_source = "current"
                        stats["new_h"] += 1
                    else:
                        stats["rejected_h"] += 1

                H = last_valid_H

                if H is not None and homography_source != "current":
                    homography_source = "previous"
                    stats["reused_h"] += 1

                radar = base_radar.copy()

                if H is not None:
                    for tr in tracks:
                        foot = tr.get("foot_point")
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

                        if not (
                            0 <= x_cm <= PITCH_LENGTH_CM
                            and 0 <= y_cm <= PITCH_WIDTH_CM
                        ):
                            continue

                        track_id = int(tr.get("track_id", -1))
                        class_name = tr.get("class_name", "player")

                        pitch_tracks.append({
                            **tr,
                            "pitch_position_m": [
                                round(x_cm / 100.0, 3),
                                round(y_cm / 100.0, 3),
                            ],
                        })

                        stats["projected"] += 1

                        rx, ry = pitch_to_radar_xy(
                            x_cm, y_cm, rw, rh, margin
                        )

                        color = (
                            (0, 255, 255)
                            if class_name == "player"
                            else
                            (255, 0, 255)
                            if class_name == "goalkeeper"
                            else
                            (0, 165, 255)
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

            else:
                stats["paused_frames"] += 1
                last_valid_H = None
                last_metrics = None

                cv2.rectangle(
                    frame,
                    (18, h - 68),
                    (330, h - 18),
                    (20, 20, 20),
                    -1,
                )
                cv2.putText(
                    frame,
                    f"TACTICAL ANALYSIS PAUSED - {scene_label}",
                    (30, h - 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (190, 190, 190),
                    1,
                    cv2.LINE_AA,
                )

            cv2.rectangle(
                frame,
                (0, 0),
                (w, 74),
                (18, 18, 18),
                -1,
            )

            cv2.putText(
                frame,
                (
                    f"Scene: {scene_label} | "
                    f"Analysis: {'ON' if analysis_enabled else 'OFF'}"
                ),
                (18, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 220, 0) if analysis_enabled else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"H: {homography_source} | "
                    f"pitch players: {len(pitch_tracks)}"
                ),
                (18, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)

            payload = {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / fps, 5),
                "scene": scene_label,
                "analysis_enabled": analysis_enabled,
                "camera": {
                    "homography_source": homography_source,
                    "homography_image_to_pitch": (
                        last_valid_H.tolist()
                        if analysis_enabled and last_valid_H is not None
                        else None
                    ),
                    "metrics": last_metrics if analysis_enabled else None,
                    "visible_pitch_keypoint_ids": visible_ids,
                },
                "players": pitch_tracks,
            }

            out.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )

            frame_index += 1

            if frame_index == 1 or frame_index % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"Processed {frame_index}/{total} | "
                    f"analysis={stats['analysis_frames']} | "
                    f"paused={stats['paused_frames']} | "
                    f"projected={stats['projected']} | "
                    f"{frame_index / max(elapsed, 1e-6):.1f} FPS"
                )

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - started

    print("=" * 76)
    print("DONE")
    print(f"Frames processed     : {frame_index}")
    print(f"Analysis ON frames   : {stats['analysis_frames']}")
    print(f"Paused frames        : {stats['paused_frames']}")
    print(f"New homographies     : {stats['new_h']}")
    print(f"Reused homographies  : {stats['reused_h']}")
    print(f"Rejected homographies: {stats['rejected_h']}")
    print(f"Projected samples    : {stats['projected']}")
    print(f"Elapsed              : {elapsed:.2f} s")
    print(f"Output               : {output_path.resolve()}")
    print(f"MatchState           : {match_state_path.resolve()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
