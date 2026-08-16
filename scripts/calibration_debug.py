from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Roboflow Sports soccer pitch geometry, centimeters.
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

# 1-based edge indices from Roboflow Sports config.
PITCH_EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
    (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
    (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
    (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - calibration debug overlay"
    )
    p.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument(
        "--pitch",
        default=r"output\pitch_keypoints.jsonl",
    )
    p.add_argument(
        "--scenes",
        default=r"output\scene_labels_v2.jsonl",
    )
    p.add_argument(
        "--output",
        default=r"output\calibration_debug.mp4",
    )
    p.add_argument(
        "--jsonl",
        default=r"output\calibration_debug.jsonl",
    )
    p.add_argument(
        "--min-keypoints",
        type=int,
        default=4,
    )
    p.add_argument(
        "--max-image-error-px",
        type=float,
        default=25.0,
        help="Reject current-frame solution above this median image reprojection error.",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows[int(item["frame_index"])] = item
    return rows


def collect_correspondences(keypoints: list[dict]):
    src = []
    dst = []
    ids = []

    for kp in keypoints:
        kp_id = int(kp["keypoint_id"])
        if 0 <= kp_id < len(PITCH_VERTICES):
            x = float(kp["x"])
            y = float(kp["y"])
            if x <= 1 or y <= 1:
                continue

            src.append([x, y])
            dst.append(PITCH_VERTICES[kp_id].tolist())
            ids.append(kp_id)

    return (
        np.asarray(src, dtype=np.float32),
        np.asarray(dst, dtype=np.float32),
        ids,
    )


def fit_candidate(src: np.ndarray, dst: np.ndarray, method: str):
    if len(src) < 4:
        return None

    if method == "direct":
        H, mask = cv2.findHomography(src, dst, method=0)
    elif method == "ransac":
        # Threshold is in target coordinate units (centimeters).
        # 150 cm is intentionally much tighter than the old 500 cm value.
        H, mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=150.0,
            maxIters=4000,
            confidence=0.995,
        )
    else:
        raise ValueError(method)

    if H is None:
        return None

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    projected_img = cv2.perspectiveTransform(
        dst.reshape(-1, 1, 2),
        H_inv.astype(np.float64),
    ).reshape(-1, 2)

    errors_px = np.linalg.norm(projected_img - src, axis=1)

    inliers = (
        int(mask.sum())
        if mask is not None
        else len(src)
    )

    return {
        "method": method,
        "H": H,
        "H_inv": H_inv,
        "median_error_px": float(np.median(errors_px)),
        "mean_error_px": float(np.mean(errors_px)),
        "max_error_px": float(np.max(errors_px)),
        "inliers": inliers,
    }


def choose_candidate(src: np.ndarray, dst: np.ndarray):
    candidates = []

    for method in ("direct", "ransac"):
        item = fit_candidate(src, dst, method)
        if item is not None:
            candidates.append(item)

    if not candidates:
        return None, []

    # Primary criterion: median image reprojection error.
    # Tiny tie-break preference for more inliers.
    candidates.sort(
        key=lambda c: (
            c["median_error_px"],
            -c["inliers"],
        )
    )

    return candidates[0], candidates


def project_pitch_vertices(H_inv: np.ndarray):
    pts = cv2.perspectiveTransform(
        PITCH_VERTICES.reshape(-1, 1, 2),
        H_inv.astype(np.float64),
    ).reshape(-1, 2)

    return pts


def safe_point(pt, width, height, margin=300):
    x, y = float(pt[0]), float(pt[1])

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    if (
        x < -margin
        or x > width + margin
        or y < -margin
        or y > height + margin
    ):
        return None

    return int(round(x)), int(round(y))


def draw_projected_pitch(
    frame: np.ndarray,
    projected_vertices: np.ndarray,
):
    h, w = frame.shape[:2]

    # Cyan = model pitch geometry projected back into the TV image.
    edge_color = (255, 255, 0)

    for a, b in PITCH_EDGES:
        p1 = safe_point(
            projected_vertices[a - 1],
            w,
            h,
        )
        p2 = safe_point(
            projected_vertices[b - 1],
            w,
            h,
        )

        if p1 is None or p2 is None:
            continue

        cv2.line(
            frame,
            p1,
            p2,
            edge_color,
            2,
            cv2.LINE_AA,
        )

    # Draw pitch model vertices too.
    for idx, pt in enumerate(projected_vertices):
        p = safe_point(pt, w, h)
        if p is None:
            continue

        cv2.circle(
            frame,
            p,
            4,
            (255, 255, 0),
            -1,
            cv2.LINE_AA,
        )


def draw_detected_keypoints(
    frame: np.ndarray,
    keypoints: list[dict],
):
    for kp in keypoints:
        x = int(round(float(kp["x"])))
        y = int(round(float(kp["y"])))
        kp_id = int(kp["keypoint_id"])

        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            str(kp_id),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()

    source = Path(args.source)
    pitch_path = Path(args.pitch)
    scenes_path = Path(args.scenes)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl)

    for path in (source, pitch_path, scenes_path):
        if not path.exists():
            raise FileNotFoundError(path)

    pitch_by_frame = read_jsonl(pitch_path)
    scene_by_frame = read_jsonl(scenes_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")

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
        raise RuntimeError(f"Could not create output video: {output_path}")

    last_valid = None

    stats = {
        "frames": 0,
        "current": 0,
        "reused": 0,
        "rejected": 0,
        "disabled": 0,
    }

    with jsonl_path.open("w", encoding="utf-8") as out:
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            scene = scene_by_frame.get(
                frame_index,
                {"label": "OTHER", "analysis_enabled": False},
            )

            scene_label = scene.get("label", "OTHER")
            analysis_enabled = bool(
                scene.get("analysis_enabled", False)
            )

            keypoints = pitch_by_frame.get(
                frame_index,
                {"keypoints": []},
            ).get("keypoints", [])

            source_kind = "none"
            chosen = None
            candidate_summaries = []

            if analysis_enabled:
                src, dst, ids = collect_correspondences(keypoints)

                if len(src) >= args.min_keypoints:
                    chosen, all_candidates = choose_candidate(src, dst)

                    candidate_summaries = [
                        {
                            "method": c["method"],
                            "median_error_px": round(
                                c["median_error_px"], 3
                            ),
                            "mean_error_px": round(
                                c["mean_error_px"], 3
                            ),
                            "max_error_px": round(
                                c["max_error_px"], 3
                            ),
                            "inliers": c["inliers"],
                        }
                        for c in all_candidates
                    ]

                    if (
                        chosen is not None
                        and chosen["median_error_px"]
                        <= args.max_image_error_px
                    ):
                        last_valid = chosen
                        source_kind = "current"
                        stats["current"] += 1
                    else:
                        chosen = None
                        stats["rejected"] += 1

                if chosen is None and last_valid is not None:
                    chosen = last_valid
                    source_kind = "previous"
                    stats["reused"] += 1

            else:
                # Hard reset on non-tactical shot.
                last_valid = None
                stats["disabled"] += 1

            debug_frame = frame.copy()

            if chosen is not None:
                projected_vertices = project_pitch_vertices(
                    chosen["H_inv"]
                )
                draw_projected_pitch(
                    debug_frame,
                    projected_vertices,
                )

            # Red points = detected pitch keypoints.
            draw_detected_keypoints(
                debug_frame,
                keypoints,
            )

            cv2.rectangle(
                debug_frame,
                (0, 0),
                (width, 96),
                (15, 15, 15),
                -1,
            )

            color = (
                (0, 220, 0)
                if analysis_enabled
                else (0, 0, 255)
            )

            cv2.putText(
                debug_frame,
                (
                    f"Scene: {scene_label} | "
                    f"Analysis: {'ON' if analysis_enabled else 'OFF'} | "
                    f"H: {source_kind}"
                ),
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

            if chosen is not None:
                cv2.putText(
                    debug_frame,
                    (
                        f"method={chosen['method']} | "
                        f"median err={chosen['median_error_px']:.1f}px | "
                        f"mean={chosen['mean_error_px']:.1f}px | "
                        f"inliers={chosen['inliers']} | "
                        f"kp={len(keypoints)}"
                    ),
                    (18, 61),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug_frame,
                    "CYAN = projected pitch model | RED = detected keypoints",
                    (18, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    debug_frame,
                    (
                        f"No usable homography | "
                        f"kp={len(keypoints)}"
                    ),
                    (18, 61),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(debug_frame)

            payload = {
                "frame_index": frame_index,
                "timestamp_seconds": round(
                    frame_index / fps,
                    5,
                ),
                "scene": scene_label,
                "analysis_enabled": analysis_enabled,
                "homography_source": source_kind,
                "selected_method": (
                    chosen["method"]
                    if chosen is not None
                    else None
                ),
                "selected_median_error_px": (
                    round(chosen["median_error_px"], 3)
                    if chosen is not None
                    else None
                ),
                "candidate_metrics": candidate_summaries,
            }

            out.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )

            frame_index += 1
            stats["frames"] += 1

            if frame_index == 1 or frame_index % 25 == 0:
                print(
                    f"Processed {frame_index}/{total_frames} | "
                    f"current={stats['current']} | "
                    f"reused={stats['reused']} | "
                    f"rejected={stats['rejected']} | "
                    f"disabled={stats['disabled']}"
                )

    cap.release()
    writer.release()

    print("=" * 78)
    print("DONE")
    print(f"Frames             : {stats['frames']}")
    print(f"Current H          : {stats['current']}")
    print(f"Reused H           : {stats['reused']}")
    print(f"Rejected current H : {stats['rejected']}")
    print(f"Analysis disabled  : {stats['disabled']}")
    print(f"Video              : {output_path.resolve()}")
    print(f"JSONL              : {jsonl_path.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
