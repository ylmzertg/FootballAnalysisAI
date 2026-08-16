from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Roboflow Sports pitch geometry, centimeters.
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

# 1-based edge indices copied from the Roboflow Sports soccer config.
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
        description="FootballAnalysisAI - confidence-aware calibration debug v2"
    )
    p.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument("--pitch", default=r"output\pitch_keypoints.jsonl")
    p.add_argument("--scenes", default=r"output\scene_labels_v2.jsonl")
    p.add_argument("--output", default=r"output\calibration_debug_v2.mp4")
    p.add_argument("--jsonl", default=r"output\calibration_debug_v2.jsonl")
    p.add_argument("--min-keypoints", type=int, default=4)

    p.add_argument(
        "--confidence-thresholds",
        default="0.20,0.30,0.40,0.50,0.60,0.70",
        help="Candidate keypoint-confidence thresholds.",
    )
    p.add_argument(
        "--max-keypoint-error-px",
        type=float,
        default=20.0,
        help="Hard rejection on median image-space keypoint reprojection error.",
    )
    p.add_argument(
        "--max-line-error-px",
        type=float,
        default=22.0,
        help="Hard rejection on median projected-line to white-line distance.",
    )
    p.add_argument(
        "--min-visible-line-samples",
        type=int,
        default=80,
        help="Require enough projected pitch-line samples inside the image.",
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


def build_white_line_distance(frame: np.ndarray) -> np.ndarray:
    """
    Build an approximate football-pitch white-line distance map.

    We intentionally don't require perfect segmentation. We only need a useful
    secondary score that punishes projected pitch models that visibly miss
    the painted field lines.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Grass: broad range for TV broadcasts.
    grass = cv2.inRange(
        hsv,
        np.array([25, 35, 25], dtype=np.uint8),
        np.array([100, 255, 255], dtype=np.uint8),
    )

    # Allow a white pixel to count as a field line only if it lies close to grass.
    grass_near = cv2.dilate(
        grass,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    )

    # Bright / relatively unsaturated paint.
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 115], dtype=np.uint8),
        np.array([179, 100, 255], dtype=np.uint8),
    )

    white = cv2.bitwise_and(white, grass_near)

    # Remove tiny speckles while preserving long painted lines.
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )

    # Distance to nearest white-line candidate.
    inv = cv2.bitwise_not(white)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    return dist


def collect_correspondences(
    keypoints: list[dict],
    confidence_threshold: float,
):
    src, dst, ids, confs = [], [], [], []

    for kp in keypoints:
        kp_id = int(kp["keypoint_id"])
        confidence = float(kp.get("confidence", 1.0))

        if confidence < confidence_threshold:
            continue

        if not (0 <= kp_id < len(PITCH_VERTICES)):
            continue

        x = float(kp["x"])
        y = float(kp["y"])

        if x <= 1 or y <= 1:
            continue

        src.append([x, y])
        dst.append(PITCH_VERTICES[kp_id].tolist())
        ids.append(kp_id)
        confs.append(confidence)

    return (
        np.asarray(src, dtype=np.float32),
        np.asarray(dst, dtype=np.float32),
        ids,
        confs,
    )


def target_coverage_metrics(dst: np.ndarray):
    if len(dst) < 4:
        return 0.0, 0.0, 0.0

    x_span = float(dst[:, 0].max() - dst[:, 0].min())
    y_span = float(dst[:, 1].max() - dst[:, 1].min())

    hull = cv2.convexHull(dst.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))

    return x_span, y_span, hull_area


def fit_homography(src: np.ndarray, dst: np.ndarray, mode: str):
    if len(src) < 4:
        return None

    if mode == "direct":
        H, inlier_mask = cv2.findHomography(src, dst, method=0)

    elif mode.startswith("ransac"):
        threshold_cm = float(mode.split("_")[1])
        H, inlier_mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold_cm,
            maxIters=5000,
            confidence=0.995,
        )

    else:
        raise ValueError(mode)

    if H is None:
        return None

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    # Known-keypoint reprojection error, in image pixels.
    reprojected_img = cv2.perspectiveTransform(
        dst.reshape(-1, 1, 2),
        H_inv.astype(np.float64),
    ).reshape(-1, 2)

    errors = np.linalg.norm(reprojected_img - src, axis=1)

    return {
        "mode": mode,
        "H": H,
        "H_inv": H_inv,
        "inliers": (
            int(inlier_mask.sum())
            if inlier_mask is not None
            else len(src)
        ),
        "kp_median_error_px": float(np.median(errors)),
        "kp_mean_error_px": float(np.mean(errors)),
        "kp_max_error_px": float(np.max(errors)),
    }


def dense_pitch_line_points(step_cm: float = 80.0) -> np.ndarray:
    """
    Densely sample all straight field edges so projected line alignment can be scored.
    """
    points = []

    for a, b in PITCH_EDGES:
        p1 = PITCH_VERTICES[a - 1]
        p2 = PITCH_VERTICES[b - 1]

        length = float(np.linalg.norm(p2 - p1))
        n = max(2, int(length / step_cm) + 1)

        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        segment = p1[None, :] * (1.0 - t) + p2[None, :] * t
        points.append(segment)

    return np.concatenate(points, axis=0).astype(np.float32)


DENSE_LINE_POINTS = dense_pitch_line_points()


def score_line_alignment(
    H_inv: np.ndarray,
    distance_map: np.ndarray,
):
    h, w = distance_map.shape[:2]

    projected = cv2.perspectiveTransform(
        DENSE_LINE_POINTS.reshape(-1, 1, 2),
        H_inv.astype(np.float64),
    ).reshape(-1, 2)

    finite = np.isfinite(projected).all(axis=1)

    x = projected[:, 0]
    y = projected[:, 1]

    inside = (
        finite
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )

    visible = projected[inside]

    if len(visible) == 0:
        return float("inf"), float("inf"), 0

    xi = np.clip(np.rint(visible[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(visible[:, 1]).astype(int), 0, h - 1)

    distances = distance_map[yi, xi]

    return (
        float(np.median(distances)),
        float(np.mean(distances)),
        int(len(visible)),
    )


def candidate_score(candidate: dict):
    """
    Lower is better.

    Keypoint fit is necessary but can be misleading when only a local cluster
    is visible. Field-line alignment is therefore given substantial weight.
    """
    coverage_penalty = 0.0

    # Prefer correspondences that span both axes of the real pitch.
    if candidate["target_x_span_cm"] < 1500:
        coverage_penalty += 20.0
    if candidate["target_y_span_cm"] < 1500:
        coverage_penalty += 20.0
    if candidate["target_hull_area_cm2"] < 1_000_000:
        coverage_penalty += 20.0

    low_count_penalty = max(0, 6 - candidate["num_keypoints"]) * 4.0

    return (
        candidate["kp_median_error_px"]
        + 1.25 * candidate["line_median_error_px"]
        + coverage_penalty
        + low_count_penalty
    )


def build_candidates(
    frame: np.ndarray,
    keypoints: list[dict],
    confidence_thresholds: list[float],
    min_keypoints: int,
):
    distance_map = build_white_line_distance(frame)
    candidates = []

    modes = (
        "direct",
        "ransac_75",
        "ransac_100",
        "ransac_150",
    )

    for conf_thr in confidence_thresholds:
        src, dst, ids, confs = collect_correspondences(
            keypoints,
            conf_thr,
        )

        if len(src) < min_keypoints:
            continue

        x_span, y_span, hull_area = target_coverage_metrics(dst)

        for mode in modes:
            fitted = fit_homography(src, dst, mode)
            if fitted is None:
                continue

            line_med, line_mean, visible_line_samples = score_line_alignment(
                fitted["H_inv"],
                distance_map,
            )

            item = {
                **fitted,
                "confidence_threshold": conf_thr,
                "num_keypoints": len(src),
                "keypoint_ids": ids,
                "mean_keypoint_confidence": (
                    float(np.mean(confs))
                    if confs
                    else 0.0
                ),
                "target_x_span_cm": x_span,
                "target_y_span_cm": y_span,
                "target_hull_area_cm2": hull_area,
                "line_median_error_px": line_med,
                "line_mean_error_px": line_mean,
                "visible_line_samples": visible_line_samples,
            }

            item["score"] = candidate_score(item)
            candidates.append(item)

    candidates.sort(key=lambda c: c["score"])
    return candidates


def project_vertices(H_inv: np.ndarray):
    return cv2.perspectiveTransform(
        PITCH_VERTICES.reshape(-1, 1, 2),
        H_inv.astype(np.float64),
    ).reshape(-1, 2)


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


def draw_pitch_overlay(
    frame: np.ndarray,
    H_inv: np.ndarray,
):
    h, w = frame.shape[:2]
    vertices = project_vertices(H_inv)

    # CYAN = selected pitch model.
    for a, b in PITCH_EDGES:
        p1 = safe_point(vertices[a - 1], w, h)
        p2 = safe_point(vertices[b - 1], w, h)

        if p1 is None or p2 is None:
            continue

        cv2.line(
            frame,
            p1,
            p2,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_keypoints(
    frame: np.ndarray,
    keypoints: list[dict],
    selected_ids: set[int],
):
    for kp in keypoints:
        kp_id = int(kp["keypoint_id"])
        conf = float(kp.get("confidence", 1.0))
        x = int(round(float(kp["x"])))
        y = int(round(float(kp["y"])))

        selected = kp_id in selected_ids

        color = (0, 0, 255) if selected else (90, 90, 90)
        radius = 6 if selected else 4

        cv2.circle(
            frame,
            (x, y),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"{kp_id}:{conf:.2f}",
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def serialize_candidate(c: dict):
    return {
        "mode": c["mode"],
        "confidence_threshold": round(
            c["confidence_threshold"], 3
        ),
        "num_keypoints": c["num_keypoints"],
        "keypoint_ids": c["keypoint_ids"],
        "mean_keypoint_confidence": round(
            c["mean_keypoint_confidence"], 4
        ),
        "kp_median_error_px": round(
            c["kp_median_error_px"], 3
        ),
        "line_median_error_px": round(
            c["line_median_error_px"], 3
        ),
        "visible_line_samples": c["visible_line_samples"],
        "target_x_span_cm": round(
            c["target_x_span_cm"], 1
        ),
        "target_y_span_cm": round(
            c["target_y_span_cm"], 1
        ),
        "target_hull_area_cm2": round(
            c["target_hull_area_cm2"], 1
        ),
        "score": round(c["score"], 3),
    }


def main():
    args = parse_args()

    source = Path(args.source)
    pitch_path = Path(args.pitch)
    scenes_path = Path(args.scenes)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl)

    for p in (source, pitch_path, scenes_path):
        if not p.exists():
            raise FileNotFoundError(p)

    pitch = read_jsonl(pitch_path)
    scenes = read_jsonl(scenes_path)

    thresholds = [
        float(x.strip())
        for x in args.confidence_thresholds.split(",")
        if x.strip()
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

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

    frame_index = 0
    last_good = None

    stats = {
        "current": 0,
        "previous": 0,
        "rejected": 0,
        "disabled": 0,
    }

    with jsonl_path.open("w", encoding="utf-8") as out:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            scene = scenes.get(
                frame_index,
                {"label": "OTHER", "analysis_enabled": False},
            )

            label = scene.get("label", "OTHER")
            enabled = bool(scene.get("analysis_enabled", False))

            keypoints = pitch.get(
                frame_index,
                {"keypoints": []},
            ).get("keypoints", [])

            chosen = None
            current_candidates = []
            source_kind = "none"

            if enabled:
                current_candidates = build_candidates(
                    frame,
                    keypoints,
                    thresholds,
                    args.min_keypoints,
                )

                if current_candidates:
                    best = current_candidates[0]

                    good = (
                        best["kp_median_error_px"]
                        <= args.max_keypoint_error_px
                        and best["line_median_error_px"]
                        <= args.max_line_error_px
                        and best["visible_line_samples"]
                        >= args.min_visible_line_samples
                    )

                    if good:
                        chosen = best
                        last_good = best
                        source_kind = "current"
                        stats["current"] += 1
                    else:
                        stats["rejected"] += 1

                if chosen is None and last_good is not None:
                    chosen = last_good
                    source_kind = "previous"
                    stats["previous"] += 1

            else:
                last_good = None
                stats["disabled"] += 1

            debug = frame.copy()

            if chosen is not None:
                draw_pitch_overlay(debug, chosen["H_inv"])

            selected_ids = (
                set(chosen["keypoint_ids"])
                if chosen is not None
                else set()
            )

            draw_keypoints(
                debug,
                keypoints,
                selected_ids,
            )

            cv2.rectangle(
                debug,
                (0, 0),
                (w, 112),
                (15, 15, 15),
                -1,
            )

            status_color = (
                (0, 220, 0)
                if source_kind == "current"
                else
                (0, 210, 255)
                if source_kind == "previous"
                else
                (0, 0, 255)
            )

            cv2.putText(
                debug,
                (
                    f"Scene: {label} | "
                    f"H: {source_kind}"
                ),
                (18, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                status_color,
                2,
                cv2.LINE_AA,
            )

            if chosen is not None:
                cv2.putText(
                    debug,
                    (
                        f"{chosen['mode']} | "
                        f"conf>={chosen['confidence_threshold']:.2f} | "
                        f"kp={chosen['num_keypoints']} | "
                        f"kpErr={chosen['kp_median_error_px']:.1f}px | "
                        f"lineErr={chosen['line_median_error_px']:.1f}px"
                    ),
                    (18, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.51,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug,
                    (
                        f"score={chosen['score']:.1f} | "
                        f"visible line samples={chosen['visible_line_samples']} | "
                        f"selected keypoints={chosen['keypoint_ids']}"
                    ),
                    (18, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                debug,
                "CYAN=model lines | RED=selected keypoints | GRAY=discarded keypoints",
                (18, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

            writer.write(debug)

            payload = {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / fps, 5),
                "scene": label,
                "homography_source": source_kind,
                "selected": (
                    serialize_candidate(chosen)
                    if chosen is not None
                    else None
                ),
                "top_candidates": [
                    serialize_candidate(c)
                    for c in current_candidates[:5]
                ],
                "homography_image_to_pitch": (
                    chosen["H"].tolist()
                    if chosen is not None
                    else None
                ),
            }

            out.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )

            frame_index += 1

            if frame_index == 1 or frame_index % 25 == 0:
                print(
                    f"Processed {frame_index}/{total} | "
                    f"current={stats['current']} | "
                    f"previous={stats['previous']} | "
                    f"rejected={stats['rejected']} | "
                    f"disabled={stats['disabled']}"
                )

    cap.release()
    writer.release()

    print("=" * 78)
    print("DONE")
    print(f"Frames           : {frame_index}")
    print(f"Current H        : {stats['current']}")
    print(f"Previous H       : {stats['previous']}")
    print(f"Rejected H       : {stats['rejected']}")
    print(f"Disabled frames  : {stats['disabled']}")
    print(f"Output           : {output_path.resolve()}")
    print(f"JSONL            : {jsonl_path.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
