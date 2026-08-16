from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.pnlcalib import PnLCalibAdapter
from core.team_classifier_v24 import (
    GOALKEEPER,
    PLAYER,
    REFEREE,
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    DetectionObservation,
    TeamAssignment,
    TeamClassifierV2,
    TeamClassifierV2Config,
)

from core.track_identity import (
    CanonicalIdentityConfig,
    build_canonical_alias_map,
    collapse_frame_records,
)


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

ROLE_HINTS = {
    "player": PLAYER,
    "goalkeeper": GOALKEEPER,
    "referee": REFEREE,
}

TEAM_COLORS = {
    TEAM_A: (255, 90, 40),
    TEAM_B: (40, 70, 255),
    UNKNOWN: (160, 160, 160),
}
ROLE_COLORS = {
    REFEREE: (0, 215, 255),
    GOALKEEPER: (220, 90, 220),
}
OUTSIDE_COLOR = (115, 115, 115)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "FootballAnalysisAI - Team Classifier V2.4 + PnLCalib + Canonical ID spatial role gate"
        )
    )
    p.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument("--tracking", default=r"output\player_tracking.jsonl")
    p.add_argument(
        "--pnl-root",
        default=r"E:\Youtube\SporAnimasyon\CalibrationEngines\PnLCalib",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output",
        default=r"output\team_classification_v24_pnl_exact.mp4",
    )
    p.add_argument(
        "--jsonl",
        default=r"output\team_classification_v24_pnl_exact.jsonl",
    )
    p.add_argument(
        "--calibration-json",
        default=r"output\team_classification_v24_pnl_exact_calibration.json",
    )
    p.add_argument(
        "--calibration-frames-dir",
        default=r"output\v24_pnl_frames_exact",
    )
    p.add_argument(
        "--calibration-stride",
        type=int,
        default=1,
        help=(
            "V2.4 validation defaults to exact per-frame PnLCalib. "
            "5 is the GTX 1050 validation default; Radar v4 can later use a denser cadence."
        ),
    )
    p.add_argument(
        "--max-calibration-gap",
        type=int,
        default=0,
        help="V2.4 exact validation does not reuse nearby homographies by default.",
    )
    p.add_argument(
        "--min-calibration-quality",
        type=float,
        default=0.0,
        help="Optional PnL quality_score floor. 0 keeps every successful solution.",
    )
    p.add_argument(
        "--pitch-margin",
        type=float,
        default=0.0,
        help="Exact-PnL validation uses strict 0..105 x 0..68 pitch bounds by default.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Process at most this many frames. Use -1 for all tracked frames.",
    )
    p.add_argument("--embedding-stride", type=int, default=5)
    p.add_argument("--bootstrap-min-samples", type=int, default=30)
    return p.parse_args()


def resolve_project_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_jsonl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows[int(item["frame_index"])] = item
    return rows


def transform_point(H: np.ndarray, x: float, y: float) -> Optional[tuple[float, float]]:
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(float(p[2])) < 1e-12:
        return None
    p = p / p[2]
    if not np.isfinite(p).all():
        return None
    return float(p[0]), float(p[1])


def inside_pitch(xy: tuple[float, float], margin: float) -> bool:
    x, y = xy
    return (
        -margin <= x <= PITCH_LENGTH_M + margin
        and -margin <= y <= PITCH_WIDTH_M + margin
    )


def extract_calibration_frames(
    video_path: Path,
    frame_indices: list[int],
    output_dir: Path,
) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(frame_indices)
    result: dict[int, Path] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")

    i = 0
    last = max(frame_indices) if frame_indices else -1
    try:
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                path = output_dir / f"frame_{i:06d}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not write calibration frame: {path}")
                result[i] = path
            i += 1
    finally:
        cap.release()

    missing = sorted(wanted - set(result))
    if missing:
        raise RuntimeError(f"Could not extract calibration frames: {missing[:10]}")
    return result


def select_calibration_samples(
    tracked_frames: list[int],
    stride: int,
) -> list[int]:
    if not tracked_frames:
        return []
    stride = max(1, stride)
    start, end = tracked_frames[0], tracked_frames[-1]
    samples = list(range(start, end + 1, stride))
    if end not in samples:
        samples.append(end)
    return sorted(set(samples))


def build_calibration_map(
    source: Path,
    pnl_root: Path,
    tracked_frames: list[int],
    output_dir: Path,
    stride: int,
    device: str,
    min_quality: float,
) -> tuple[dict[int, dict], list[dict]]:
    sample_indices = select_calibration_samples(tracked_frames, stride)
    if not sample_indices:
        return {}, []

    print(
        f"[V2.3] Extracting {len(sample_indices)} PnL calibration samples "
        f"(stride={max(1, stride)})..."
    )
    frame_paths = extract_calibration_frames(source, sample_indices, output_dir)

    adapter = PnLCalibAdapter(
        pnl_root=pnl_root,
        device=device,
        pnl_refine=True,
    )
    ready, message = adapter.health_check()
    print(message)
    if not ready:
        raise RuntimeError(message)

    print("[V2.3] Running PnLCalib. Models are loaded once for all sample frames...")
    started = time.perf_counter()
    results = adapter.calibrate_many_with_metrics(
        [frame_paths[i] for i in sample_indices]
    )
    elapsed = time.perf_counter() - started

    calibration_map: dict[int, dict] = {}
    serializable: list[dict] = []

    for frame_index, item in zip(sample_indices, results):
        record = dict(item)
        record["frame_index"] = frame_index
        quality = float(record.get("quality_score", 0.0) or 0.0)
        accepted = record.get("status") == "ok" and quality >= min_quality
        record["accepted_for_v23"] = bool(accepted)
        serializable.append(record)
        if accepted:
            calibration_map[frame_index] = record

    print(
        f"[V2.3] PnLCalib accepted {len(calibration_map)}/{len(sample_indices)} "
        f"samples in {elapsed:.2f}s"
    )
    return calibration_map, serializable


def nearest_calibration(
    frame_index: int,
    calibration_map: dict[int, dict],
    calibration_keys: list[int],
    max_gap: int,
) -> tuple[Optional[dict], Optional[int]]:
    if not calibration_keys:
        return None, None

    pos = bisect.bisect_left(calibration_keys, frame_index)
    candidates = []
    if pos < len(calibration_keys):
        candidates.append(calibration_keys[pos])
    if pos > 0:
        candidates.append(calibration_keys[pos - 1])

    if not candidates:
        return None, None

    best = min(candidates, key=lambda idx: abs(idx - frame_index))
    if abs(best - frame_index) > max(0, max_gap):
        return None, None
    return calibration_map[best], best


def build_identity_geometry_records(
    tracking: dict[int, dict],
    tracked_frames: list[int],
    calibration_map: dict[int, dict],
    pitch_margin: float,
) -> dict[int, list[dict]]:
    """Project tracked foot points on exact successful PnL frames for duplicate-ID discovery."""
    frame_records: dict[int, list[dict]] = {}
    for frame_index in tracked_frames:
        calib = calibration_map.get(frame_index)
        if calib is None:
            continue
        H = np.asarray(calib["homography_image_to_pitch"], dtype=np.float64)
        records: list[dict] = []
        for tr in tracking.get(frame_index, {"tracks": []}).get("tracks", []):
            foot = tr.get("foot_point")
            bbox = tr.get("bbox_xyxy")
            if not foot or len(foot) < 2 or not bbox or len(bbox) != 4:
                continue
            xy = transform_point(H, float(foot[0]), float(foot[1]))
            if xy is None or not inside_pitch(xy, pitch_margin):
                continue
            rec = dict(tr)
            rec["pitch_xy"] = [float(xy[0]), float(xy[1])]
            records.append(rec)
        if records:
            frame_records[frame_index] = records
    return frame_records


def make_observation(
    track: dict,
    pitch_xy: Optional[tuple[float, float]],
) -> Optional[DetectionObservation]:
    try:
        track_id = int(track.get("track_id", -1))
        bbox = track.get("bbox_xyxy")
        if track_id < 0 or not bbox or len(bbox) != 4:
            return None

        class_name = str(track.get("class_name", "player")).lower()
        role_hint = ROLE_HINTS.get(class_name, PLAYER)

        return DetectionObservation(
            track_id=track_id,
            bbox_xyxy=tuple(float(v) for v in bbox),
            confidence=float(track.get("confidence", 1.0)),
            pitch_xy=pitch_xy,
            role_hint=role_hint,
        )
    except Exception:
        return None


def assignment_to_dict(a: TeamAssignment) -> dict:
    return {
        "track_id": int(a.track_id),
        "team": a.team,
        "role": a.role,
        "confidence": round(float(a.confidence), 5),
        "raw_team": a.raw_team,
        "raw_confidence": round(float(a.raw_confidence), 5),
        "id_switch_suspected": bool(a.id_switch_suspected),
        "reason": a.reason,
    }


def draw_assignment(
    frame: np.ndarray,
    bbox,
    assignment: TeamAssignment,
    pitch_xy: Optional[tuple[float, float]],
):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))

    color = ROLE_COLORS.get(
        assignment.role,
        TEAM_COLORS.get(assignment.team, TEAM_COLORS[UNKNOWN]),
    )
    thickness = 3 if assignment.id_switch_suspected else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if assignment.role == REFEREE:
        role_text = "REF"
    elif assignment.role == GOALKEEPER:
        role_text = f"GK/{assignment.team}"
    else:
        role_text = assignment.team

    coord = ""
    if pitch_xy is not None:
        coord = f" | {pitch_xy[0]:.1f},{pitch_xy[1]:.1f}m"

    text = f"ID {assignment.track_id} | {role_text} | {assignment.confidence:.2f}{coord}"
    if assignment.id_switch_suspected:
        text += " | SWITCH?"

    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    label_y = max(th + base + 4, y1)
    cv2.rectangle(
        frame,
        (x1, label_y - th - base - 5),
        (min(w - 1, x1 + tw + 6), label_y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 3, label_y - base - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )


def draw_outside(frame: np.ndarray, track: dict, pitch_xy: Optional[tuple[float, float]]):
    bbox = track.get("bbox_xyxy")
    if not bbox:
        return
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), OUTSIDE_COLOR, 1, cv2.LINE_AA)
    tid = int(track.get("track_id", -1))
    coord = (
        f"{pitch_xy[0]:.1f},{pitch_xy[1]:.1f}m"
        if pitch_xy is not None
        else "NO_PNL"
    )
    cv2.putText(
        frame,
        f"ID {tid} | OUTSIDE | {coord}",
        (x1, max(15, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        OUTSIDE_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_mini_pitch(
    frame: np.ndarray,
    points: list[tuple[float, float, str, str, int]],
):
    # Compact radar for spatial-role validation, not the final Radar v4 renderer.
    panel_w = min(340, max(240, frame.shape[1] // 4))
    panel_h = int(round(panel_w * PITCH_WIDTH_M / PITCH_LENGTH_M))
    margin = 12
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (39, 105, 46)

    def p(x, y):
        px = margin + int(round((x / PITCH_LENGTH_M) * (panel_w - 2 * margin)))
        py = margin + int(round((y / PITCH_WIDTH_M) * (panel_h - 2 * margin)))
        return px, py

    white = (230, 230, 230)
    cv2.rectangle(panel, p(0, 0), p(105, 68), white, 1, cv2.LINE_AA)
    cv2.line(panel, p(52.5, 0), p(52.5, 68), white, 1, cv2.LINE_AA)
    cx, cy = p(52.5, 34)
    radius = max(3, int(round(9.15 / 105 * (panel_w - 2 * margin))))
    cv2.circle(panel, (cx, cy), radius, white, 1, cv2.LINE_AA)

    for x, y, team, role, tid in points:
        if not (0 <= x <= 105 and 0 <= y <= 68):
            continue
        color = ROLE_COLORS.get(role, TEAM_COLORS.get(team, TEAM_COLORS[UNKNOWN]))
        cv2.circle(panel, p(x, y), 5, color, -1, cv2.LINE_AA)

    x0 = frame.shape[1] - panel_w - 8
    y0 = frame.shape[0] - panel_h - 8
    if x0 >= 0 and y0 >= 0:
        roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
        cv2.addWeighted(panel, 0.88, roi, 0.12, 0.0, roi)


def main():
    args = parse_args()

    source = Path(args.source)
    tracking_path = resolve_project_path(args.tracking)
    pnl_root = Path(args.pnl_root)
    output_path = resolve_project_path(args.output)
    jsonl_path = resolve_project_path(args.jsonl)
    calibration_json_path = resolve_project_path(args.calibration_json)
    calibration_frames_dir = resolve_project_path(args.calibration_frames_dir)

    for p in (source, tracking_path, pnl_root):
        if not p.exists():
            raise FileNotFoundError(p)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_json_path.parent.mkdir(parents=True, exist_ok=True)

    tracking = read_jsonl(tracking_path)
    tracked_frames = sorted(tracking)
    if args.max_frames >= 0:
        tracked_frames = [i for i in tracked_frames if i < args.max_frames]
    if not tracked_frames:
        raise RuntimeError("Tracking JSONL contains no frames in the requested range.")

    print("=" * 92)
    print("FootballAnalysisAI - Team Classifier V2.4 + PnLCalib + Canonical ID")
    print("Harness version    : 2.4-canonical-id-gk-consensus")
    print(f"Source             : {source}")
    print(f"Tracking           : {tracking_path}")
    print(f"PnLCalib           : {pnl_root}")
    print(f"Calibration stride : {args.calibration_stride}")
    print(f"Pitch margin       : {args.pitch_margin:.2f} m")
    print("=" * 92)

    calibration_map, calibration_records = build_calibration_map(
        source=source,
        pnl_root=pnl_root,
        tracked_frames=tracked_frames,
        output_dir=calibration_frames_dir,
        stride=args.calibration_stride,
        device=args.device,
        min_quality=args.min_calibration_quality,
    )
    calibration_json_path.write_text(
        json.dumps(calibration_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not calibration_map:
        raise RuntimeError(
            "PnLCalib produced no accepted calibrations. "
            "Do not continue V2.3 spatial role classification without pitch geometry."
        )

    calibration_keys = sorted(calibration_map)

    # V2.4 canonical identity prepass. This is deliberately strict: only tracks
    # that overlap in the same frames, almost perfectly overlap in image space,
    # and project to virtually the same PnL pitch location are merged.
    identity_records = build_identity_geometry_records(
        tracking=tracking,
        tracked_frames=tracked_frames,
        calibration_map=calibration_map,
        pitch_margin=max(0.0, args.pitch_margin),
    )
    alias_map, duplicate_pairs = build_canonical_alias_map(
        identity_records,
        CanonicalIdentityConfig(
            min_overlap_frames=2,
            max_median_pitch_distance_m=0.75,
            min_median_bbox_iou=0.80,
        ),
    )
    print("[V2.4] Canonical duplicate-track pairs:")
    if duplicate_pairs:
        for pair in duplicate_pairs:
            canonical = alias_map.get(pair.track_a, pair.track_a)
            print(
                f"  ID {pair.track_a} <-> ID {pair.track_b} => canonical {canonical} | "
                f"overlap={pair.overlap_frames} | pitch={pair.median_pitch_distance_m:.2f}m | "
                f"IoU={pair.median_bbox_iou:.3f}"
            )
    else:
        print("  NONE")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    config = TeamClassifierV2Config(
        embedding_stride=max(1, args.embedding_stride),
        bootstrap_min_samples=max(8, args.bootstrap_min_samples),
        use_deep_embedding=True,
        embedding_device="auto",
        pitch_role_margin_m=max(0.0, args.pitch_margin),
    )
    classifier = TeamClassifierV2(config)

    print(f"Embedding backend  : {classifier.embedding_backend}")
    print(f"PnL accepted       : {len(calibration_map)}/{len(calibration_records)}")
    print("=" * 92)

    stats = Counter()
    ref_by_track = defaultdict(int)
    gk_by_track = defaultdict(int)
    outside_by_track = defaultdict(int)
    calibration_use = Counter()

    frame_index = 0
    started = time.perf_counter()

    with jsonl_path.open("w", encoding="utf-8") as out:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index > tracked_frames[-1]:
                    break
                if args.max_frames >= 0 and frame_index >= args.max_frames:
                    break

                row = tracking.get(frame_index, {"tracks": []})
                tracks = row.get("tracks", [])

                calib, calib_frame = nearest_calibration(
                    frame_index,
                    calibration_map,
                    calibration_keys,
                    args.max_calibration_gap,
                )

                H = None
                if calib is not None:
                    H = np.asarray(
                        calib["homography_image_to_pitch"],
                        dtype=np.float64,
                    )
                    calibration_use["available"] += 1
                    if calib_frame == frame_index:
                        calibration_use["exact"] += 1
                    else:
                        calibration_use["nearest"] += 1
                else:
                    calibration_use["missing"] += 1

                observations: list[DetectionObservation] = []
                track_by_id: dict[int, dict] = {}
                pitch_by_id: dict[int, tuple[float, float]] = {}
                outside: list[tuple[dict, Optional[tuple[float, float]], str]] = []
                candidate_records: list[dict] = []

                for tr in tracks:
                    tid = int(tr.get("track_id", -1))
                    foot = tr.get("foot_point")

                    pitch_xy: Optional[tuple[float, float]] = None
                    if H is not None and foot and len(foot) >= 2:
                        pitch_xy = transform_point(
                            H,
                            float(foot[0]),
                            float(foot[1]),
                        )

                    if H is None or pitch_xy is None:
                        rec = dict(tr)
                        rec["pitch_xy"] = None
                        candidate_records.append(rec)
                        stats["no_pitch_geometry_tracks"] += 1
                        continue

                    if not inside_pitch(pitch_xy, max(0.0, args.pitch_margin)):
                        outside.append((tr, pitch_xy, "outside_pitch"))
                        outside_by_track[tid] += 1
                        stats["outside_pitch"] += 1
                        continue

                    rec = dict(tr)
                    rec["pitch_xy"] = [float(pitch_xy[0]), float(pitch_xy[1])]
                    candidate_records.append(rec)
                    stats["inside_pitch"] += 1

                canonical_records = collapse_frame_records(candidate_records, alias_map)
                stats["duplicate_raw_suppressed"] += max(
                    0, len(candidate_records) - len(canonical_records)
                )

                for tr in canonical_records:
                    pitch_value = tr.get("pitch_xy")
                    pitch_xy = (
                        (float(pitch_value[0]), float(pitch_value[1]))
                        if pitch_value is not None
                        else None
                    )
                    obs = make_observation(tr, pitch_xy)
                    if obs is None:
                        continue
                    observations.append(obs)
                    track_by_id[obs.track_id] = tr
                    if pitch_xy is not None:
                        pitch_by_id[obs.track_id] = pitch_xy

                if not classifier.is_ready:
                    classify_input = [
                        o for o in observations if o.role_hint == PLAYER
                    ]
                else:
                    classify_input = observations

                assignments = classifier.classify_frame(
                    frame,
                    classify_input,
                    frame_index,
                )
                by_id = {a.track_id: a for a in assignments}

                # During bootstrap, role hints are deliberately kept out of team
                # prototype learning. They become visible once the classifier is ready.
                frame_payload_tracks = []
                radar_points = []

                for obs in observations:
                    a = by_id.get(obs.track_id)
                    if a is None:
                        continue
                    tr = track_by_id[obs.track_id]
                    pitch_xy = pitch_by_id.get(obs.track_id)

                    draw_assignment(frame, tr["bbox_xyxy"], a, pitch_xy)

                    stats["assignments"] += 1
                    if a.team == TEAM_A:
                        stats["team_a"] += 1
                    elif a.team == TEAM_B:
                        stats["team_b"] += 1
                    else:
                        stats["unknown"] += 1
                    if a.role == REFEREE:
                        stats["referee"] += 1
                        ref_by_track[a.track_id] += 1
                    if a.role == GOALKEEPER:
                        stats["goalkeeper"] += 1
                        gk_by_track[a.track_id] += 1
                    if a.id_switch_suspected:
                        stats["id_switch_suspected"] += 1

                    if pitch_xy is not None:
                        radar_points.append(
                            (
                                pitch_xy[0],
                                pitch_xy[1],
                                a.team,
                                a.role,
                                a.track_id,
                            )
                        )

                    enriched = dict(tr)
                    enriched["pitch_xy"] = (
                        [round(pitch_xy[0], 4), round(pitch_xy[1], 4)]
                        if pitch_xy is not None
                        else None
                    )
                    enriched["spatial_status"] = (
                        "inside_pitch" if pitch_xy is not None else "no_calibration"
                    )
                    enriched["team_v24"] = assignment_to_dict(a)
                    frame_payload_tracks.append(enriched)

                for tr, pitch_xy, reason in outside:
                    draw_outside(frame, tr, pitch_xy)
                    enriched = dict(tr)
                    enriched["pitch_xy"] = (
                        [round(pitch_xy[0], 4), round(pitch_xy[1], 4)]
                        if pitch_xy is not None
                        else None
                    )
                    enriched["spatial_status"] = reason
                    enriched["team_v24"] = {
                        "track_id": int(tr.get("track_id", -1)),
                        "team": UNKNOWN,
                        "role": "OUTSIDE_PITCH",
                        "confidence": 1.0,
                        "raw_team": UNKNOWN,
                        "raw_confidence": 0.0,
                        "id_switch_suspected": False,
                        "reason": "pnl_spatial_gate",
                    }
                    frame_payload_tracks.append(enriched)

                draw_mini_pitch(frame, radar_points)

                cv2.rectangle(frame, (0, 0), (width, 92), (18, 18, 18), -1)
                cv2.putText(
                    frame,
                    (
                        f"Team Classifier V2.4 + PnL | "
                        f"{'READY' if classifier.is_ready else 'BOOTSTRAP'} | "
                        f"{classifier.embedding_backend}"
                    ),
                    (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    (
                        f"Frame {frame_index} | PnL={calib_frame if calib_frame is not None else 'NONE'} | "
                        f"inside={len(observations)} outside={len(outside)} | "
                        f"REF={stats['referee']} GK={stats['goalkeeper']}"
                    ),
                    (18, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (225, 225, 225),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Spatial gate: off-pitch tracks are excluded from team/referee/goalkeeper decisions",
                    (18, 79),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(frame)

                out.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": round(frame_index / fps, 5),
                            "pnl_calibration_frame": calib_frame,
                            "pnl_quality_score": (
                                float(calib.get("quality_score", 0.0))
                                if calib is not None
                                else None
                            ),
                            "classifier": classifier.debug_state(),
                            "tracks": frame_payload_tracks,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                frame_index += 1
                stats["frames"] += 1

                if frame_index == 1 or frame_index % 25 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"Processed {frame_index}/{tracked_frames[-1] + 1} | "
                        f"ready={classifier.is_ready} | "
                        f"A={stats['team_a']} B={stats['team_b']} "
                        f"REF={stats['referee']} GK={stats['goalkeeper']} "
                        f"outside={stats['outside_pitch']} | "
                        f"{frame_index / max(elapsed, 1e-6):.2f} FPS"
                    )
        finally:
            cap.release()
            writer.release()

    elapsed = time.perf_counter() - started

    print("=" * 92)
    print("DONE - Team Classifier V2.4 + PnL + CANONICAL ID")
    print(f"Frames processed        : {stats['frames']}")
    print(f"Assignments             : {stats['assignments']}")
    print(f"TEAM_A samples          : {stats['team_a']}")
    print(f"TEAM_B samples          : {stats['team_b']}")
    print(f"UNKNOWN samples         : {stats['unknown']}")
    print(f"Referee samples         : {stats['referee']}")
    print(f"Goalkeeper samples      : {stats['goalkeeper']}")
    print(f"Outside-pitch samples   : {stats['outside_pitch']}")
    print(f"No-geometry samples     : {stats['no_pitch_geometry_tracks']}")
    print(f"ID-switch suspicions    : {stats['id_switch_suspected']}")
    print(f"Duplicate raw suppressed: {stats['duplicate_raw_suppressed']}")
    print(f"Canonical alias count   : {sum(1 for raw, can in alias_map.items() if raw != can)}")
    print(f"Embedding backend       : {classifier.embedding_backend}")
    print(f"Classifier ready        : {classifier.is_ready}")
    print(f"Classification FPS      : {stats['frames'] / max(elapsed, 1e-6):.2f}")
    print(
        f"Calibration use         : exact={calibration_use['exact']} "
        f"nearest={calibration_use['nearest']} missing={calibration_use['missing']}"
    )

    print("Referee track summary   :")
    if ref_by_track:
        for tid, count in sorted(ref_by_track.items(), key=lambda x: -x[1]):
            state = classifier.track_states.get(tid)
            if state is not None:
                print(
                    f"  ID {tid:>3} | REF={count:>3} | "
                    f"det_ref={state.referee_hint_frames} | "
                    f"onpitch_ref={state.referee_on_pitch_hint_frames}/"
                    f"{state.on_pitch_seen_frames}"
                )
            else:
                print(f"  ID {tid:>3} | REF={count:>3}")
    else:
        print("  NONE")

    print("Goalkeeper track summary:")
    if gk_by_track:
        for tid, count in sorted(gk_by_track.items(), key=lambda x: -x[1]):
            state = classifier.track_states.get(tid)
            if state is not None:
                total = state.goalkeeper_hint_frames + state.non_goalkeeper_hint_frames
                ratio = state.goalkeeper_hint_frames / total if total else 0.0
                print(
                    f"  ID {tid:>3} | GK={count:>3} | "
                    f"det_gk={state.goalkeeper_hint_frames}/{total} "
                    f"ratio={ratio:.2f} trusted={state.goalkeeper_trusted}"
                )
            else:
                print(f"  ID {tid:>3} | GK={count:>3}")
    else:
        print("  NONE")

    print("Canonical aliases       :")
    alias_rows = [(raw, can) for raw, can in sorted(alias_map.items()) if raw != can]
    if alias_rows:
        for raw, can in alias_rows:
            print(f"  raw ID {raw:>3} -> canonical ID {can:>3}")
    else:
        print("  NONE")

    print("Top outside-pitch tracks:")
    if outside_by_track:
        for tid, count in sorted(outside_by_track.items(), key=lambda x: -x[1])[:10]:
            print(f"  ID {tid:>3} | OUTSIDE={count:>3}")
    else:
        print("  NONE")

    print(f"Video output            : {output_path}")
    print(f"JSONL output            : {jsonl_path}")
    print(f"Calibration JSON        : {calibration_json_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
