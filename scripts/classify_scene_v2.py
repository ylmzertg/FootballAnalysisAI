from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


LABEL_PRIORITY = {
    "WIDE_FIELD": 4,
    "FIELD_PARTIAL": 3,
    "CLOSE_UP": 2,
    "OTHER": 1,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - bidirectional scene classification"
    )
    p.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument("--tracking", default=r"output\player_tracking.jsonl")
    p.add_argument("--pitch", default=r"output\pitch_keypoints.jsonl")
    p.add_argument("--output", default=r"output\scene_classification_v2.mp4")
    p.add_argument("--jsonl", default=r"output\scene_labels_v2.jsonl")
    p.add_argument("--min-wide-keypoints", type=int, default=4)
    p.add_argument("--min-wide-tracks", type=int, default=6)
    p.add_argument("--closeup-height-ratio", type=float, default=0.34)
    p.add_argument(
        "--window",
        type=int,
        default=7,
        help="Centered temporal smoothing window. Odd numbers recommended.",
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


def bbox_height_ratios(tracks: list[dict], frame_h: int) -> list[float]:
    ratios = []
    for tr in tracks:
        bbox = tr.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        _, y1, _, y2 = map(float, bbox)
        ratios.append(max(0.0, y2 - y1) / max(frame_h, 1))
    return ratios


def raw_classify(
    kp_count: int,
    tracks: list[dict],
    frame_h: int,
    min_wide_keypoints: int,
    min_wide_tracks: int,
    closeup_height_ratio: float,
) -> tuple[str, dict]:
    track_count = len(tracks)
    ratios = bbox_height_ratios(tracks, frame_h)

    median_h = float(np.median(ratios)) if ratios else 0.0
    max_h = float(np.max(ratios)) if ratios else 0.0

    # Strong tactical-camera evidence.
    if kp_count >= min_wide_keypoints and track_count >= min_wide_tracks:
        label = "WIDE_FIELD"

    # Some pitch visible, but camera may be tighter / transition shot.
    elif kp_count >= 3 and track_count >= 4:
        label = "FIELD_PARTIAL"

    # Little pitch geometry + large human subject => close-up.
    elif kp_count <= 1 and (
        max_h >= closeup_height_ratio
        or (track_count <= 4 and median_h >= closeup_height_ratio * 0.65)
    ):
        label = "CLOSE_UP"

    else:
        label = "OTHER"

    return label, {
        "pitch_keypoints": kp_count,
        "track_count": track_count,
        "median_bbox_height_ratio": round(median_h, 4),
        "max_bbox_height_ratio": round(max_h, 4),
    }


def centered_smooth(raw_labels: list[str], window: int) -> list[str]:
    if not raw_labels:
        return []

    window = max(1, int(window))
    if window % 2 == 0:
        window += 1

    radius = window // 2
    smoothed = []

    for i in range(len(raw_labels)):
        start = max(0, i - radius)
        stop = min(len(raw_labels), i + radius + 1)
        chunk = raw_labels[start:stop]
        counts = Counter(chunk)

        winner = max(
            counts,
            key=lambda label: (
                counts[label],
                LABEL_PRIORITY.get(label, 0),
            ),
        )
        smoothed.append(winner)

    return smoothed


def color_for_label(label: str):
    return {
        "WIDE_FIELD": (0, 220, 0),
        "FIELD_PARTIAL": (0, 210, 255),
        "CLOSE_UP": (0, 0, 255),
        "OTHER": (180, 180, 180),
    }.get(label, (255, 255, 255))


def main():
    args = parse_args()

    source = Path(args.source)
    tracking_path = Path(args.tracking)
    pitch_path = Path(args.pitch)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl)

    for p in (source, tracking_path, pitch_path):
        if not p.exists():
            raise FileNotFoundError(p)

    tracking = read_jsonl(tracking_path)
    pitch = read_jsonl(pitch_path)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(source)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Pass 1: classify every frame independently.
    raw_labels = []
    metrics_by_frame = []

    for frame_index in range(total_frames):
        tracks = tracking.get(
            frame_index, {"tracks": []}
        ).get("tracks", [])

        keypoints = pitch.get(
            frame_index, {"keypoints": []}
        ).get("keypoints", [])

        label, metrics = raw_classify(
            kp_count=len(keypoints),
            tracks=tracks,
            frame_h=height,
            min_wide_keypoints=args.min_wide_keypoints,
            min_wide_tracks=args.min_wide_tracks,
            closeup_height_ratio=args.closeup_height_ratio,
        )

        raw_labels.append(label)
        metrics_by_frame.append(metrics)

    # Pass 2: centered smoothing uses both past and future frames.
    labels = centered_smooth(raw_labels, args.window)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter(labels)

    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    with jsonl_path.open("w", encoding="utf-8") as out:
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw_label = raw_labels[frame_index]
            label = labels[frame_index]
            metrics = metrics_by_frame[frame_index]

            # Conservative: tactical analysis only on true wide view.
            analysis_enabled = label == "WIDE_FIELD"

            payload = {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / fps, 5),
                "raw_label": raw_label,
                "label": label,
                "analysis_enabled": analysis_enabled,
                "metrics": metrics,
            }

            out.write(json.dumps(payload, ensure_ascii=False) + "\n")

            color = color_for_label(label)

            cv2.rectangle(frame, (0, 0), (width, 88), (18, 18, 18), -1)

            cv2.putText(
                frame,
                f"Scene: {label}  (raw: {raw_label})",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"KP={metrics['pitch_keypoints']} | "
                    f"tracks={metrics['track_count']} | "
                    f"bbox_med={metrics['median_bbox_height_ratio']:.2f} | "
                    f"analysis={'ON' if analysis_enabled else 'OFF'}"
                ),
                (18, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_index += 1

    cap.release()
    writer.release()

    print("=" * 76)
    print("DONE")
    print(f"Frames processed : {total_frames}")
    for key in ("WIDE_FIELD", "FIELD_PARTIAL", "CLOSE_UP", "OTHER"):
        print(f"{key:15}: {counts.get(key, 0)}")
    print(f"Video output     : {output_path.resolve()}")
    print(f"Scene JSONL      : {jsonl_path.resolve()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
