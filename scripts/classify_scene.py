from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="FootballAnalysisAI - scene classification"
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
        default=r"output\scene_classification.mp4",
    )
    parser.add_argument(
        "--jsonl",
        default=r"output\scene_labels.jsonl",
    )
    parser.add_argument(
        "--min-wide-keypoints",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--min-wide-tracks",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--closeup-height-ratio",
        type=float,
        default=0.34,
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows[int(item["frame_index"])] = item
    return rows


def bbox_height_ratios(tracks: list[dict], frame_h: int) -> list[float]:
    result = []
    for tr in tracks:
        bbox = tr.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        _, y1, _, y2 = map(float, bbox)
        h = max(0.0, y2 - y1)
        result.append(h / max(frame_h, 1))
    return result


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

    # Strong wide-field signal: enough pitch geometry + many players visible.
    if kp_count >= min_wide_keypoints and track_count >= min_wide_tracks:
        label = "WIDE_FIELD"

    # Strong close-up signal: little/no pitch geometry and a very large subject.
    elif kp_count <= 1 and (
        max_h >= closeup_height_ratio or
        (track_count <= 4 and median_h >= closeup_height_ratio * 0.65)
    ):
        label = "CLOSE_UP"

    # Partial field / transitional football camera.
    elif kp_count >= 2 and track_count >= 4:
        label = "FIELD_PARTIAL"

    else:
        label = "OTHER"

    metrics = {
        "pitch_keypoints": kp_count,
        "track_count": track_count,
        "median_bbox_height_ratio": round(median_h, 4),
        "max_bbox_height_ratio": round(max_h, 4),
    }

    return label, metrics


def majority_label(history: deque[str]) -> str:
    if not history:
        return "OTHER"

    counts: dict[str, int] = {}
    for label in history:
        counts[label] = counts.get(label, 0) + 1

    # Prefer stable football-view classes in ties.
    priority = {
        "WIDE_FIELD": 4,
        "FIELD_PARTIAL": 3,
        "CLOSE_UP": 2,
        "OTHER": 1,
    }

    return max(
        counts,
        key=lambda k: (counts[k], priority.get(k, 0))
    )


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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError(f"Could not create output video: {output_path}")

    history: deque[str] = deque(
        maxlen=max(1, args.smoothing_window)
    )

    counts = {
        "WIDE_FIELD": 0,
        "FIELD_PARTIAL": 0,
        "CLOSE_UP": 0,
        "OTHER": 0,
    }

    frame_index = 0

    print("=" * 78)
    print("FootballAnalysisAI - Scene Classification")
    print(f"Source   : {source}")
    print(f"Tracking : {tracking_path}")
    print(f"Pitch    : {pitch_path}")
    print(f"Video    : {output_path}")
    print(f"JSONL    : {jsonl_path}")
    print("=" * 78)

    with jsonl_path.open("w", encoding="utf-8") as out:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                tracks = tracking_by_frame.get(
                    frame_index, {"tracks": []}
                ).get("tracks", [])

                keypoints = pitch_by_frame.get(
                    frame_index, {"keypoints": []}
                ).get("keypoints", [])

                raw_label, metrics = raw_classify(
                    kp_count=len(keypoints),
                    tracks=tracks,
                    frame_h=height,
                    min_wide_keypoints=args.min_wide_keypoints,
                    min_wide_tracks=args.min_wide_tracks,
                    closeup_height_ratio=args.closeup_height_ratio,
                )

                history.append(raw_label)
                label = majority_label(history)

                counts[label] = counts.get(label, 0) + 1

                payload = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 5),
                    "raw_label": raw_label,
                    "label": label,
                    "analysis_enabled": label == "WIDE_FIELD",
                    "metrics": metrics,
                }

                out.write(
                    json.dumps(payload, ensure_ascii=False) + "\n"
                )

                color = color_for_label(label)

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, 86),
                    (20, 20, 20),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Scene: {label}",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
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
                        f"analysis={'ON' if label == 'WIDE_FIELD' else 'OFF'}"
                    ),
                    (18, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(frame)

                frame_index += 1

                if frame_index == 1 or frame_index % 25 == 0:
                    print(
                        f"Processed {frame_index}/{total_frames} | "
                        f"wide={counts['WIDE_FIELD']} | "
                        f"partial={counts['FIELD_PARTIAL']} | "
                        f"close={counts['CLOSE_UP']} | "
                        f"other={counts['OTHER']}"
                    )

        finally:
            cap.release()
            writer.release()

    print("=" * 78)
    print("DONE")
    print(f"Frames processed : {frame_index}")
    for key in ("WIDE_FIELD", "FIELD_PARTIAL", "CLOSE_UP", "OTHER"):
        print(f"{key:15}: {counts.get(key, 0)}")
    print(f"Video output     : {output_path.resolve()}")
    print(f"Scene JSONL      : {jsonl_path.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
