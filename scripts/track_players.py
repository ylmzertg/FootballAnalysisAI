from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import supervision as sv
import torch
from ultralytics import YOLO


CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee",
}

TRACKED_CLASS_IDS = {1, 2, 3}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FootballAnalysisAI - player tracking with ByteTrack"
    )
    parser.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    parser.add_argument(
        "--model",
        default=r"models\football-player-detection.pt",
    )
    parser.add_argument(
        "--output",
        default=r"output\player_tracking.mp4",
    )
    parser.add_argument(
        "--jsonl",
        default=r"output\player_tracking.jsonl",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--minimum-consecutive-frames",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--lost-track-buffer",
        type=int,
        default=30,
    )
    return parser.parse_args()


def color_for_track(track_id: int):
    # Deterministic BGR pseudo-color without external palettes.
    return (
        int((37 * track_id) % 200 + 55),
        int((97 * track_id) % 200 + 55),
        int((157 * track_id) % 200 + 55),
    )


def main():
    args = parse_args()

    source = Path(args.source)
    model_path = Path(args.model)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl)

    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO(str(model_path))

    tracker = sv.ByteTrack(
        track_activation_threshold=args.conf,
        lost_track_buffer=args.lost_track_buffer,
        frame_rate=fps,
        minimum_consecutive_frames=args.minimum_consecutive_frames,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    print("=" * 76)
    print("FootballAnalysisAI - Player Tracking")
    print(f"Source : {source}")
    print(f"Model  : {model_path}")
    print(f"Output : {output_path}")
    print(f"JSONL  : {jsonl_path}")
    print(f"Torch  : {torch.__version__}")
    print(f"CUDA   : {use_cuda}")
    if use_cuda:
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"FPS    : {fps:.2f}")
    print(f"imgsz  : {args.imgsz}")
    print("=" * 76)

    started = time.perf_counter()
    frame_index = 0
    total_tracked_detections = 0
    unique_ids: set[int] = set()

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if args.max_frames >= 0 and frame_index >= args.max_frames:
                    break

                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=device,
                    verbose=False,
                    batch=1,
                )[0]

                detections = sv.Detections.from_ultralytics(result)

                if len(detections) > 0 and detections.class_id is not None:
                    mask = [
                        int(class_id) in TRACKED_CLASS_IDS
                        for class_id in detections.class_id
                    ]
                    detections = detections[mask]

                tracked = tracker.update_with_detections(detections)

                records = []

                if len(tracked) > 0:
                    for i in range(len(tracked)):
                        x1, y1, x2, y2 = map(
                            float, tracked.xyxy[i].tolist()
                        )

                        confidence = (
                            float(tracked.confidence[i])
                            if tracked.confidence is not None
                            else 0.0
                        )

                        class_id = (
                            int(tracked.class_id[i])
                            if tracked.class_id is not None
                            else -1
                        )

                        track_id = (
                            int(tracked.tracker_id[i])
                            if tracked.tracker_id is not None
                            else -1
                        )

                        if track_id < 0:
                            continue

                        unique_ids.add(track_id)
                        total_tracked_detections += 1

                        label = CLASS_NAMES.get(
                            class_id,
                            str(class_id),
                        )

                        foot_x = (x1 + x2) / 2.0
                        foot_y = y2

                        records.append(
                            {
                                "track_id": track_id,
                                "class_id": class_id,
                                "class_name": label,
                                "confidence": round(confidence, 5),
                                "bbox_xyxy": [
                                    round(x1, 2),
                                    round(y1, 2),
                                    round(x2, 2),
                                    round(y2, 2),
                                ],
                                "foot_point": [
                                    round(foot_x, 2),
                                    round(foot_y, 2),
                                ],
                            }
                        )

                        color = color_for_track(track_id)

                        p1 = (int(round(x1)), int(round(y1)))
                        p2 = (int(round(x2)), int(round(y2)))

                        cv2.rectangle(
                            frame,
                            p1,
                            p2,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                        text = f"ID {track_id} | {label} {confidence:.2f}"
                        (tw, th), baseline = cv2.getTextSize(
                            text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            1,
                        )

                        top_y = max(p1[1], th + baseline + 5)

                        cv2.rectangle(
                            frame,
                            (p1[0], top_y - th - baseline - 5),
                            (p1[0] + tw + 6, top_y),
                            color,
                            -1,
                        )

                        cv2.putText(
                            frame,
                            text,
                            (p1[0] + 3, top_y - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

                        cv2.circle(
                            frame,
                            (
                                int(round(foot_x)),
                                int(round(foot_y)),
                            ),
                            4,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )

                frame_payload = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 5),
                    "tracks": records,
                }

                jsonl_file.write(
                    json.dumps(frame_payload, ensure_ascii=False) + "\n"
                )

                cv2.putText(
                    frame,
                    (
                        f"Frame: {frame_index + 1} | "
                        f"Tracked: {len(records)} | "
                        f"Unique IDs: {len(unique_ids)}"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(frame)

                frame_index += 1

                if frame_index == 1 or frame_index % 25 == 0:
                    elapsed = time.perf_counter() - started
                    effective_fps = frame_index / max(elapsed, 1e-6)
                    print(
                        f"Processed {frame_index}"
                        f"/{total_frames if total_frames > 0 else '?'}"
                        f" | tracked={total_tracked_detections}"
                        f" | unique_ids={len(unique_ids)}"
                        f" | {effective_fps:.2f} FPS"
                    )

        finally:
            cap.release()
            writer.release()

    elapsed = time.perf_counter() - started

    print("=" * 76)
    print("DONE")
    print(f"Frames processed    : {frame_index}")
    print(f"Tracked detections  : {total_tracked_detections}")
    print(f"Unique track IDs    : {len(unique_ids)}")
    print(f"Elapsed             : {elapsed:.1f} s")
    print(
        f"Average FPS         : "
        f"{frame_index / max(elapsed, 1e-6):.2f}"
    )
    print(f"Video output        : {output_path.resolve()}")
    print(f"Tracking JSONL      : {jsonl_path.resolve()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
