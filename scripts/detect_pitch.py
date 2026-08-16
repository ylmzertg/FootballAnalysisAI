from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="FootballAnalysisAI - pitch keypoint detection"
    )
    parser.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    parser.add_argument(
        "--model",
        default=r"models\football-pitch-detection.pt",
    )
    parser.add_argument(
        "--output",
        default=r"output\pitch_keypoints.mp4",
    )
    parser.add_argument(
        "--jsonl",
        default=r"output\pitch_keypoints.jsonl",
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
        help="Object confidence threshold passed to Ultralytics",
    )
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.20,
        help="Minimum keypoint confidence to draw/save",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    source = Path(args.source)
    model_path = Path(args.model)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl)

    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    if not model_path.exists():
        raise FileNotFoundError(f"Pitch model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"

    model = YOLO(str(model_path))

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

    print("=" * 76)
    print("FootballAnalysisAI - Pitch Keypoint Detection")
    print(f"Source : {source}")
    print(f"Model  : {model_path}")
    print(f"Output : {output_path}")
    print(f"JSONL  : {jsonl_path}")
    print(f"Torch  : {torch.__version__}")
    print(f"CUDA   : {use_cuda}")
    if use_cuda:
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"imgsz  : {args.imgsz}")
    print("=" * 76)

    frame_index = 0
    frames_with_keypoints = 0
    total_visible_keypoints = 0
    started = time.perf_counter()

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

                visible = []

                if result.keypoints is not None and result.keypoints.xy is not None:
                    xy_tensor = result.keypoints.xy

                    if len(xy_tensor) > 0:
                        xy = xy_tensor[0].detach().cpu().numpy()

                        conf = None
                        if result.keypoints.conf is not None:
                            conf = (
                                result.keypoints.conf[0]
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        for kp_id, point in enumerate(xy):
                            x = float(point[0])
                            y = float(point[1])

                            kp_conf = (
                                float(conf[kp_id])
                                if conf is not None and kp_id < len(conf)
                                else 1.0
                            )

                            if x <= 1 or y <= 1 or kp_conf < args.kp_conf:
                                continue

                            visible.append(
                                {
                                    "keypoint_id": int(kp_id),
                                    "x": round(x, 2),
                                    "y": round(y, 2),
                                    "confidence": round(kp_conf, 5),
                                }
                            )

                            center = (
                                int(round(x)),
                                int(round(y)),
                            )

                            cv2.circle(
                                frame,
                                center,
                                6,
                                (0, 0, 255),
                                -1,
                                cv2.LINE_AA,
                            )

                            cv2.putText(
                                frame,
                                str(kp_id),
                                (center[0] + 7, center[1] - 7),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.50,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                if visible:
                    frames_with_keypoints += 1
                    total_visible_keypoints += len(visible)

                payload = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 5),
                    "keypoints": visible,
                }

                jsonl_file.write(
                    json.dumps(payload, ensure_ascii=False) + "\n"
                )

                cv2.putText(
                    frame,
                    (
                        f"Frame: {frame_index + 1} | "
                        f"Pitch keypoints: {len(visible)}"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
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
                        f" | frames_with_kp={frames_with_keypoints}"
                        f" | visible_kp={total_visible_keypoints}"
                        f" | {effective_fps:.2f} FPS"
                    )

        finally:
            cap.release()
            writer.release()

    elapsed = time.perf_counter() - started

    print("=" * 76)
    print("DONE")
    print(f"Frames processed        : {frame_index}")
    print(f"Frames with keypoints   : {frames_with_keypoints}")
    print(f"Visible keypoints total : {total_visible_keypoints}")
    print(f"Elapsed                 : {elapsed:.1f} s")
    print(
        f"Average FPS             : "
        f"{frame_index / max(elapsed, 1e-6):.2f}"
    )
    print(f"Video output            : {output_path.resolve()}")
    print(f"Keypoint JSONL          : {jsonl_path.resolve()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
