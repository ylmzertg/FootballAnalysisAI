from __future__ import annotations

import argparse
import time

import cv2
import torch
from ultralytics import YOLO

from core.runtime_paths import resolve_project_path


CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FootballAnalysisAI - player detection smoke test"
    )
    parser.add_argument(
        "--source",
        default=r"input\input.mp4",
        help="Input video path. Relative paths resolve from the project root.",
    )
    parser.add_argument(
        "--model",
        default=r"models\football-player-detection.pt",
        help="YOLO model path. Relative paths resolve from the project root.",
    )
    parser.add_argument(
        "--output",
        default=r"output\player_detection.mp4",
        help="Output video path. Relative paths resolve from the project root.",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum frames for first smoke test. Use -1 for full video.",
    )
    return parser.parse_args()


def color_for_class(class_id: int):
    colors = {
        0: (0, 255, 255),
        1: (255, 0, 255),
        2: (0, 255, 0),
        3: (0, 165, 255),
    }
    return colors.get(class_id, (255, 255, 255))


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    model_path = resolve_project_path(args.model)
    output = resolve_project_path(args.output)

    if not source.exists():
        raise FileNotFoundError(
            f"Input video not found: {source}\n"
            "Copy a video to input\\input.mp4 or pass --source <path>."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Copy football-player-detection.pt to models\\ or pass --model <path>."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"

    print("=" * 72)
    print("FootballAnalysisAI - Player Detection")
    print(f"Source : {source}")
    print(f"Model  : {model_path}")
    print(f"Output : {output}")
    print(f"Torch  : {torch.__version__}")
    print(f"CUDA   : {use_cuda}")
    if use_cuda:
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"imgsz  : {args.imgsz}")
    print("=" * 72)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output}")

    frame_index = 0
    detections_total = 0
    started = time.perf_counter()

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

            frame_detection_count = 0

            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].detach().cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    confidence = float(box.conf[0].detach().cpu().item())
                    class_id = int(box.cls[0].detach().cpu().item())

                    label = CLASS_NAMES.get(
                        class_id,
                        result.names.get(class_id, str(class_id))
                        if isinstance(result.names, dict)
                        else str(class_id),
                    )
                    color = color_for_class(class_id)

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA
                    )
                    text = f"{label} {confidence:.2f}"
                    (tw, th), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                    )
                    ty = max(y1, th + baseline + 4)
                    cv2.rectangle(
                        frame,
                        (x1, ty - th - baseline - 4),
                        (x1 + tw + 6, ty),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        text,
                        (x1 + 3, ty - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    frame_detection_count += 1

            detections_total += frame_detection_count
            cv2.putText(
                frame,
                f"Frame: {frame_index + 1} | Detections: {frame_detection_count}",
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
                print(
                    f"Processed {frame_index}/{total_frames if total_frames > 0 else '?'} frames"
                    f" | detections={detections_total}"
                    f" | {frame_index / max(elapsed, 1e-6):.2f} FPS"
                )
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - started
    print("=" * 72)
    print("DONE")
    print(f"Frames processed : {frame_index}")
    print(f"Total detections : {detections_total}")
    print(f"Elapsed          : {elapsed:.1f} s")
    print(f"Average FPS      : {frame_index / max(elapsed, 1e-6):.2f}")
    print(f"Output           : {output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
