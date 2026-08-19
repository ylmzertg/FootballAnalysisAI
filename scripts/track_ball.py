
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.ball_tracker import BallCandidate, BallTemporalTracker, BallTrackerConfig
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(description="FootballAnalysisAI - Ball Tracking v1")
    p.add_argument("--source", default=r"input\input.mp4")
    p.add_argument("--model", default=r"models\football-ball-detection.pt")
    p.add_argument("--output", default=r"output\ball_tracking.mp4")
    p.add_argument("--jsonl", default=r"output\ball_tracking.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--slice-size", type=int, default=640)
    p.add_argument("--slice-overlap", type=float, default=0.20)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--nms-iou", type=float, default=0.10)
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--max-jump-px", type=float, default=260.0)
    return p.parse_args()


def resolve_device(value: str):
    value = value.strip().lower()
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if value in {"cuda", "cuda:0"}:
        return 0
    return value


def starts(length: int, size: int, overlap: float):
    if length <= size:
        return [0]
    overlap = min(max(overlap, 0.0), 0.8)
    step = max(1, int(round(size * (1 - overlap))))
    xs = list(range(0, max(1, length - size + 1), step))
    end = max(0, length - size)
    if xs[-1] != end:
        xs.append(end)
    return sorted(set(xs))


def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2-x1) * np.maximum(0.0, y2-y1)
    a = max(0.0, box[2]-box[0]) * max(0.0, box[3]-box[1])
    b = np.maximum(0.0, boxes[:,2]-boxes[:,0]) * np.maximum(0.0, boxes[:,3]-boxes[:,1])
    return inter / np.maximum(a+b-inter, 1e-9)


def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou_one_to_many(boxes[i], boxes[rest]) <= threshold]
    return keep


def detect_candidates(frame, model, device, slice_size, overlap, imgsz, conf, nms_iou):
    h, w = frame.shape[:2]
    raw_boxes, raw_scores = [], []
    for y0 in starts(h, slice_size, overlap):
        for x0 in starts(w, slice_size, overlap):
            crop = frame[y0:min(h,y0+slice_size), x0:min(w,x0+slice_size)]
            result = model.predict(
                source=crop, imgsz=imgsz, conf=conf,
                device=device, verbose=False, batch=1
            )[0]
            if result.boxes is None:
                continue
            names = result.names if isinstance(result.names, dict) else {}
            for box in result.boxes:
                cid = int(box.cls[0].detach().cpu().item())
                cname = str(names.get(cid, cid)).lower()
                if "ball" not in cname and cid != 0:
                    continue
                x1,y1,x2,y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                score = float(box.conf[0].detach().cpu().item())
                raw_boxes.append([x1+x0, y1+y0, x2+x0, y2+y0])
                raw_scores.append(score)
    if not raw_boxes:
        return []
    boxes = np.asarray(raw_boxes, dtype=float)
    scores = np.asarray(raw_scores, dtype=float)
    keep = nms(boxes, scores, nms_iou)
    out = []
    for i in keep:
        x1,y1,x2,y2 = boxes[i].tolist()
        out.append(BallCandidate(
            bbox_xyxy=(x1,y1,x2,y2),
            center_xy=((x1+x2)/2, (y1+y2)/2),
            confidence=float(scores[i])
        ))
    return out


def main():
    args = parse_args()
    source = resolve_project_path(args.source)
    model_path = resolve_project_path(args.model)
    output = resolve_project_path(args.output)
    jsonl_path = resolve_project_path(args.jsonl)
    if not source.exists():
        raise FileNotFoundError(source)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)
    )

    tracker = BallTemporalTracker(BallTrackerConfig(
        max_gap_frames=max(0,args.max_gap),
        max_jump_px=max(1.0,args.max_jump_px),
    ))

    print("="*76)
    print("FootballAnalysisAI - Ball Tracking v1")
    print(f"Source : {source}")
    print(f"Model  : {model_path}")
    print(f"Device : {device}")
    print("="*76)

    frame_index = detected = predicted = missing = total_candidates = 0
    started = time.perf_counter()

    with jsonl_path.open("w", encoding="utf-8") as out:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.max_frames >= 0 and frame_index >= args.max_frames:
                    break

                candidates = detect_candidates(
                    frame, model, device,
                    max(160,args.slice_size), args.slice_overlap,
                    args.imgsz, args.conf, args.nms_iou
                )
                total_candidates += len(candidates)
                result = tracker.update(candidates, frame_index)

                if result.detected:
                    detected += 1
                elif result.predicted:
                    predicted += 1
                else:
                    missing += 1

                annotated = frame.copy()
                for c in candidates:
                    x1,y1,x2,y2 = map(int, map(round, c.bbox_xyxy))
                    cv2.rectangle(annotated, (x1,y1),(x2,y2),(110,110,110),1)

                if result.center_xy is not None:
                    cx,cy = map(int, map(round, result.center_xy))
                    color = (0,255,0) if result.detected else (0,215,255)
                    label = f"BALL {result.confidence:.2f}" if result.detected else f"BALL PRED {result.gap_frames}"
                    cv2.circle(annotated,(cx,cy),10,color,2,cv2.LINE_AA)
                    cv2.putText(annotated,label,(cx+12,max(18,cy-8)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.48,color,1,cv2.LINE_AA)

                writer.write(annotated)

                out.write(json.dumps({
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index/fps,5),
                    "candidate_count": len(candidates),
                    "detected": bool(result.detected),
                    "predicted": bool(result.predicted),
                    "gap_frames": int(result.gap_frames),
                    "confidence": round(float(result.confidence),6),
                    "center_xy": (
                        [round(result.center_xy[0],3), round(result.center_xy[1],3)]
                        if result.center_xy is not None else None
                    ),
                    "bbox_xyxy": (
                        [round(v,3) for v in result.bbox_xyxy]
                        if result.bbox_xyxy is not None else None
                    ),
                }, ensure_ascii=False) + "\n")

                frame_index += 1
                if frame_index == 1 or frame_index % 10 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"Processed {frame_index}/{total if total>0 else '?'}"
                        f" | detected={detected}"
                        f" predicted={predicted}"
                        f" missing={missing}"
                        f" candidates={total_candidates}"
                        f" | {frame_index/max(elapsed,1e-6):.2f} FPS"
                    )
        finally:
            cap.release()
            writer.release()

    elapsed = time.perf_counter() - started
    print("="*76)
    print("DONE - Ball Tracking v1")
    print(f"Frames processed : {frame_index}")
    print(f"Detected frames  : {detected}")
    print(f"Predicted frames : {predicted}")
    print(f"Missing frames   : {missing}")
    print(f"Total candidates : {total_candidates}")
    print(f"Average FPS      : {frame_index/max(elapsed,1e-6):.2f}")
    print(f"Video output     : {output}")
    print(f"JSONL output     : {jsonl_path}")
    print("="*76)


if __name__ == "__main__":
    main()
