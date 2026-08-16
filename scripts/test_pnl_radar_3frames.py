from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from adapters.pnlcalib import PnLCalibAdapter


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - PnLCalib 3-frame radar validation"
    )
    p.add_argument(
        "--video",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument(
        "--tracking",
        default=r"output\player_tracking.jsonl",
    )
    p.add_argument(
        "--pnl-root",
        default=r"E:\Youtube\SporAnimasyon\CalibrationEngines\PnLCalib",
    )
    p.add_argument(
        "--frames",
        default="25,50,75",
    )
    p.add_argument(
        "--output-dir",
        default=r"output\pnl_radar_validation",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
    )
    return p.parse_args()


def read_tracking(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows[int(item["frame_index"])] = item
    return rows


def extract_frames(video_path: Path, frame_indices: list[int], out_dir: Path) -> dict[int, Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output = {}

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index}")

        path = out_dir / f"frame_{frame_index}.jpg"
        cv2.imwrite(str(path), frame)
        output[frame_index] = path

    cap.release()
    return output


def transform_point(H: np.ndarray, x: float, y: float):
    p = H @ np.array([x, y, 1.0], dtype=np.float64)

    if abs(p[2]) < 1e-12:
        return None

    p /= p[2]

    if not np.isfinite(p).all():
        return None

    return float(p[0]), float(p[1])


def color_for_track(track_id: int):
    # deterministic BGR
    return (
        int(60 + (37 * track_id) % 180),
        int(60 + (97 * track_id) % 180),
        int(60 + (157 * track_id) % 180),
    )


def pitch_to_px(
    x_m: float,
    y_m: float,
    width: int,
    height: int,
    margin: int,
):
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin

    px = margin + (x_m / PITCH_LENGTH_M) * usable_w
    py = margin + (y_m / PITCH_WIDTH_M) * usable_h

    return int(round(px)), int(round(py))


def draw_pitch(width: int = 900):
    height = int(round(width * PITCH_WIDTH_M / PITCH_LENGTH_M))
    margin = 30

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (39, 105, 46)

    # mowing stripes
    usable_w = width - 2 * margin
    stripe = usable_w / 12

    for i in range(12):
        if i % 2 == 0:
            x0 = int(round(margin + i * stripe))
            x1 = int(round(margin + (i + 1) * stripe))
            cv2.rectangle(
                img,
                (x0, margin),
                (x1, height - margin),
                (43, 114, 50),
                -1,
            )

    def p(x, y):
        return pitch_to_px(x, y, width, height, margin)

    white = (235, 235, 235)
    t = 2

    # outer boundary
    cv2.rectangle(img, p(0, 0), p(105, 68), white, t, cv2.LINE_AA)

    # halfway
    cv2.line(img, p(52.5, 0), p(52.5, 68), white, t, cv2.LINE_AA)

    # center circle
    cx, cy = p(52.5, 34)
    radius = int(round(9.15 / 105.0 * (width - 2 * margin)))
    cv2.circle(img, (cx, cy), radius, white, t, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, white, -1, cv2.LINE_AA)

    # boxes
    for left in (True, False):
        if left:
            x_goal = 0.0
            x_pen = 16.5
            x_goalbox = 5.5
            x_spot = 11.0
        else:
            x_goal = 105.0
            x_pen = 88.5
            x_goalbox = 99.5
            x_spot = 94.0

        cv2.rectangle(
            img,
            p(min(x_goal, x_pen), 13.84),
            p(max(x_goal, x_pen), 54.16),
            white,
            t,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            img,
            p(min(x_goal, x_goalbox), 24.84),
            p(max(x_goal, x_goalbox), 43.16),
            white,
            t,
            cv2.LINE_AA,
        )

        cv2.circle(img, p(x_spot, 34.0), 3, white, -1, cv2.LINE_AA)

    return img, margin


def annotate_source(frame: np.ndarray, tracks: list[dict]):
    out = frame.copy()

    for tr in tracks:
        foot = tr.get("foot_point")
        if not foot:
            continue

        track_id = int(tr.get("track_id", -1))
        color = color_for_track(track_id)

        x, y = int(round(foot[0])), int(round(foot[1]))

        cv2.circle(
            out,
            (x, y),
            7,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            f"ID {track_id}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def render_validation(
    frame: np.ndarray,
    tracks: list[dict],
    H_list: list[list[float]],
):
    H = np.asarray(H_list, dtype=np.float64)

    radar, margin = draw_pitch()
    radar_h, radar_w = radar.shape[:2]

    projected = []
    rejected = []

    for tr in tracks:
        foot = tr.get("foot_point")
        if not foot:
            continue

        track_id = int(tr.get("track_id", -1))
        role = tr.get("class_name", "player")

        result = transform_point(
            H,
            float(foot[0]),
            float(foot[1]),
        )

        if result is None:
            rejected.append({
                "track_id": track_id,
                "reason": "invalid_transform",
            })
            continue

        x_m, y_m = result

        # strict pitch bounds for quality validation
        if not (
            0.0 <= x_m <= PITCH_LENGTH_M
            and 0.0 <= y_m <= PITCH_WIDTH_M
        ):
            rejected.append({
                "track_id": track_id,
                "reason": "outside_pitch",
                "pitch_position_m": [
                    round(x_m, 3),
                    round(y_m, 3),
                ],
            })
            continue

        projected.append({
            "track_id": track_id,
            "class_name": role,
            "pitch_position_m": [
                round(x_m, 3),
                round(y_m, 3),
            ],
            "foot_point": foot,
        })

        rx, ry = pitch_to_px(
            x_m,
            y_m,
            radar_w,
            radar_h,
            margin,
        )

        color = color_for_track(track_id)

        cv2.circle(
            radar,
            (rx, ry),
            9,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            radar,
            str(track_id),
            (rx + 10, ry - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    source = annotate_source(frame, tracks)

    # add headers
    cv2.rectangle(
        source,
        (0, 0),
        (source.shape[1], 48),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        source,
        f"IMAGE FOOT POINTS | tracks={len(tracks)}",
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        radar,
        (0, 0),
        (radar.shape[1], 48),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        radar,
        (
            f"PnL RADAR | projected={len(projected)} | "
            f"rejected={len(rejected)}"
        ),
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    # fit both panels to same height then concatenate
    target_h = max(source.shape[0], radar.shape[0])

    def fit_height(img):
        if img.shape[0] == target_h:
            return img
        scale = target_h / img.shape[0]
        return cv2.resize(
            img,
            (int(round(img.shape[1] * scale)), target_h),
            interpolation=cv2.INTER_AREA,
        )

    source = fit_height(source)
    radar = fit_height(radar)

    combined = np.hstack([source, radar])

    return combined, projected, rejected


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]

    video = Path(args.video)
    tracking_path = (
        Path(args.tracking)
        if Path(args.tracking).is_absolute()
        else project_root / args.tracking
    )
    pnl_root = Path(args.pnl_root)
    output_dir = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else project_root / args.output_dir
    )

    frame_indices = [
        int(x.strip())
        for x in args.frames.split(",")
        if x.strip()
    ]

    for p in (video, tracking_path, pnl_root):
        if not p.exists():
            raise FileNotFoundError(p)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_frames_dir = output_dir / "frames"
    temp_frames_dir.mkdir(parents=True, exist_ok=True)

    tracking = read_tracking(tracking_path)

    frames = extract_frames(
        video,
        frame_indices,
        temp_frames_dir,
    )

    adapter = PnLCalibAdapter(
        pnl_root=pnl_root,
        device=args.device,
        pnl_refine=True,
    )

    ready, message = adapter.health_check()
    print(message)

    if not ready:
        raise RuntimeError(message)

    calibration_results = adapter.calibrate_many_with_metrics(
        [frames[i] for i in frame_indices]
    )

    validation_json = []
    rendered = []

    for frame_index, calibration in zip(
        frame_indices,
        calibration_results,
    ):
        frame = cv2.imread(str(frames[frame_index]))

        tracks = tracking.get(
            frame_index,
            {"tracks": []},
        ).get("tracks", [])

        if calibration.get("status") != "ok":
            print(
                f"Frame {frame_index}: calibration failed -> "
                f"{calibration.get('error')}"
            )
            validation_json.append({
                "frame_index": frame_index,
                "calibration": calibration,
                "projected": [],
                "rejected": [],
            })
            continue

        combined, projected, rejected = render_validation(
            frame,
            tracks,
            calibration["homography_image_to_pitch"],
        )

        out_path = output_dir / f"pnl_radar_frame_{frame_index}.jpg"
        cv2.imwrite(str(out_path), combined)
        rendered.append(combined)

        validation_json.append({
            "frame_index": frame_index,
            "calibration": calibration,
            "tracked_objects": len(tracks),
            "projected": projected,
            "rejected": rejected,
        })

        print(
            f"Frame {frame_index}: "
            f"tracks={len(tracks)} | "
            f"projected={len(projected)} | "
            f"rejected={len(rejected)} | "
            f"rep_err={calibration.get('rep_err')}"
        )

    json_path = output_dir / "validation.json"
    json_path.write_text(
        json.dumps(
            validation_json,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # inspection video
    if rendered:
        max_h = max(img.shape[0] for img in rendered)
        max_w = max(img.shape[1] for img in rendered)

        normalized = []
        for img in rendered:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[:] = (15, 15, 15)

            y0 = (max_h - img.shape[0]) // 2
            x0 = (max_w - img.shape[1]) // 2

            canvas[
                y0:y0 + img.shape[0],
                x0:x0 + img.shape[1],
            ] = img

            normalized.append(canvas)

        video_out = output_dir / "pnl_radar_validation.mp4"

        writer = cv2.VideoWriter(
            str(video_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (max_w, max_h),
        )

        for img in normalized:
            # 2 seconds per benchmark frame
            for _ in range(50):
                writer.write(img)

        writer.release()

        print(f"Video -> {video_out}")

    print(f"JSON  -> {json_path}")


if __name__ == "__main__":
    main()
