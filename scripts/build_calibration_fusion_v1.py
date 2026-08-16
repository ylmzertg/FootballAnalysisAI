from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.tvcalib import TVCalibAdapter
from core.calibration_fusion import build_fused_geometry


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - PnLCalib primary + TVCalib fallback fusion"
    )
    p.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument(
        "--tracking",
        default=r"output\player_tracking.jsonl",
    )
    p.add_argument(
        "--pnl-json",
        default=r"output\team_classification_v24_pnl_exact_calibration.json",
    )
    p.add_argument(
        "--tv-root",
        default=r"E:\Youtube\SporAnimasyon\CalibrationEngines\tvcalib",
    )
    p.add_argument(
        "--tv-checkpoint",
        default="",
    )
    p.add_argument("--tv-device", default="cuda")
    p.add_argument("--tv-optim-steps", type=int, default=800)
    p.add_argument("--tv-tau", type=float, default=0.017)
    p.add_argument(
        "--tv-frames-dir",
        default=r"output\tvcalib_fallback_frames",
    )
    p.add_argument(
        "--tv-raw-json",
        default=r"output\tvcalib_fallback_results.json",
    )
    p.add_argument(
        "--output",
        default=r"output\calibration_fusion_v1.json",
    )
    return p.parse_args()


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_tracking_frame_indices(path: Path) -> list[int]:
    result = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            result.append(int(item["frame_index"]))
    return sorted(set(result))


def pnl_record_accepted(record: dict) -> bool:
    if record.get("status") != "ok":
        return False
    for key in ("accepted_for_v24", "accepted_for_v23", "accepted"):
        if key in record:
            return bool(record[key])
    return record.get("homography_image_to_pitch") is not None


def extract_frames(
    source: Path,
    frame_indices: list[int],
    output_dir: Path,
) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(frame_indices)
    result = {}

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")

    i = 0
    max_index = max(wanted) if wanted else -1

    try:
        while i <= max_index:
            ok, frame = cap.read()
            if not ok:
                break

            if i in wanted:
                path = output_dir / f"frame_{i:06d}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not write frame {i}: {path}")
                result[i] = path

            i += 1

    finally:
        cap.release()

    missing = sorted(wanted - set(result))
    if missing:
        raise RuntimeError(
            f"Could not extract fallback frames: {missing[:10]}"
        )

    return result


def main():
    args = parse_args()

    source = Path(args.source)
    tracking = resolve(args.tracking)
    pnl_json = resolve(args.pnl_json)
    tv_root = Path(args.tv_root)
    tv_frames_dir = resolve(args.tv_frames_dir)
    tv_raw_json = resolve(args.tv_raw_json)
    output = resolve(args.output)

    for p in (source, tracking, pnl_json, tv_root):
        if not p.exists():
            raise FileNotFoundError(p)

    frame_indices = read_tracking_frame_indices(tracking)
    pnl_records = json.loads(pnl_json.read_text(encoding="utf-8"))
    pnl_by_frame = {
        int(x["frame_index"]): x
        for x in pnl_records
        if "frame_index" in x
    }

    pnl_ok = [
        fi
        for fi in frame_indices
        if fi in pnl_by_frame and pnl_record_accepted(pnl_by_frame[fi])
    ]
    fallback_frames = [fi for fi in frame_indices if fi not in set(pnl_ok)]

    print("=" * 88)
    print("FootballAnalysisAI - CALIBRATION FUSION V1")
    print("Policy          : PnLCalib PRIMARY -> TVCalib FALLBACK")
    print(f"Tracked frames  : {len(frame_indices)}")
    print(f"PnL accepted    : {len(pnl_ok)}")
    print(f"TV fallback req : {len(fallback_frames)}")
    print("=" * 88)

    tv_records = []

    if fallback_frames:
        extracted = extract_frames(
            source,
            fallback_frames,
            tv_frames_dir,
        )

        checkpoint = (
            Path(args.tv_checkpoint)
            if args.tv_checkpoint
            else None
        )

        adapter = TVCalibAdapter(
            tv_root=tv_root,
            checkpoint=checkpoint,
            device=args.tv_device,
            optim_steps=args.tv_optim_steps,
            tau=args.tv_tau,
        )

        ready, message = adapter.health_check()
        print(message)
        if not ready:
            raise RuntimeError(message)

        print(
            f"Running TVCalib on {len(fallback_frames)} PnL-missing frames..."
        )

        raw = adapter.calibrate_many_with_metrics(
            [extracted[fi] for fi in fallback_frames]
        )

        for fi, item in zip(fallback_frames, raw):
            rec = dict(item)
            rec["frame_index"] = fi
            tv_records.append(rec)

    tv_raw_json.parent.mkdir(parents=True, exist_ok=True)
    tv_raw_json.write_text(
        json.dumps(tv_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fused = build_fused_geometry(
        frame_indices,
        pnl_records,
        tv_records,
    )

    summary = {
        "total_frames": len(fused),
        "pnlcalib": sum(
            x.get("status") == "ok" and x.get("engine") == "PnLCalib"
            for x in fused
        ),
        "tvcalib": sum(
            x.get("status") == "ok" and x.get("engine") == "TVCalib"
            for x in fused
        ),
        "missing": sum(x.get("status") != "ok" for x in fused),
        "policy": "PnLCalib_PRIMARY__TVCalib_FALLBACK",
        "tvcalib_tau": args.tv_tau,
        "tvcalib_optim_steps": args.tv_optim_steps,
    }

    payload = {
        "version": "calibration-fusion-v1",
        "summary": summary,
        "frames": fused,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 88)
    print("DONE - CALIBRATION FUSION V1")
    print(f"PnLCalib : {summary['pnlcalib']}")
    print(f"TVCalib  : {summary['tvcalib']}")
    print(f"Missing  : {summary['missing']}")
    print(f"Coverage : {(summary['pnlcalib'] + summary['tvcalib']) / max(1, summary['total_frames']) * 100:.1f}%")
    print(f"TV raw   : {tv_raw_json}")
    print(f"Fusion   : {output}")
    print("=" * 88)


if __name__ == "__main__":
    main()
