from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from adapters.pnlcalib import PnLCalibAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PNL_ROOT = Path(r"E:\Youtube\SporAnimasyon\CalibrationEngines\PnLCalib")


def transform_image_point(H_list, x: float, y: float):
    H = np.asarray(H_list, dtype=np.float64)
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    p /= p[2]
    return float(p[0]), float(p[1])


def main():
    frames = [
        PNL_ROOT / "benchmark_frames" / "frame_25.jpg",
        PNL_ROOT / "benchmark_frames" / "frame_50.jpg",
        PNL_ROOT / "benchmark_frames" / "frame_75.jpg",
    ]

    adapter = PnLCalibAdapter(
        pnl_root=PNL_ROOT,
        device="cuda:0",
        pnl_refine=True,
    )

    ready, message = adapter.health_check()
    print(message)

    if not ready:
        raise SystemExit(1)

    results = adapter.calibrate_many_with_metrics(frames)

    output = PROJECT_ROOT / "output" / "pnl_adapter_3frames.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 76)

    for frame, item in zip(frames, results):
        print(f"Frame: {frame.name}")
        print(f"  status       : {item.get('status')}")

        if item.get("status") != "ok":
            print(f"  error        : {item.get('error')}")
            continue

        print(f"  rep_err      : {item.get('rep_err')}")
        print(f"  quality      : {item.get('quality_score'):.4f}")
        print(f"  mode         : {item.get('mode')}")
        print(f"  ransac       : {item.get('use_ransac')}")

        H = item["homography_image_to_pitch"]
        print("  H image->pitch:")
        for row in H:
            print("   ", " ".join(f"{v: .8f}" for v in row))

    print("=" * 76)
    print(f"JSON -> {output}")


if __name__ == "__main__":
    main()
