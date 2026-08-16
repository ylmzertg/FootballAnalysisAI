from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


FOCUS_IDS = {11, 28, 17, 20, 31, 30, 22, 35}


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose overlapping/duplicate role tracks using exact PnL pitch_xy."
    )
    p.add_argument(
        "--jsonl",
        default=r"output\team_classification_v23_pnl_exact.jsonl",
    )
    p.add_argument(
        "--max-median-distance",
        type=float,
        default=2.5,
        help="Metres. Pair is a strong duplicate candidate below this median distance.",
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=2,
    )
    return p.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def bbox_iou(a, b):
    if not a or not b or len(a) != 4 or len(b) != 4:
        return None

    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return None

    return inter / union


def pitch_distance(a, b):
    if a is None or b is None:
        return None
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except Exception:
        return None


def role_family(track):
    v = track.get("team_v23") or {}
    role = str(v.get("role", ""))
    src = str(track.get("class_name", "")).lower()

    if role == "REFEREE" or src == "referee":
        return "REF"
    if role == "GOALKEEPER" or src == "goalkeeper":
        return "GK"
    return "OTHER"


def main():
    args = parse_args()
    path = Path(args.jsonl)

    if not path.exists():
        raise FileNotFoundError(path)

    # frames[frame_index][track_id] = track
    frames = {}
    role_ids = set()

    for row in load_jsonl(path):
        fi = int(row.get("frame_index", -1))
        row_map = {}

        for tr in row.get("tracks", []):
            tid = int(tr.get("track_id", -1))
            if tid < 0:
                continue
            row_map[tid] = tr

            if role_family(tr) in {"REF", "GK"}:
                role_ids.add(tid)

        frames[fi] = row_map

    role_ids |= FOCUS_IDS
    role_ids = sorted(role_ids)

    results = []

    for i, a_id in enumerate(role_ids):
        for b_id in role_ids[i + 1:]:
            overlap_frames = []
            distances = []
            ious = []
            same_family = []

            for fi, row in frames.items():
                a = row.get(a_id)
                b = row.get(b_id)
                if a is None or b is None:
                    continue

                overlap_frames.append(fi)

                da = a.get("pitch_xy")
                db = b.get("pitch_xy")
                d = pitch_distance(da, db)
                if d is not None:
                    distances.append(d)

                iou = bbox_iou(a.get("bbox_xyxy"), b.get("bbox_xyxy"))
                if iou is not None:
                    ious.append(iou)

                same_family.append(role_family(a) == role_family(b))

            if len(overlap_frames) < args.min_overlap:
                continue

            med_dist = median(distances) if distances else None
            min_dist = min(distances) if distances else None
            max_dist = max(distances) if distances else None

            med_iou = median(ious) if ious else None
            max_iou = max(ious) if ious else None

            duplicate_score = 0
            if med_dist is not None and med_dist <= args.max_median_distance:
                duplicate_score += 2
            if min_dist is not None and min_dist <= 1.0:
                duplicate_score += 1
            if med_iou is not None and med_iou >= 0.25:
                duplicate_score += 2
            if max_iou is not None and max_iou >= 0.50:
                duplicate_score += 1
            if same_family and sum(same_family) / len(same_family) >= 0.7:
                duplicate_score += 1

            results.append({
                "a": a_id,
                "b": b_id,
                "overlap": len(overlap_frames),
                "first": min(overlap_frames),
                "last": max(overlap_frames),
                "median_dist": med_dist,
                "min_dist": min_dist,
                "max_dist": max_dist,
                "median_iou": med_iou,
                "max_iou": max_iou,
                "score": duplicate_score,
            })

    results.sort(
        key=lambda r: (
            -r["score"],
            r["median_dist"] if r["median_dist"] is not None else 9999,
            -r["overlap"],
        )
    )

    print("=" * 118)
    print("FootballAnalysisAI - OVERLAPPING ROLE TRACK DIAGNOSTIC")
    print(f"Input: {path}")
    print("=" * 118)

    print("\nSTRONG DUPLICATE CANDIDATES")
    print("-" * 118)

    strong = [r for r in results if r["score"] >= 4]

    if not strong:
        print("NONE")
    else:
        for r in strong:
            print(
                f"ID {r['a']:>3} <-> ID {r['b']:>3} | "
                f"overlap={r['overlap']:>3} frames ({r['first']}..{r['last']}) | "
                f"pitch median={r['median_dist']:.2f}m "
                f"min={r['min_dist']:.2f}m max={r['max_dist']:.2f}m | "
                f"IoU median={r['median_iou'] if r['median_iou'] is not None else '-'} "
                f"max={r['max_iou'] if r['max_iou'] is not None else '-'} | "
                f"score={r['score']}"
            )

    print("\nFOCUS PAIRS")
    print("-" * 118)

    focus_pairs = {
        tuple(sorted(x))
        for x in [
            (11, 28),
            (20, 31),
            (20, 30),
            (31, 30),
            (20, 22),
            (20, 35),
            (31, 35),
            (30, 35),
        ]
    }

    found = set()

    for r in results:
        pair = tuple(sorted((r["a"], r["b"])))
        if pair not in focus_pairs:
            continue
        found.add(pair)

        md = "-" if r["median_dist"] is None else f"{r['median_dist']:.2f}m"
        mi = "-" if r["median_iou"] is None else f"{r['median_iou']:.3f}"

        print(
            f"ID {r['a']:>3} <-> ID {r['b']:>3} | "
            f"overlap={r['overlap']:>3} | median pitch={md} | "
            f"median IoU={mi} | score={r['score']}"
        )

    for pair in sorted(focus_pairs - found):
        print(f"ID {pair[0]:>3} <-> ID {pair[1]:>3} | no >= {args.min_overlap}-frame overlap")

    print("\nINTERPRETATION")
    print("-" * 118)
    print("score >= 4 : strong duplicate-track candidate")
    print("score 2-3  : inspect visually / possible nearby different player")
    print("score 0-1  : likely different physical people")
    print("=" * 118)


if __name__ == "__main__":
    main()
