from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose Team Classifier V2.1 referee/role decisions."
    )
    p.add_argument(
        "--tracking",
        default=r"output\player_tracking.jsonl",
        help="Original ByteTrack JSONL",
    )
    p.add_argument(
        "--classification",
        default=r"output\team_classification_v21.jsonl",
        help="V2.1 classification JSONL",
    )
    return p.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    args = parse_args()
    tracking_path = Path(args.tracking)
    class_path = Path(args.classification)

    if not tracking_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_path}")
    if not class_path.exists():
        raise FileNotFoundError(f"Classification file not found: {class_path}")

    source_classes = defaultdict(Counter)
    source_conf = defaultdict(list)
    source_frames = defaultdict(list)

    for row in load_jsonl(tracking_path):
        frame_index = int(row.get("frame_index", -1))
        for tr in row.get("tracks", []):
            tid = int(tr.get("track_id", -1))
            if tid < 0:
                continue
            cls = str(tr.get("class_name", "UNKNOWN"))
            source_classes[tid][cls] += 1
            try:
                source_conf[tid].append(float(tr.get("confidence", 0.0)))
            except Exception:
                pass
            source_frames[tid].append(frame_index)

    assigned_roles = defaultdict(Counter)
    assigned_teams = defaultdict(Counter)
    reasons = defaultdict(Counter)
    ref_frames = defaultdict(list)

    for row in load_jsonl(class_path):
        frame_index = int(row.get("frame_index", -1))
        for tr in row.get("tracks", []):
            tid = int(tr.get("track_id", -1))
            if tid < 0:
                continue

            v2 = tr.get("team_v2") or {}
            role = str(v2.get("role", "UNKNOWN"))
            team = str(v2.get("team", "UNKNOWN"))
            reason = str(v2.get("reason", ""))

            assigned_roles[tid][role] += 1
            assigned_teams[tid][team] += 1

            for part in reason.split(";"):
                part = part.strip()
                if part:
                    reasons[tid][part] += 1

            if role == "REFEREE":
                ref_frames[tid].append(frame_index)

    track_ids = sorted(
        set(source_classes) | set(assigned_roles),
        key=lambda tid: (
            -assigned_roles[tid].get("REFEREE", 0),
            tid,
        ),
    )

    print("=" * 110)
    print("FootballAnalysisAI - Team Classifier V2.1 ROLE DIAGNOSTIC")
    print(f"Tracking       : {tracking_path}")
    print(f"Classification : {class_path}")
    print("=" * 110)

    print("\nREFEREE / ROLE SUMMARY")
    print("-" * 110)

    header = (
        f"{'ID':>4} | {'SRC CLASS COUNTS':<28} | {'ROLE COUNTS':<32} | "
        f"{'TEAM COUNTS':<32} | {'REF FRAMES':>10}"
    )
    print(header)
    print("-" * len(header))

    for tid in track_ids:
        ref_count = assigned_roles[tid].get("REFEREE", 0)
        if ref_count <= 0:
            continue

        src = dict(source_classes[tid])
        roles = dict(assigned_roles[tid])
        teams = dict(assigned_teams[tid])

        print(
            f"{tid:>4} | "
            f"{str(src):<28.28} | "
            f"{str(roles):<32.32} | "
            f"{str(teams):<32.32} | "
            f"{ref_count:>10}"
        )

        top_reasons = reasons[tid].most_common(8)
        if top_reasons:
            print("     reasons:", ", ".join(f"{k}={v}" for k, v in top_reasons))

        frames = ref_frames[tid]
        if frames:
            print(
                f"     REF frame range: {min(frames)}..{max(frames)}"
                f" | first={frames[:8]}"
                f"{' ...' if len(frames) > 8 else ''}"
            )

        confs = source_conf[tid]
        if confs:
            print(
                f"     detector confidence: "
                f"mean={sum(confs)/len(confs):.3f}, "
                f"min={min(confs):.3f}, max={max(confs):.3f}"
            )

    print("\nTRACKS WITH SOURCE 'referee' HINT")
    print("-" * 110)
    hinted = []
    for tid in sorted(source_classes):
        count = sum(
            n for cls, n in source_classes[tid].items()
            if cls.lower() == "referee"
        )
        if count:
            hinted.append((tid, count, dict(source_classes[tid])))

    if not hinted:
        print("NONE")
        print(
            "=> ByteTrack JSONL does not contain a referee class hint. "
            "All referee decisions are therefore coming from TeamClassifierV2 auto-correction."
        )
    else:
        for tid, count, counts in sorted(hinted, key=lambda x: -x[1]):
            print(f"ID {tid:>3} | referee hint frames={count:>3} | source={counts}")

    print("\nTOP NON-REF TRACKS (for comparison)")
    print("-" * 110)
    non_ref = []
    for tid in track_ids:
        if assigned_roles[tid].get("REFEREE", 0) == 0:
            total = sum(assigned_roles[tid].values())
            if total:
                non_ref.append((total, tid))

    for total, tid in sorted(non_ref, reverse=True)[:12]:
        print(
            f"ID {tid:>3} | source={dict(source_classes[tid])} | "
            f"roles={dict(assigned_roles[tid])} | teams={dict(assigned_teams[tid])}"
        )

    print("\nDIAGNOSIS HINT")
    print("-" * 110)
    if not hinted:
        print(
            "No trusted referee detector hints exist. Recommended next fix: "
            "disable automatic REF promotion as a hard label, keep it as REF_CANDIDATE, "
            "and only finalize referee after a dedicated role cue / later PnL-aware stage."
        )
    else:
        print(
            "Trusted referee hints exist. Recommended next fix: "
            "learn referee prototype only from those hinted tracks and completely disable "
            "prototype learning from auto candidates."
        )

    print("=" * 110)


if __name__ == "__main__":
    main()
