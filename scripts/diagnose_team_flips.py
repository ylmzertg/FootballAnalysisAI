from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    return p.parse_args()


def assignment(track):
    return track.get("team_v25") or track.get("team_v24") or {}


def main():
    args = parse_args()
    histories = defaultdict(list)

    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame_index"])
            for track in row.get("tracks", []):
                tid = int(track.get("track_id", -1))
                team = str(assignment(track).get("team", "UNKNOWN"))
                if tid >= 0 and team in {"TEAM_A", "TEAM_B"}:
                    histories[tid].append((frame, team))

    rows = []
    for tid, history in histories.items():
        transitions = 0
        prev = None
        counts = Counter(team for _, team in history)
        for _, team in history:
            if prev is not None and team != prev:
                transitions += 1
            prev = team
        if transitions:
            rows.append((
                transitions, tid, len(history),
                counts["TEAM_A"], counts["TEAM_B"],
                history[0][0], history[-1][0],
            ))

    rows.sort(reverse=True)
    print("tracks with A/B transitions:", len(rows))
    if not rows:
        print("NONE")
        return

    for flips, tid, samples, a, b, first, last in rows[:30]:
        print(
            f"ID {tid:>3} | flips={flips:>3} | samples={samples:>3} "
            f"| A={a:>3} B={b:>3} | frames={first}..{last}"
        )


if __name__ == "__main__":
    main()
