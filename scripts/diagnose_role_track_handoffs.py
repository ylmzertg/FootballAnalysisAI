from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median


REFEREE = "REFEREE"
GOALKEEPER = "GOALKEEPER"
OUTSIDE = "OUTSIDE_PITCH"


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose ByteTrack role-ID fragmentation after exact PnLCalib."
    )
    p.add_argument(
        "--jsonl",
        default=r"output\team_classification_v23_pnl_exact.jsonl",
    )
    p.add_argument(
        "--max-gap",
        type=int,
        default=5,
        help="Maximum frame gap considered a possible ID handoff.",
    )
    p.add_argument(
        "--max-pitch-distance",
        type=float,
        default=10.0,
        help="Maximum metres between old-track end and new-track start.",
    )
    return p.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


@dataclass
class TrackSummary:
    track_id: int
    frames: list[int]
    pitch: list[tuple[int, tuple[float, float]]]
    source_classes: Counter
    roles: Counter
    teams: Counter
    spatial: Counter
    ref_hint_frames: list[int]
    gk_hint_frames: list[int]

    @property
    def first_frame(self):
        return min(self.frames)

    @property
    def last_frame(self):
        return max(self.frames)

    @property
    def first_pitch(self):
        return self.pitch[0][1] if self.pitch else None

    @property
    def last_pitch(self):
        return self.pitch[-1][1] if self.pitch else None

    @property
    def median_pitch(self):
        if not self.pitch:
            return None
        xs = [p[1][0] for p in self.pitch]
        ys = [p[1][1] for p in self.pitch]
        return (median(xs), median(ys))

    @property
    def dominant_role(self):
        return self.roles.most_common(1)[0][0] if self.roles else "UNKNOWN"

    @property
    def dominant_source(self):
        return self.source_classes.most_common(1)[0][0] if self.source_classes else "UNKNOWN"


def build_tracks(path: Path):
    raw = defaultdict(lambda: {
        "frames": [],
        "pitch": [],
        "source_classes": Counter(),
        "roles": Counter(),
        "teams": Counter(),
        "spatial": Counter(),
        "ref_hint_frames": [],
        "gk_hint_frames": [],
    })

    for row in load_jsonl(path):
        fi = int(row.get("frame_index", -1))
        for tr in row.get("tracks", []):
            tid = int(tr.get("track_id", -1))
            if tid < 0:
                continue

            d = raw[tid]
            d["frames"].append(fi)

            source_cls = str(tr.get("class_name", "UNKNOWN")).lower()
            d["source_classes"][source_cls] += 1
            if source_cls == "referee":
                d["ref_hint_frames"].append(fi)
            elif source_cls == "goalkeeper":
                d["gk_hint_frames"].append(fi)

            spatial = str(tr.get("spatial_status", "UNKNOWN"))
            d["spatial"][spatial] += 1

            xy = tr.get("pitch_xy")
            if xy is not None and len(xy) == 2 and spatial == "inside_pitch":
                d["pitch"].append((fi, (float(xy[0]), float(xy[1]))))

            cls = tr.get("team_v23") or {}
            d["roles"][str(cls.get("role", "UNKNOWN"))] += 1
            d["teams"][str(cls.get("team", "UNKNOWN"))] += 1

    result = {}
    for tid, d in raw.items():
        d["pitch"].sort(key=lambda x: x[0])
        result[tid] = TrackSummary(track_id=tid, **d)
    return result


def role_interest(t: TrackSummary):
    return (
        t.roles.get(REFEREE, 0) > 0
        or t.roles.get(GOALKEEPER, 0) > 0
        or t.source_classes.get("referee", 0) > 0
        or t.source_classes.get("goalkeeper", 0) > 0
    )


def role_family(t: TrackSummary):
    ref_score = t.roles.get(REFEREE, 0) + t.source_classes.get("referee", 0)
    gk_score = t.roles.get(GOALKEEPER, 0) + t.source_classes.get("goalkeeper", 0)
    if ref_score >= gk_score and ref_score > 0:
        return "REF"
    if gk_score > 0:
        return "GK"
    return "OTHER"


def handoff_candidates(tracks, max_gap, max_distance):
    candidates = []
    vals = [t for t in tracks.values() if role_interest(t)]

    for a in vals:
        for b in vals:
            if a.track_id == b.track_id:
                continue
            if role_family(a) != role_family(b):
                continue

            gap = b.first_frame - a.last_frame
            if gap < 0 or gap > max_gap:
                continue

            dist = distance(a.last_pitch, b.first_pitch)
            if not math.isfinite(dist) or dist > max_distance:
                continue

            candidates.append((gap, dist, a, b))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates


def fmt_xy(xy):
    if xy is None:
        return "-"
    return f"({xy[0]:.1f},{xy[1]:.1f})"


def main():
    args = parse_args()
    path = Path(args.jsonl)

    if not path.exists():
        raise FileNotFoundError(path)

    tracks = build_tracks(path)

    print("=" * 118)
    print("FootballAnalysisAI - ROLE TRACK HANDOFF DIAGNOSTIC")
    print(f"Input: {path}")
    print("=" * 118)

    print("\nROLE-RELEVANT TRACKS")
    print("-" * 118)

    interesting = [t for t in tracks.values() if role_interest(t)]
    interesting.sort(
        key=lambda t: (
            0 if role_family(t) == "REF" else 1,
            -t.roles.get(REFEREE, 0),
            -t.roles.get(GOALKEEPER, 0),
            t.track_id,
        )
    )

    for t in interesting:
        print(
            f"ID {t.track_id:>3} | family={role_family(t):>3} | "
            f"frames={t.first_frame:>3}..{t.last_frame:<3} ({len(t.frames):>3}) | "
            f"source={dict(t.source_classes)}"
        )
        print(
            f"       roles={dict(t.roles)} | teams={dict(t.teams)} | "
            f"spatial={dict(t.spatial)}"
        )
        print(
            f"       pitch first={fmt_xy(t.first_pitch)} "
            f"last={fmt_xy(t.last_pitch)} median={fmt_xy(t.median_pitch)} | "
            f"ref_hint={len(t.ref_hint_frames)} gk_hint={len(t.gk_hint_frames)}"
        )

    candidates = handoff_candidates(
        tracks,
        max_gap=max(0, args.max_gap),
        max_distance=max(0.0, args.max_pitch_distance),
    )

    print("\nPOSSIBLE ROLE ID HANDOFFS")
    print("-" * 118)

    if not candidates:
        print("NONE")
    else:
        for gap, dist, a, b in candidates:
            print(
                f"{role_family(a)}: ID {a.track_id} -> ID {b.track_id} | "
                f"gap={gap} frame(s) | pitch distance={dist:.2f} m | "
                f"{fmt_xy(a.last_pitch)} -> {fmt_xy(b.first_pitch)}"
            )

    print("\nFOCUS")
    print("-" * 118)
    for tid in (11, 28, 17, 31, 20, 30, 22, 35):
        t = tracks.get(tid)
        if t is None:
            print(f"ID {tid:>3}: not present")
            continue
        print(
            f"ID {tid:>3}: family={role_family(t)}, frames={t.first_frame}..{t.last_frame}, "
            f"roles={dict(t.roles)}, source={dict(t.source_classes)}, "
            f"spatial={dict(t.spatial)}, first={fmt_xy(t.first_pitch)}, "
            f"last={fmt_xy(t.last_pitch)}"
        )

    print("=" * 118)


if __name__ == "__main__":
    main()
