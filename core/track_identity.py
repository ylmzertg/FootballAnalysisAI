from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import math


@dataclass
class CanonicalIdentityConfig:
    min_overlap_frames: int = 2
    max_median_pitch_distance_m: float = 0.75
    min_median_bbox_iou: float = 0.80


@dataclass
class DuplicatePair:
    track_a: int
    track_b: int
    overlap_frames: int
    median_pitch_distance_m: float
    median_bbox_iou: float


class _UnionFind:
    def __init__(self, ids: Iterable[int]):
        self.parent = {int(i): int(i) for i in ids}

    def find(self, x: int) -> int:
        x = int(x)
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def pitch_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 2 or len(b) != 2:
        return float("inf")
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def build_canonical_alias_map(
    frame_records: Mapping[int, Sequence[Mapping]],
    config: CanonicalIdentityConfig | None = None,
) -> tuple[dict[int, int], list[DuplicatePair]]:
    """Build an offline canonical identity map for exact duplicate ByteTrack IDs.

    Each input record should contain: track_id, bbox_xyxy, pitch_xy. Only pairs
    that overlap in time with both *very high bbox overlap* and *very small PnL
    pitch distance* are merged. Nearby players are deliberately not merged.
    """
    config = config or CanonicalIdentityConfig()

    ids: set[int] = set()
    first_frame: dict[int, int] = {}
    seen_count: Counter[int] = Counter()
    pair_distances: MutableMapping[tuple[int, int], list[float]] = defaultdict(list)
    pair_ious: MutableMapping[tuple[int, int], list[float]] = defaultdict(list)

    for frame_index, records in frame_records.items():
        valid = []
        for rec in records:
            try:
                tid = int(rec["track_id"])
                bbox = rec["bbox_xyxy"]
                pitch = rec["pitch_xy"]
                if bbox is None or pitch is None:
                    continue
                ids.add(tid)
                first_frame[tid] = min(frame_index, first_frame.get(tid, frame_index))
                seen_count[tid] += 1
                valid.append((tid, bbox, pitch))
            except Exception:
                continue

        for i in range(len(valid)):
            a_id, a_bbox, a_pitch = valid[i]
            for j in range(i + 1, len(valid)):
                b_id, b_bbox, b_pitch = valid[j]
                key = tuple(sorted((a_id, b_id)))
                pair_distances[key].append(pitch_distance(a_pitch, b_pitch))
                pair_ious[key].append(bbox_iou(a_bbox, b_bbox))

    duplicates: list[DuplicatePair] = []
    uf = _UnionFind(ids)

    for key in pair_distances:
        distances = pair_distances[key]
        ious = pair_ious[key]
        if len(distances) < max(1, config.min_overlap_frames):
            continue
        med_d = float(median(distances))
        med_iou = float(median(ious))
        if (
            med_d <= config.max_median_pitch_distance_m
            and med_iou >= config.min_median_bbox_iou
        ):
            a, b = key
            duplicates.append(
                DuplicatePair(
                    track_a=a,
                    track_b=b,
                    overlap_frames=len(distances),
                    median_pitch_distance_m=med_d,
                    median_bbox_iou=med_iou,
                )
            )
            uf.union(a, b)

    # Convert each union component to a stable canonical ID. Prefer the track
    # that existed first; then the longer-lived track; then the smaller raw ID.
    groups: MutableMapping[int, list[int]] = defaultdict(list)
    for tid in ids:
        groups[uf.find(tid)].append(tid)

    alias_map: dict[int, int] = {}
    for members in groups.values():
        canonical = min(
            members,
            key=lambda tid: (
                first_frame.get(tid, 10**9),
                -seen_count.get(tid, 0),
                tid,
            ),
        )
        for tid in members:
            alias_map[tid] = canonical

    duplicates.sort(
        key=lambda d: (
            -d.overlap_frames,
            d.median_pitch_distance_m,
            -d.median_bbox_iou,
        )
    )
    return alias_map, duplicates


def collapse_frame_records(
    records: Sequence[Mapping],
    alias_map: Mapping[int, int],
) -> list[dict]:
    """Collapse simultaneous duplicate raw tracks into one canonical observation.

    The representative bbox is chosen by detector confidence. Detector class hints
    are confidence-weighted across duplicate boxes. Raw IDs are preserved in the
    output for debugging/auditability.
    """
    groups: MutableMapping[int, list[Mapping]] = defaultdict(list)
    for rec in records:
        try:
            raw_id = int(rec["track_id"])
        except Exception:
            continue
        groups[int(alias_map.get(raw_id, raw_id))].append(rec)

    collapsed: list[dict] = []
    for canonical_id, members in groups.items():
        representative = max(
            members,
            key=lambda r: float(r.get("confidence", 0.0) or 0.0),
        )
        out = dict(representative)
        raw_ids = sorted({int(m["track_id"]) for m in members})
        out["raw_track_ids"] = raw_ids
        out["raw_track_id"] = int(representative["track_id"])
        out["track_id"] = int(canonical_id)

        class_scores: Counter[str] = Counter()
        for m in members:
            cls = str(m.get("class_name", "player")).lower()
            conf = max(0.01, float(m.get("confidence", 1.0) or 1.0))
            class_scores[cls] += conf
        if class_scores:
            out["class_name"] = class_scores.most_common(1)[0][0]

        # Prefer the representative geometry, but average PnL coordinates when
        # duplicate boxes both have valid pitch positions.
        pitches = [m.get("pitch_xy") for m in members if m.get("pitch_xy") is not None]
        if pitches:
            out["pitch_xy"] = [
                sum(float(p[0]) for p in pitches) / len(pitches),
                sum(float(p[1]) for p in pitches) / len(pitches),
            ]
        collapsed.append(out)

    return collapsed
