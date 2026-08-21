from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from core.deep_kit_encoder_v29 import (
    DeepKitEncoder,
    DeepKitEncoderConfig,
)
from core.runtime_paths import resolve_project_path
from core.team_deep_embedding_v29 import (
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    DeepClusterConfig,
    cluster_segment_embeddings,
    map_clusters_from_votes,
)
from core.team_identity_reconciler_v26 import (
    SegmentObservation,
    build_track_segments,
    segment_vote,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "FootballAnalysisAI - Team Identity v2.9 "
            "deep kit embedding + balanced clustering"
        )
    )
    p.add_argument("--source", required=True)
    p.add_argument("--team-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--diagnostics-json", required=True)

    p.add_argument("--device", default="auto")
    p.add_argument("--sample-stride", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=12)
    return p.parse_args()


def read_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def assignment(track):
    return (
        track.get("team_v25")
        or track.get("team_v24")
        or {}
    )


def build_observations(rows):
    observations = []

    for row in rows:
        frame_index = int(row["frame_index"])

        for track in row.get("tracks", []):
            a = assignment(track)
            role = str(a.get("role", "PLAYER")).upper()

            if role in {
                "REFEREE",
                "GOALKEEPER",
                "OUTSIDE_PITCH",
            }:
                continue

            tid = int(track.get("track_id", -1))
            if tid < 0:
                continue

            observations.append(
                SegmentObservation(
                    frame_index=frame_index,
                    track_id=tid,
                    team=str(a.get("team", UNKNOWN)),
                    confidence=float(a.get("confidence", 0.0) or 0.0),
                    raw_team=str(a.get("raw_team", UNKNOWN)),
                    raw_confidence=float(a.get("raw_confidence", 0.0) or 0.0),
                    id_switch_suspected=bool(
                        a.get("id_switch_suspected", False)
                    ),
                    role=role,
                )
            )

    return observations


def segment_lookup(segments):
    lookup = {}

    for segment in segments:
        for obs in segment.observations:
            lookup[(obs.frame_index, obs.track_id)] = segment.segment_id

    return lookup


def crop_player(frame, bbox):
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = [
        int(round(float(v)))
        for v in bbox
    ]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    crop = frame[y1:y2, x1:x2]

    return crop if crop.size > 0 else None


def collect_embeddings(
    *,
    source,
    rows,
    lookup,
    encoder,
    stride,
    max_samples,
):
    wanted = defaultdict(list)

    for row in rows:
        frame_index = int(row["frame_index"])

        for track in row.get("tracks", []):
            tid = int(track.get("track_id", -1))
            sid = lookup.get((frame_index, tid))

            if sid is not None:
                wanted[frame_index].append((sid, track))

    embeddings = defaultdict(list)
    counts = Counter()
    last_sample = {}

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    frame_index = 0
    last = max(wanted) if wanted else -1

    try:
        while frame_index <= last:
            ok, frame = cap.read()

            if not ok:
                break

            for sid, track in wanted.get(frame_index, []):
                if counts[sid] >= max_samples:
                    continue

                previous = last_sample.get(sid, -10_000)

                if frame_index - previous < stride:
                    continue

                bbox = track.get("bbox_xyxy")

                if bbox is None or len(bbox) != 4:
                    continue

                crop = crop_player(frame, bbox)

                if crop is None:
                    continue

                try:
                    embedding = encoder.encode_crop(crop)
                except Exception:
                    continue

                embeddings[sid].append(embedding)
                counts[sid] += 1
                last_sample[sid] = frame_index

            frame_index += 1

    finally:
        cap.release()

    result = {}

    for sid, rows_ in embeddings.items():
        if not rows_:
            continue

        matrix = np.vstack(rows_)

        # Mean of normalized embeddings is more stable than per-frame clustering.
        feature = np.mean(matrix, axis=0)
        norm = float(np.linalg.norm(feature))

        if norm > 1e-12:
            feature /= norm

        result[sid] = feature.astype(np.float32)

    return result, dict(counts)


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    team_path = resolve_project_path(args.team_jsonl)
    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)
    diagnostics_out = resolve_project_path(args.diagnostics_json)

    rows = read_rows(team_path)

    observations = build_observations(rows)
    segments = build_track_segments(
        observations,
        gap_split_frames=12,
    )
    lookup = segment_lookup(segments)

    print("Loading deep kit encoder...")

    encoder = DeepKitEncoder(
        DeepKitEncoderConfig(
            device=args.device,
            batch_size=24,
        )
    )

    print(f"Embedding device: {encoder.device}")

    embeddings, sample_counts = collect_embeddings(
        source=source,
        rows=rows,
        lookup=lookup,
        encoder=encoder,
        stride=max(1, args.sample_stride),
        max_samples=max(3, args.max_samples),
    )

    config = DeepClusterConfig()

    assignments, centers = cluster_segment_embeddings(
        embeddings,
        sample_counts,
        config,
    )

    votes = {
        segment.segment_id: segment_vote(segment)
        for segment in segments
    }

    vote_summary = {
        sid: (
            vote.team,
            vote.ratio,
            vote.samples,
        )
        for sid, vote in votes.items()
    }

    mapping, mapping_confidence = map_clusters_from_votes(
        assignments,
        vote_summary,
    )

    segment_team = {
        sid: (
            mapping[cluster.cluster_id]
            if cluster.reliable
            else UNKNOWN
        )
        for sid, cluster in assignments.items()
    }

    counts_by_cluster = Counter(
        cluster.cluster_id
        for cluster in assignments.values()
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    stats = Counter()

    with jsonl_out.open("w", encoding="utf-8") as out:
        try:
            for row in rows:
                ok, frame = cap.read()

                if not ok:
                    break

                payload = dict(row)
                payload_tracks = []

                for track in row.get("tracks", []):
                    enriched = dict(track)
                    original = assignment(track)

                    role = str(
                        original.get("role", "PLAYER")
                    ).upper()

                    tid = int(track.get("track_id", -1))
                    sid = lookup.get((frame_index, tid))

                    if (
                        role in {
                            "REFEREE",
                            "GOALKEEPER",
                            "OUTSIDE_PITCH",
                        }
                        or sid is None
                    ):
                        payload_tracks.append(enriched)
                        continue

                    resolved = segment_team.get(sid, UNKNOWN)

                    if resolved == UNKNOWN:
                        resolved = str(
                            original.get("team", UNKNOWN)
                        )

                    cluster = assignments.get(sid)

                    v29 = dict(original)
                    v29["team"] = resolved
                    v29["v29_original_team"] = str(
                        original.get("team", UNKNOWN)
                    )
                    v29["v29_segment_id"] = sid
                    v29["v29_cluster_id"] = (
                        int(cluster.cluster_id)
                        if cluster is not None
                        else None
                    )
                    v29["v29_cluster_margin"] = (
                        round(float(cluster.margin), 5)
                        if cluster is not None
                        else None
                    )
                    v29["v29_cluster_reliable"] = bool(
                        cluster.reliable
                    ) if cluster else False

                    overridden = (
                        resolved
                        != str(original.get("team", UNKNOWN))
                    )

                    v29["v29_overridden"] = overridden

                    enriched["team_v25_original"] = original
                    enriched["team_v29"] = v29
                    enriched["team_v25"] = v29

                    payload_tracks.append(enriched)

                    if overridden:
                        stats["overridden_samples"] += 1

                payload["tracks"] = payload_tracks

                out.write(
                    json.dumps(payload, ensure_ascii=False)
                    + "\n"
                )

                for track in payload_tracks:
                    a = (
                        track.get("team_v29")
                        or assignment(track)
                    )

                    role = str(
                        a.get("role", "PLAYER")
                    ).upper()

                    if role in {"REFEREE", "OUTSIDE_PITCH"}:
                        continue

                    bbox = track.get("bbox_xyxy")

                    if bbox is None or len(bbox) != 4:
                        continue

                    team = str(a.get("team", UNKNOWN))

                    x1, y1, x2, y2 = [
                        int(round(float(v)))
                        for v in bbox
                    ]

                    color = (
                        (255, 90, 40)
                        if team == TEAM_A
                        else (
                            (40, 70, 255)
                            if team == TEAM_B
                            else (150, 150, 150)
                        )
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        f"ID {track.get('track_id')} {team}",
                        (x1, max(16, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                writer.write(frame)
                frame_index += 1

        finally:
            cap.release()
            writer.release()

    diagnostics = {
        "cluster_counts": dict(counts_by_cluster),
        "mapping": mapping,
        "mapping_confidence": mapping_confidence,
        "embedding_device": encoder.device,
        "segments": [
            {
                "segment_id": sid,
                "cluster": cluster.__dict__,
                "feature_samples": sample_counts.get(sid, 0),
                "resolved_team": segment_team.get(sid),
                "v25_vote": votes[sid].__dict__
                if sid in votes
                else None,
            }
            for sid, cluster in assignments.items()
        ],
    }

    diagnostics_out.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    override_segments = []

    by_sid = {
        s.segment_id: s
        for s in segments
    }

    for sid, resolved in segment_team.items():
        segment = by_sid.get(sid)

        if segment is None:
            continue

        old_counts = Counter(
            obs.team
            for obs in segment.observations
            if obs.team in {TEAM_A, TEAM_B}
        )

        old = (
            old_counts.most_common(1)[0][0]
            if old_counts
            else UNKNOWN
        )

        if (
            resolved in {TEAM_A, TEAM_B}
            and resolved != old
        ):
            override_segments.append(
                (
                    sid,
                    old,
                    resolved,
                    assignments[sid].margin,
                )
            )

    print("=" * 94)
    print(
        "DONE - Team Identity v2.9 "
        "DEEP KIT EMBEDDING + BALANCED CLUSTERING"
    )
    print(f"Embedding device      : {encoder.device}")
    print(f"Segments embedded     : {len(assignments)}")
    print(f"Cluster counts        : {dict(counts_by_cluster)}")
    print(f"Cluster mapping       : {mapping}")
    print(f"Mapping confidence    : {mapping_confidence:.3f}")
    print(f"Overridden segments   : {len(override_segments)}")
    print(f"Overridden samples    : {stats['overridden_samples']}")

    print("Overrides:")

    if not override_segments:
        print("  NONE")
    else:
        for sid, old, new, margin in override_segments:
            print(
                f"  {sid:<8} "
                f"{old} -> {new} "
                f"| margin={float(margin):.3f}"
            )

    print(f"Video output          : {output}")
    print(f"JSONL output          : {jsonl_out}")
    print(f"Diagnostics           : {diagnostics_out}")
    print("=" * 94)


if __name__ == "__main__":
    main()
