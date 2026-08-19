from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from core.attack_direction import (
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    DirectionPlayer,
    AttackDirectionResolver,
    DefensiveLineEstimator,
)
from core.attack_direction_temporal import (
    DirectionEvidence,
    TemporalAttackDirectionResolver,
    TemporalDirectionConfig,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Attack Direction + Defensive Line v1.1 temporal consensus"
    )
    p.add_argument("--source", default=r"input\input.mp4")
    p.add_argument(
        "--team-jsonl",
        default=r"output\team_classification_v25_pnl_exact.jsonl",
    )
    p.add_argument(
        "--output",
        default=r"output\attack_direction_defline_v11.mp4",
    )
    p.add_argument(
        "--jsonl",
        default=r"output\attack_direction_defline_v11.jsonl",
    )
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--min-evidence-frames", type=int, default=8)
    p.add_argument("--min-consensus-ratio", type=float, default=0.80)
    p.add_argument("--team-a-attacks", choices=["auto", "left", "right"], default="auto")
    p.add_argument("--team-b-attacks", choices=["auto", "left", "right"], default="auto")
    return p.parse_args()


def read_jsonl(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rows[int(item["frame_index"])] = item
    return rows


def assignment(track: dict):
    return track.get("team_v25") or track.get("team_v24") or {}


def parse_players(row: dict) -> list[DirectionPlayer]:
    out = []
    for tr in row.get("tracks", []):
        xy = tr.get("pitch_xy")
        if not xy or len(xy) < 2:
            continue

        a = assignment(tr)
        team = str(a.get("team", "UNKNOWN"))
        role = str(a.get("role", "PLAYER")).upper()
        source_class = str(tr.get("class_name", "")).strip().lower()

        if team not in {TEAM_A, TEAM_B}:
            continue
        if source_class == "referee" or role in {"REFEREE", "OUTSIDE_PITCH"}:
            continue

        tid = int(tr.get("track_id", -1))
        if tid < 0:
            continue

        out.append(
            DirectionPlayer(
                track_id=tid,
                team=team,
                role=role,
                pitch_xy=(float(xy[0]), float(xy[1])),
            )
        )
    return out


def override_to_direction(value: str):
    if value == "right":
        return "PLUS_X"
    if value == "left":
        return "MINUS_X"
    return None


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    team_path = resolve_project_path(args.team_jsonl)
    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)

    for p in (source, team_path):
        if not p.exists():
            raise FileNotFoundError(p)

    rows = read_jsonl(team_path)
    frames = sorted(rows)

    if args.max_frames >= 0:
        frames = frames[:args.max_frames]

    if not frames:
        raise RuntimeError("No frames.")

    frame_resolver = AttackDirectionResolver()
    evidence = []

    override_a = override_to_direction(args.team_a_attacks)
    override_b = override_to_direction(args.team_b_attacks)

    # Prepass: collect only frame-level direct evidence.
    for frame_idx in frames:
        players = parse_players(rows[frame_idx])

        ra = frame_resolver.resolve(
            TEAM_A,
            players,
            args.team_a_attacks if args.team_a_attacks != "auto" else None,
        )
        rb = frame_resolver.resolve(
            TEAM_B,
            players,
            args.team_b_attacks if args.team_b_attacks != "auto" else None,
        )

        if ra.direction != UNKNOWN:
            evidence.append(
                DirectionEvidence(
                    TEAM_A,
                    ra.direction,
                    ra.confidence,
                    ra.source,
                    frame_idx,
                )
            )

        if rb.direction != UNKNOWN:
            evidence.append(
                DirectionEvidence(
                    TEAM_B,
                    rb.direction,
                    rb.confidence,
                    rb.source,
                    frame_idx,
                )
            )

    temporal = TemporalAttackDirectionResolver(
        TemporalDirectionConfig(
            min_evidence_frames=max(1, args.min_evidence_frames),
            min_consensus_ratio=min(1.0, max(0.5, args.min_consensus_ratio)),
            allow_opponent_inference=True,
        )
    )

    consensus = temporal.resolve_pair(evidence)

    # Explicit CLI overrides are authoritative.
    if override_a is not None:
        c = consensus[TEAM_A]
        consensus[TEAM_A] = type(c)(
            team=TEAM_A,
            direction=override_a,
            confidence=1.0,
            source="override",
            evidence_frames=c.evidence_frames,
            plus_x_frames=c.plus_x_frames,
            minus_x_frames=c.minus_x_frames,
        )

    if override_b is not None:
        c = consensus[TEAM_B]
        consensus[TEAM_B] = type(c)(
            team=TEAM_B,
            direction=override_b,
            confidence=1.0,
            source="override",
            evidence_frames=c.evidence_frames,
            plus_x_frames=c.plus_x_frames,
            minus_x_frames=c.minus_x_frames,
        )

    ca = consensus[TEAM_A]
    cb = consensus[TEAM_B]

    print("=" * 88)
    print("FootballAnalysisAI - Attack Direction v1.1 TEMPORAL CONSENSUS")
    print(
        f"TEAM_A: {ca.direction} | source={ca.source} | "
        f"evidence={ca.evidence_frames} (+x={ca.plus_x_frames}, -x={ca.minus_x_frames}) "
        f"| conf={ca.confidence:.2f}"
    )
    print(
        f"TEAM_B: {cb.direction} | source={cb.source} | "
        f"evidence={cb.evidence_frames} (+x={cb.plus_x_frames}, -x={cb.minus_x_frames}) "
        f"| conf={cb.confidence:.2f}"
    )
    print("=" * 88)

    estimator = DefensiveLineEstimator()

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_set = set(frames)
    last = max(frames)
    idx = 0

    with jsonl_out.open("w", encoding="utf-8") as out:
        try:
            while idx <= last:
                ok, frame = cap.read()
                if not ok:
                    break

                if idx not in frame_set:
                    idx += 1
                    continue

                players = parse_players(rows[idx])

                la = estimator.estimate(
                    TEAM_A,
                    players,
                    ca.direction,
                )
                lb = estimator.estimate(
                    TEAM_B,
                    players,
                    cb.direction,
                )

                cv2.rectangle(frame, (0, 0), (w, 112), (18, 18, 18), -1)

                cv2.putText(
                    frame,
                    (
                        f"Attack Direction v1.1 | "
                        f"A={ca.direction} ({ca.source}) | "
                        f"B={cb.direction} ({cb.source})"
                    ),
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"A line x={la.line_x:.1f}m conf={la.confidence:.2f}"
                        if la.line_x is not None
                        else "A defensive line: UNKNOWN"
                    ),
                    (18, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 140, 90),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"B line x={lb.line_x:.1f}m conf={lb.confidence:.2f}"
                        if lb.line_x is not None
                        else "B defensive line: UNKNOWN"
                    ),
                    (18, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (100, 130, 255),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(frame)

                out.write(
                    json.dumps(
                        {
                            "frame_index": idx,
                            "attack_direction": {
                                TEAM_A: ca.__dict__,
                                TEAM_B: cb.__dict__,
                            },
                            "defensive_line": {
                                TEAM_A: la.__dict__,
                                TEAM_B: lb.__dict__,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                idx += 1

        finally:
            cap.release()
            writer.release()

    print("DONE - Attack Direction + Defensive Line v1.1")
    print(f"Video output: {output}")
    print(f"JSONL output: {jsonl_out}")


if __name__ == "__main__":
    main()
