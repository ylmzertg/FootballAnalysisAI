from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.goalkeeper_team_resolver import (
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    ResolverPlayer,
    GoalkeeperTeamResolver,
    GoalkeeperTeamResolverConfig,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Goalkeeper Team Resolver v1"
    )
    p.add_argument(
        "--team-jsonl",
        default=r"output\team_classification_v25_pnl_exact.jsonl",
    )
    p.add_argument(
        "--output-json",
        default=r"output\goalkeeper_team_resolver_v1.json",
    )
    p.add_argument("--neighbor-radius-m", type=float, default=28.0)
    p.add_argument("--min-neighbors", type=int, default=2)
    p.add_argument("--min-evidence-frames", type=int, default=4)
    return p.parse_args()


def assignment(track):
    return track.get("team_v25") or track.get("team_v24") or {}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_players(row):
    out = []

    for tr in row.get("tracks", []):
        xy = tr.get("pitch_xy")
        if not xy or len(xy) < 2:
            continue

        a = assignment(tr)
        team = str(a.get("team", UNKNOWN))
        role = str(a.get("role", "PLAYER")).upper()
        source_class = str(tr.get("class_name", "")).strip().lower()

        if source_class == "referee" or role in {"REFEREE", "OUTSIDE_PITCH"}:
            continue

        tid = int(tr.get("track_id", -1))
        if tid < 0:
            continue

        out.append(
            ResolverPlayer(
                track_id=tid,
                team=team,
                role=role,
                pitch_xy=(float(xy[0]), float(xy[1])),
            )
        )

    return out


def main():
    args = parse_args()

    team_path = resolve_project_path(args.team_jsonl)
    output = resolve_project_path(args.output_json)

    if not team_path.exists():
        raise FileNotFoundError(team_path)

    rows = read_jsonl(team_path)

    resolver = GoalkeeperTeamResolver(
        GoalkeeperTeamResolverConfig(
            neighbor_radius_m=max(5.0, args.neighbor_radius_m),
            min_neighbors=max(1, args.min_neighbors),
            min_evidence_frames=max(1, args.min_evidence_frames),
        )
    )

    evidence = []

    for row in rows:
        frame_index = int(row["frame_index"])
        players = parse_players(row)

        gks = [
            p for p in players
            if p.role == "GOALKEEPER"
        ]

        for gk in gks:
            e = resolver.frame_evidence(
                frame_index,
                gk,
                players,
            )
            if e is not None:
                evidence.append(e)

    consensus = resolver.consensus(evidence)

    payload = {
        "evidence": [e.__dict__ for e in evidence],
        "consensus": {
            str(k): v.__dict__
            for k, v in consensus.items()
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 84)
    print("FootballAnalysisAI - Goalkeeper Team Resolver v1")

    if not consensus:
        print("No goalkeeper team consensus.")
    else:
        for gk_id, c in sorted(consensus.items()):
            print(
                f"GK ID {gk_id:>3} | team={c.resolved_team} "
                f"| evidence={c.evidence_frames} "
                f"(A={c.team_a_frames}, B={c.team_b_frames}) "
                f"| median_x={c.median_goalkeeper_x:.2f}m "
                f"| conf={c.confidence:.2f}"
            )

            if c.resolved_team in {TEAM_A, TEAM_B}:
                attack_direction = (
                    "PLUS_X"
                    if c.median_goalkeeper_x < 52.5
                    else "MINUS_X"
                )
                print(
                    f"          => {c.resolved_team} attacks {attack_direction}"
                )

    print(f"Output: {output}")
    print("=" * 84)


if __name__ == "__main__":
    main()
