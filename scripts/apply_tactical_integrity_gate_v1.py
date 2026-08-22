from __future__ import annotations

import argparse
import json
from pathlib import Path


TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"

BEST = "BEST"
GOOD = "GOOD"
RISKY = "RISKY"
BLOCKED = "BLOCKED"


def assignment(track):
    return (
        track.get("team_v31")
        or track.get("team_v3")
        or track.get("team_v29")
        or track.get("team_v28")
        or track.get("team_v27")
        or track.get("team_v26")
        or track.get("team_v25")
        or track.get("team_v24")
        or {}
    )


def read_jsonl(path):
    rows = {}

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            rows[int(row["frame_index"])] = row

    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "FootballAnalysisAI - Tactical Integrity Gate V1"
        )
    )

    p.add_argument("--team-jsonl", required=True)
    p.add_argument("--pass-options-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)

    p.add_argument(
        "--best-min-score",
        type=float,
        default=0.66,
    )
    p.add_argument(
        "--best-min-margin",
        type=float,
        default=0.06,
    )
    p.add_argument(
        "--min-best-space",
        type=float,
        default=2.4,
    )
    p.add_argument(
        "--min-best-clearance",
        type=float,
        default=2.0,
    )

    return p.parse_args()


def main():
    args = parse_args()

    team_rows = read_jsonl(args.team_jsonl)
    pass_rows = read_jsonl(args.pass_options_jsonl)

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "frames": 0,
        "options_in": 0,
        "options_out": 0,
        "team_mismatch_dropped": 0,
        "best_demoted_margin": 0,
        "best_demoted_space": 0,
        "best_demoted_clearance": 0,
        "best_kept": 0,
    }

    with output.open("w", encoding="utf-8") as out:
        for frame_index in sorted(pass_rows):
            row = dict(pass_rows[frame_index])
            team_row = team_rows.get(frame_index, {})

            stats["frames"] += 1

            possessor_id = row.get("possessor_track_id")
            possessor_team = str(
                row.get("possessor_team") or "UNKNOWN"
            )

            team_by_id = {}

            for track in team_row.get("tracks", []):
                tid = int(track.get("track_id", -1))
                a = assignment(track)

                if tid >= 0:
                    team_by_id[tid] = str(
                        a.get("team", "UNKNOWN")
                    )

            gated = []
            gate_events = []

            for option in row.get("options", []):
                stats["options_in"] += 1

                item = dict(option)

                receiver_id = int(
                    item.get("receiver_track_id", -1)
                )
                receiver_team = team_by_id.get(
                    receiver_id,
                    "UNKNOWN",
                )

                # HARD GUARD 1:
                # Receiver must belong to possessor's team.
                if (
                    possessor_team not in {TEAM_A, TEAM_B}
                    or receiver_team != possessor_team
                ):
                    stats["team_mismatch_dropped"] += 1

                    gate_events.append({
                        "receiver_track_id": receiver_id,
                        "action": "DROP",
                        "reason": "TEAM_MISMATCH",
                    })
                    continue

                category = str(
                    item.get("category", RISKY)
                ).upper()

                score = float(
                    item.get("score", 0.0) or 0.0
                )

                receiver_space = item.get(
                    "receiver_space_m"
                )
                lane_clearance = item.get(
                    "lane_clearance_m"
                )

                # BLOCKED can never be promoted.
                if category == BLOCKED:
                    item["category"] = BLOCKED
                    gated.append(item)
                    continue

                # HARD GUARD 2:
                # Very tight receiver => cannot be BEST.
                if (
                    category == BEST
                    and receiver_space is not None
                    and float(receiver_space)
                    < args.min_best_space
                ):
                    item["category"] = (
                        GOOD
                        if score >= 0.50
                        else RISKY
                    )

                    stats["best_demoted_space"] += 1

                    gate_events.append({
                        "receiver_track_id": receiver_id,
                        "action": "DEMOTE",
                        "reason": "TIGHT_RECEIVER_SPACE",
                    })

                # HARD GUARD 3:
                # Narrow passing lane => cannot be BEST.
                if (
                    item["category"] == BEST
                    and lane_clearance is not None
                    and float(lane_clearance)
                    < args.min_best_clearance
                ):
                    item["category"] = (
                        GOOD
                        if score >= 0.50
                        else RISKY
                    )

                    stats["best_demoted_clearance"] += 1

                    gate_events.append({
                        "receiver_track_id": receiver_id,
                        "action": "DEMOTE",
                        "reason": "NARROW_PASSING_LANE",
                    })

                gated.append(item)

            # BEST confidence / separation guard.
            best_indices = [
                i
                for i, item in enumerate(gated)
                if str(item.get("category", "")).upper()
                == BEST
            ]

            if best_indices:
                best_i = best_indices[0]
                best = gated[best_i]

                best_score = float(
                    best.get("score", 0.0) or 0.0
                )

                competing_scores = [
                    float(
                        item.get("score", 0.0)
                        or 0.0
                    )
                    for i, item in enumerate(gated)
                    if (
                        i != best_i
                        and str(
                            item.get(
                                "category",
                                "",
                            )
                        ).upper()
                        != BLOCKED
                    )
                ]

                second_score = (
                    max(competing_scores)
                    if competing_scores
                    else 0.0
                )

                margin = best_score - second_score

                if (
                    best_score < args.best_min_score
                    or margin < args.best_min_margin
                ):
                    best["category"] = (
                        GOOD
                        if best_score >= 0.50
                        else RISKY
                    )

                    stats["best_demoted_margin"] += 1

                    gate_events.append({
                        "receiver_track_id": int(
                            best.get(
                                "receiver_track_id",
                                -1,
                            )
                        ),
                        "action": "DEMOTE",
                        "reason": "BEST_NOT_DECISIVE",
                        "best_score": round(
                            best_score,
                            5,
                        ),
                        "second_score": round(
                            second_score,
                            5,
                        ),
                        "margin": round(
                            margin,
                            5,
                        ),
                    })
                else:
                    stats["best_kept"] += 1

            # Exactly one BEST maximum.
            seen_best = False

            for item in gated:
                if (
                    str(
                        item.get("category", "")
                    ).upper()
                    == BEST
                ):
                    if seen_best:
                        item["category"] = GOOD
                    else:
                        seen_best = True

            for rank, item in enumerate(
                gated,
                start=1,
            ):
                item["rank"] = rank

            stats["options_out"] += len(gated)

            row["options"] = gated
            row["tactical_integrity_gate"] = {
                "version": "V1",
                "status": (
                    "ADJUSTED"
                    if gate_events
                    else "PASS"
                ),
                "events": gate_events,
                "best_min_score": (
                    args.best_min_score
                ),
                "best_min_margin": (
                    args.best_min_margin
                ),
                "min_best_space_m": (
                    args.min_best_space
                ),
                "min_best_clearance_m": (
                    args.min_best_clearance
                ),
            }

            out.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 72)
    print("DONE - Tactical Integrity Gate V1")

    for key, value in stats.items():
        print(f"{key:28}: {value}")

    print(f"Output                      : {output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
