from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from core.analyst_incident_v1 import (
    ErrorTimelineItem,
    build_incidents,
)
from core.marking_analysis_v1 import (
    MarkingAnalyzerV1,
    MarkingPlayer,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "FootballAnalysisAI - Marking + Analyst Incident v1"
        )
    )

    p.add_argument(
        "--team-jsonl",
        required=True,
    )
    p.add_argument(
        "--possession-jsonl",
        required=True,
    )
    p.add_argument(
        "--direction-jsonl",
        required=True,
    )
    p.add_argument(
        "--errors-timeline-json",
        required=True,
    )
    p.add_argument(
        "--pass-options-jsonl",
        required=True,
    )
    p.add_argument(
        "--shot-jsonl",
        required=True,
    )

    p.add_argument(
        "--marking-jsonl",
        required=True,
    )
    p.add_argument(
        "--incidents-json",
        required=True,
    )

    return p.parse_args()


def read_jsonl(path):
    rows = {}

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            rows[int(row["frame_index"])] = row

    return rows


def assignment(track):
    return (
        track.get("team_v29")
        or track.get("team_v25")
        or track.get("team_v24")
        or {}
    )


def parse_players(team_row):
    players = []

    for track in team_row.get(
        "tracks",
        [],
    ):
        xy = track.get(
            "pitch_xy"
        )

        if (
            xy is None
            or len(xy) < 2
        ):
            continue

        a = assignment(
            track
        )

        team = str(
            a.get(
                "team",
                "UNKNOWN",
            )
        )
        role = str(
            a.get(
                "role",
                "PLAYER",
            )
        ).upper()
        tid = int(
            track.get(
                "track_id",
                -1,
            )
        )

        if (
            tid < 0
            or team not in {
                "TEAM_A",
                "TEAM_B",
            }
            or role in {
                "REFEREE",
                "OUTSIDE_PITCH",
            }
        ):
            continue

        players.append(
            MarkingPlayer(
                track_id=tid,
                team=team,
                pitch_xy=(
                    float(xy[0]),
                    float(xy[1]),
                ),
                role=role,
            )
        )

    return players


def attack_direction(
    row,
    team,
):
    ad = (
        row.get(
            "attack_direction"
        )
        or {}
    )

    return str(
        (
            ad.get(team)
            or {}
        ).get(
            "direction",
            "UNKNOWN",
        )
    )


def main():
    args = parse_args()

    team_rows = read_jsonl(
        resolve_project_path(
            args.team_jsonl
        )
    )

    possession_rows = read_jsonl(
        resolve_project_path(
            args.possession_jsonl
        )
    )

    direction_rows = read_jsonl(
        resolve_project_path(
            args.direction_jsonl
        )
    )

    pass_rows = read_jsonl(
        resolve_project_path(
            args.pass_options_jsonl
        )
    )

    shot_rows = read_jsonl(
        resolve_project_path(
            args.shot_jsonl
        )
    )

    errors_payload = json.loads(
        resolve_project_path(
            args.errors_timeline_json
        ).read_text(
            encoding="utf-8"
        )
    )

    marking_out = resolve_project_path(
        args.marking_jsonl
    )

    incidents_out = resolve_project_path(
        args.incidents_json
    )

    marking_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    incidents_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analyzer = MarkingAnalyzerV1()

    marking_by_frame = {}

    common = sorted(
        set(team_rows)
        & set(possession_rows)
        & set(direction_rows)
    )

    with marking_out.open(
        "w",
        encoding="utf-8",
    ) as out:
        for frame_index in common:
            players = parse_players(
                team_rows[
                    frame_index
                ]
            )

            by_id = {
                p.track_id: p
                for p in players
            }

            possession = (
                possession_rows[
                    frame_index
                ].get(
                    "possession"
                )
                or {}
            )

            owner_id = (
                possession.get(
                    "possessor_track_id"
                )
            )

            if owner_id is None:
                assessments = []
                owner = None
                direction = "UNKNOWN"

            else:
                owner = by_id.get(
                    int(owner_id)
                )

                if (
                    owner is None
                    or owner.role
                    in {
                        "GOALKEEPER",
                        "REFEREE",
                        "OUTSIDE_PITCH",
                    }
                ):
                    assessments = []
                    direction = "UNKNOWN"

                else:
                    direction = (
                        attack_direction(
                            direction_rows[
                                frame_index
                            ],
                            owner.team,
                        )
                    )

                    assessments = (
                        analyzer.assess(
                            possessor=owner,
                            players=players,
                            attack_direction=direction,
                        )
                    )

            data = [
                a.__dict__
                for a in assessments
            ]

            marking_by_frame[
                frame_index
            ] = data

            out.write(
                json.dumps(
                    {
                        "frame_index": (
                            frame_index
                        ),
                        "possessor_track_id": (
                            owner.track_id
                            if owner
                            is not None
                            else None
                        ),
                        "possessor_team": (
                            owner.team
                            if owner
                            is not None
                            else None
                        ),
                        "attack_direction": (
                            direction
                        ),
                        "marking": data,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    pass_options_by_frame = {
        frame: row.get(
            "options",
            [],
        )
        for frame, row in (
            pass_rows.items()
        )
    }

    shot_frames = {
        frame
        for frame, row in (
            shot_rows.items()
        )
        if row.get(
            "phase_v16",
            row.get(
                "phase",
                "",
            ),
        )
        in {
            "SHOT_FLIGHT",
            "GOAL_ATTEMPT",
        }
    }

    errors = [
        ErrorTimelineItem(
            event_id=str(
                item.get(
                    "event_id"
                )
            ),
            error_type=str(
                item.get(
                    "error_type"
                )
            ),
            attacking_team=str(
                item.get(
                    "attacking_team"
                )
            ),
            defending_team=str(
                item.get(
                    "defending_team"
                )
            ),
            start_frame=int(
                item.get(
                    "start_frame"
                )
            ),
            end_frame=int(
                item.get(
                    "end_frame"
                )
            ),
            peak_frame=int(
                item.get(
                    "peak_frame"
                )
            ),
            severity=str(
                item.get(
                    "severity"
                )
            ),
            primary_track_id=(
                int(
                    item.get(
                        "primary_track_id"
                    )
                )
                if item.get(
                    "primary_track_id"
                )
                is not None
                else None
            ),
            secondary_track_id=(
                int(
                    item.get(
                        "secondary_track_id"
                    )
                )
                if item.get(
                    "secondary_track_id"
                )
                is not None
                else None
            ),
            evidence=(
                item.get(
                    "evidence"
                )
                or {}
            ),
        )
        for item in errors_payload
    ]

    incidents = build_incidents(
        errors=errors,
        pass_options_by_frame=(
            pass_options_by_frame
        ),
        marking_by_frame=(
            marking_by_frame
        ),
        shot_frames=shot_frames,
    )

    incidents_out.write_text(
        json.dumps(
            [
                incident.__dict__
                for incident
                in incidents
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 94)
    print(
        "DONE - Marking Analysis + "
        "Analyst Incident v1"
    )
    print(
        f"Marking frames        : "
        f"{len(common)}"
    )
    print(
        f"Error timeline events : "
        f"{len(errors)}"
    )
    print(
        f"Analyst incidents     : "
        f"{len(incidents)}"
    )

    print("Incidents:")

    for incident in incidents:
        print(
            f"  {incident.incident_id} "
            f"frames={incident.start_frame}.."
            f"{incident.end_frame} "
            f"peak={incident.peak_frame} "
            f"attack={incident.attacking_team} "
            f"errors={','.join(incident.error_types)} "
            f"attack_merit="
            f"{incident.attack_merit_level} "
            f"def_vulnerability="
            f"{incident.defense_vulnerability_level} "
            f"shot={incident.shot_detected}"
        )

    print(
        f"Marking JSONL         : "
        f"{marking_out}"
    )
    print(
        f"Incidents JSON        : "
        f"{incidents_out}"
    )
    print("=" * 94)


if __name__ == "__main__":
    main()
