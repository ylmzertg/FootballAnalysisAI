from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from core.pass_options_ranking_v1 import (
    BEST,
    BLOCKED,
    GOOD,
    RISKY,
    PassLane,
    PassOptionsRankerV1,
    PassPlayer,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Pass Options Ranking v1"
    )

    p.add_argument("--source", required=True)
    p.add_argument("--team-jsonl", required=True)
    p.add_argument("--possession-jsonl", required=True)
    p.add_argument("--direction-jsonl", required=True)
    p.add_argument("--tactical-jsonl", required=True)

    p.add_argument("--output", required=True)
    p.add_argument("--jsonl", required=True)

    p.add_argument(
        "--draw-options",
        type=int,
        default=3,
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
        or track.get("team_v28")
        or track.get("team_v27")
        or track.get("team_v26")
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
        pitch = track.get(
            "pitch_xy"
        )

        if (
            pitch is None
            or len(pitch) < 2
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
            PassPlayer(
                track_id=tid,
                team=team,
                pitch_xy=(
                    float(pitch[0]),
                    float(pitch[1]),
                ),
                role=role,
            )
        )

    return players


def parse_lanes(tactical_row):
    lanes = []

    for lane in tactical_row.get(
        "passing_lanes",
        [],
    ):
        lanes.append(
            PassLane(
                receiver_track_id=int(
                    lane.get(
                        "receiver_track_id",
                        -1,
                    )
                ),
                status=str(
                    lane.get(
                        "status",
                        "UNKNOWN",
                    )
                ),
                distance_m=float(
                    lane.get(
                        "distance_m",
                        0.0,
                    )
                    or 0.0
                ),
                nearest_blocker_clearance_m=(
                    float(
                        lane.get(
                            "nearest_blocker_clearance_m"
                        )
                    )
                    if lane.get(
                        "nearest_blocker_clearance_m"
                    )
                    is not None
                    else None
                ),
                blocker_track_ids=tuple(
                    int(x)
                    for x in lane.get(
                        "blocker_track_ids",
                        [],
                    )
                ),
                tactical_score=float(
                    lane.get(
                        "score",
                        0.0,
                    )
                    or 0.0
                ),
            )
        )

    return lanes


def direction_for(
    direction_row,
    team,
):
    ad = (
        direction_row.get(
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


def foot_xy(track):
    bbox = track.get(
        "bbox_xyxy"
    )

    if (
        bbox is None
        or len(bbox) != 4
    ):
        return None

    x1, y1, x2, y2 = [
        float(v)
        for v in bbox
    ]

    return (
        int(
            round(
                (x1 + x2) / 2.0
            )
        ),
        int(
            round(y2)
        ),
    )


def main():
    args = parse_args()

    source = resolve_project_path(
        args.source
    )

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

    tactical_rows = read_jsonl(
        resolve_project_path(
            args.tactical_jsonl
        )
    )

    output = resolve_project_path(
        args.output
    )
    jsonl_out = resolve_project_path(
        args.jsonl
    )

    common = sorted(
        set(team_rows)
        & set(possession_rows)
        & set(direction_rows)
        & set(tactical_rows)
    )

    ranker = PassOptionsRankerV1()

    cap = cv2.VideoCapture(
        str(source)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    ) or 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    common_set = set(
        common
    )
    last = max(
        common
    )
    frame_index = 0

    stats = Counter()

    category_colours = {
        BEST: (60, 240, 60),
        GOOD: (80, 220, 190),
        RISKY: (0, 200, 255),
        BLOCKED: (70, 70, 220),
    }

    with jsonl_out.open(
        "w",
        encoding="utf-8",
    ) as out:
        try:
            while (
                frame_index <= last
            ):
                ok, frame = (
                    cap.read()
                )

                if not ok:
                    break

                if (
                    frame_index
                    not in common_set
                ):
                    writer.write(
                        frame
                    )
                    frame_index += 1
                    continue

                team_row = (
                    team_rows[
                        frame_index
                    ]
                )

                possession = (
                    possession_rows[
                        frame_index
                    ].get(
                        "possession"
                    )
                    or {}
                )

                tactical_row = (
                    tactical_rows[
                        frame_index
                    ]
                )

                possessor_id = (
                    possession.get(
                        "possessor_track_id"
                    )
                )

                players = parse_players(
                    team_row
                )

                player_by_id = {
                    player.track_id:
                    player
                    for player in players
                }

                owner = (
                    player_by_id.get(
                        int(
                            possessor_id
                        )
                    )
                    if possessor_id
                    is not None
                    else None
                )

                options = []
                direction = "UNKNOWN"

                if (
                    owner is not None
                    and owner.role
                    not in {
                        "GOALKEEPER",
                        "REFEREE",
                        "OUTSIDE_PITCH",
                    }
                ):
                    direction = (
                        direction_for(
                            direction_rows[
                                frame_index
                            ],
                            owner.team,
                        )
                    )

                    options = ranker.rank(
                        possessor=owner,
                        players=players,
                        lanes=parse_lanes(
                            tactical_row
                        ),
                        attack_direction=direction,
                    )

                track_map = {
                    int(
                        track.get(
                            "track_id",
                            -1,
                        )
                    ): track
                    for track in team_row.get(
                        "tracks",
                        [],
                    )
                }

                if owner is not None:
                    owner_track = track_map.get(
                        owner.track_id
                    )

                    start = (
                        foot_xy(
                            owner_track
                        )
                        if owner_track
                        is not None
                        else None
                    )

                    if start is not None:
                        for option in options[
                            : max(
                                1,
                                args.draw_options,
                            )
                        ]:
                            receiver_track = (
                                track_map.get(
                                    option.receiver_track_id
                                )
                            )

                            end = (
                                foot_xy(
                                    receiver_track
                                )
                                if receiver_track
                                is not None
                                else None
                            )

                            if end is None:
                                continue

                            colour = (
                                category_colours[
                                    option.category
                                ]
                            )

                            thickness = (
                                4
                                if option.category
                                == BEST
                                else (
                                    2
                                    if option.category
                                    in {
                                        GOOD,
                                        RISKY,
                                    }
                                    else 1
                                )
                            )

                            cv2.arrowedLine(
                                frame,
                                start,
                                end,
                                colour,
                                thickness,
                                cv2.LINE_AA,
                                tipLength=0.08,
                            )

                            cv2.putText(
                                frame,
                                (
                                    f"{option.rank}. "
                                    f"{option.category} "
                                    f"{option.score:.2f}"
                                ),
                                (
                                    end[0] + 6,
                                    max(
                                        18,
                                        end[1] - 8,
                                    ),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.42,
                                colour,
                                2,
                                cv2.LINE_AA,
                            )

                for option in options:
                    stats[
                        option.category
                    ] += 1

                if options:
                    stats[
                        "frames_with_options"
                    ] += 1

                cv2.rectangle(
                    frame,
                    (
                        0,
                        0,
                    ),
                    (
                        width,
                        92,
                    ),
                    (
                        18,
                        18,
                        18,
                    ),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        f"Pass Options Ranking v1 | "
                        f"owner={possessor_id} | "
                        f"dir={direction}"
                    ),
                    (
                        18,
                        28,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (
                        235,
                        235,
                        235,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                top_summary = (
                    " | ".join(
                        (
                            f"{option.rank}:"
                            f"ID{option.receiver_track_id} "
                            f"{option.category} "
                            f"{option.score:.2f}"
                        )
                        for option
                        in options[:3]
                    )
                    if options
                    else "No ranked pass option"
                )

                cv2.putText(
                    frame,
                    top_summary,
                    (
                        18,
                        58,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (
                        180,
                        220,
                        190,
                    ),
                    1,
                    cv2.LINE_AA,
                )

                if options:
                    cv2.putText(
                        frame,
                        options[0].explanation[
                            :150
                        ],
                        (
                            18,
                            82,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (
                            150,
                            150,
                            150,
                        ),
                        1,
                        cv2.LINE_AA,
                    )

                writer.write(
                    frame
                )

                out.write(
                    json.dumps(
                        {
                            "frame_index": (
                                frame_index
                            ),
                            "timestamp_seconds": (
                                round(
                                    frame_index
                                    / fps,
                                    5,
                                )
                            ),
                            "possessor_track_id": (
                                possessor_id
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
                            "options": [
                                option.__dict__
                                for option
                                in options
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                frame_index += 1

        finally:
            cap.release()
            writer.release()

    print("=" * 90)
    print(
        "DONE - Pass Options Ranking v1"
    )
    print(
        f"Frames processed      : "
        f"{len(common)}"
    )
    print(
        f"Frames with options   : "
        f"{stats['frames_with_options']}"
    )
    print(
        f"BEST                  : "
        f"{stats[BEST]}"
    )
    print(
        f"GOOD                  : "
        f"{stats[GOOD]}"
    )
    print(
        f"RISKY                 : "
        f"{stats[RISKY]}"
    )
    print(
        f"BLOCKED               : "
        f"{stats[BLOCKED]}"
    )
    print(
        f"Video output          : "
        f"{output}"
    )
    print(
        f"JSONL output          : "
        f"{jsonl_out}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
