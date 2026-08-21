from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from core.error_detection_v1 import (
    ErrorDetectorV1,
    PlayerState,
    TacticalLane,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Error Detection v1"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--team-jsonl", required=True)
    p.add_argument("--possession-jsonl", required=True)
    p.add_argument("--direction-jsonl", required=True)
    p.add_argument("--tactical-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--jsonl", required=True)
    return p.parse_args()


def read_jsonl(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["frame_index"])] = row
    return rows


def team_assignment(track):
    return (
        track.get("team_v29")
        or track.get("team_v25")
        or track.get("team_v24")
        or {}
    )


def parse_players(row):
    players = []
    for track in row.get("tracks", []):
        xy = track.get("pitch_xy")
        if xy is None or len(xy) < 2:
            continue

        a = team_assignment(track)
        team = str(a.get("team", "UNKNOWN"))
        role = str(a.get("role", "PLAYER")).upper()

        if team not in {"TEAM_A", "TEAM_B"}:
            continue

        players.append(
            PlayerState(
                track_id=int(track.get("track_id", -1)),
                team=team,
                pitch_xy=(float(xy[0]), float(xy[1])),
                role=role,
            )
        )
    return players


def direction_for_team(direction_row, team):
    ad = direction_row.get("attack_direction") or {}
    return str(
        (ad.get(team) or {}).get("direction", "UNKNOWN")
    )


def parse_lanes(tactical_row):
    lanes = []
    for lane in tactical_row.get("passing_lanes", []):
        lanes.append(
            TacticalLane(
                receiver_track_id=int(
                    lane.get("receiver_track_id", -1)
                ),
                status=str(lane.get("status", "UNKNOWN")),
                distance_m=float(lane.get("distance_m", 0.0) or 0.0),
                blocker_track_ids=tuple(
                    int(x)
                    for x in lane.get("blocker_track_ids", [])
                ),
            )
        )
    return lanes


def draw_event(frame, event, team_row):
    track_map = {
        int(track.get("track_id", -1)): track
        for track in team_row.get("tracks", [])
    }

    track = track_map.get(event.primary_track_id)
    if track is None:
        return

    bbox = track.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        return

    x1, y1, x2, y2 = [
        int(round(float(v)))
        for v in bbox
    ]

    colour = (
        (0, 0, 255)
        if event.severity == "HIGH"
        else (0, 165, 255)
    )

    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))
    radius = max(
        18,
        int(round(max(x2 - x1, y2 - y1) * 0.65)),
    )

    cv2.circle(
        frame,
        (cx, cy),
        radius,
        colour,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        event.error_type,
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        colour,
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    team_rows = read_jsonl(resolve_project_path(args.team_jsonl))
    possession_rows = read_jsonl(
        resolve_project_path(args.possession_jsonl)
    )
    direction_rows = read_jsonl(
        resolve_project_path(args.direction_jsonl)
    )
    tactical_rows = read_jsonl(
        resolve_project_path(args.tactical_jsonl)
    )

    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)

    frames = sorted(
        set(team_rows)
        & set(possession_rows)
        & set(direction_rows)
    )

    detector = ErrorDetectorV1()

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_set = set(frames)
    last = max(frames)
    frame_index = 0
    stats = Counter()

    with jsonl_out.open("w", encoding="utf-8") as out:
        try:
            while frame_index <= last:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index not in frame_set:
                    writer.write(frame)
                    frame_index += 1
                    continue

                team_row = team_rows[frame_index]
                possession_row = possession_rows[frame_index]
                direction_row = direction_rows[frame_index]
                tactical_row = tactical_rows.get(frame_index, {})

                possession = possession_row.get("possession") or {}
                attacking_team = str(
                    possession.get(
                        "possessor_team",
                        possession.get("state", "UNKNOWN"),
                    )
                )
                possessor_id = possession.get("possessor_track_id")

                direction = direction_for_team(
                    direction_row,
                    attacking_team,
                )

                events = detector.detect(
                    frame_index=frame_index,
                    attacking_team=attacking_team,
                    possessor_track_id=(
                        int(possessor_id)
                        if possessor_id is not None
                        else None
                    ),
                    players=parse_players(team_row),
                    attack_direction=direction,
                    passing_lanes=parse_lanes(tactical_row),
                )

                for event in events:
                    stats[event.error_type] += 1
                    draw_event(frame, event, team_row)

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, 86),
                    (18, 18, 18),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        f"Error Detection v1 | "
                        f"attack={attacking_team} | "
                        f"errors={len(events)}"
                    ),
                    (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )

                summary = (
                    " | ".join(
                        f"{event.error_type}:{event.severity}"
                        for event in events[:3]
                    )
                    if events
                    else "No candidate tactical error"
                )

                cv2.putText(
                    frame,
                    summary,
                    (18, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (180, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(frame)

                out.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": round(
                                frame_index / fps,
                                5,
                            ),
                            "attacking_team": attacking_team,
                            "possessor_track_id": possessor_id,
                            "attack_direction": direction,
                            "events": [
                                event.__dict__
                                for event in events
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

    print("=" * 88)
    print("DONE - Error Detection v1")
    print(f"Frames processed       : {len(frames)}")
    print(f"LATE_PRESSURE          : {stats['LATE_PRESSURE']}")
    print(f"UNMARKED_RUNNER        : {stats['UNMARKED_RUNNER']}")
    print(f"FREE_PASSING_LANE      : {stats['FREE_PASSING_LANE']}")
    print(f"Video output           : {output}")
    print(f"JSONL output           : {jsonl_out}")
    print("=" * 88)


if __name__ == "__main__":
    main()
