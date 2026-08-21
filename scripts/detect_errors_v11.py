from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from core.error_detection_v1 import (
    ErrorDetectionConfig,
    ErrorDetectorV1,
    PlayerState,
    TacticalLane,
)
from core.error_event_sequence_v11 import sequence_error_events
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Error Detection v1.1 sequenced"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--team-jsonl", required=True)
    p.add_argument("--possession-jsonl", required=True)
    p.add_argument("--direction-jsonl", required=True)
    p.add_argument("--tactical-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--timeline-json", required=True)
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


def player_map(team_row):
    result = {}

    for track in team_row.get("tracks", []):
        xy = track.get("pitch_xy")
        if xy is None or len(xy) < 2:
            continue

        a = team_assignment(track)
        role = str(a.get("role", "PLAYER")).upper()
        team = str(a.get("team", "UNKNOWN"))
        tid = int(track.get("track_id", -1))

        if tid < 0 or team not in {"TEAM_A", "TEAM_B"}:
            continue

        result[tid] = {
            "state": PlayerState(
                track_id=tid,
                team=team,
                pitch_xy=(float(xy[0]), float(xy[1])),
                role=role,
            ),
            "track": track,
        }

    return result


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
                distance_m=float(
                    lane.get("distance_m", 0.0)
                    or 0.0
                ),
                blocker_track_ids=tuple(
                    int(x)
                    for x in lane.get(
                        "blocker_track_ids",
                        [],
                    )
                ),
            )
        )

    return lanes


def draw_circle(frame, track, label, colour):
    bbox = track.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        return

    x1, y1, x2, y2 = [
        int(round(float(v)))
        for v in bbox
    ]

    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))
    radius = max(
        18,
        int(
            round(
                max(x2 - x1, y2 - y1)
                * 0.65
            )
        ),
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
        label,
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
    team_rows = read_jsonl(
        resolve_project_path(args.team_jsonl)
    )
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
    timeline_out = resolve_project_path(
        args.timeline_json
    )

    frames = sorted(
        set(team_rows)
        & set(possession_rows)
        & set(direction_rows)
    )

    detector = ErrorDetectorV1(
        ErrorDetectionConfig(
            late_pressure_distance_m=4.8,
            severe_pressure_distance_m=6.5,
            unmarked_distance_m=5.5,
            severe_unmarked_distance_m=7.0,
            min_runner_forward_progress_m=5.0,
            max_runner_distance_from_ball_m=27.0,
            free_lane_min_distance_m=8.0,
            free_lane_severe_distance_m=15.0,
            dangerous_goal_distance_m=36.0,
        )
    )

    frame_events = {}
    frame_meta = {}

    # First pass: compute candidate events using V2.9 team identity.
    for frame_index in frames:
        team_row = team_rows[frame_index]
        possession = (
            possession_rows[frame_index].get(
                "possession"
            )
            or {}
        )

        possessor_id = possession.get(
            "possessor_track_id"
        )

        if possessor_id is None:
            frame_events[frame_index] = []
            continue

        # Ignore weak/pending ownership.
        possession_confidence = float(
            possession.get("confidence", 0.0)
            or 0.0
        )

        if possession_confidence < 0.45:
            frame_events[frame_index] = []
            continue

        pmap = player_map(team_row)
        owner = pmap.get(int(possessor_id))

        if owner is None:
            frame_events[frame_index] = []
            continue

        # Critical v1.1 fix:
        # derive the attacker's team from CURRENT V2.9 track identity, not the
        # old V2.5 possession team label.
        attacking_team = owner["state"].team

        # Generic tactical error logic excludes GK/referee. GK mistakes require
        # their own future goalkeeper-analysis module.
        if owner["state"].role in {
            "GOALKEEPER",
            "REFEREE",
            "OUTSIDE_PITCH",
        }:
            frame_events[frame_index] = []
            continue

        ordinary_players = [
            item["state"]
            for item in pmap.values()
            if item["state"].role
            not in {
                "GOALKEEPER",
                "REFEREE",
                "OUTSIDE_PITCH",
            }
        ]

        direction = direction_for_team(
            direction_rows[frame_index],
            attacking_team,
        )

        lanes = parse_lanes(
            tactical_rows.get(
                frame_index,
                {},
            )
        )

        # Additional v1.1 guard for runner/lane quality:
        # the V1 detector applies tighter geometry thresholds above.
        events = detector.detect(
            frame_index=frame_index,
            attacking_team=attacking_team,
            possessor_track_id=int(possessor_id),
            players=ordinary_players,
            attack_direction=direction,
            passing_lanes=lanes,
        )

        frame_events[frame_index] = events
        frame_meta[frame_index] = {
            "attacking_team": attacking_team,
            "possessor_track_id": int(possessor_id),
            "possession_confidence": possession_confidence,
            "attack_direction": direction,
        }

    timeline = sequence_error_events(
        frame_events
    )

    timeline_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timeline_out.write_text(
        json.dumps(
            [event.__dict__ for event in timeline],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Build active-event lookup. Only sequenced, sustained events are rendered.
    active_by_frame = {}

    for event in timeline:
        for frame in range(
            event.start_frame,
            event.end_frame + 1,
        ):
            active_by_frame.setdefault(
                frame,
                [],
            ).append(event)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    ) or 25.0
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    last = max(frames)

    with jsonl_out.open(
        "w",
        encoding="utf-8",
    ) as out:
        try:
            while frame_index <= last:
                ok, frame = cap.read()
                if not ok:
                    break

                team_row = team_rows.get(
                    frame_index,
                    {},
                )
                tracks = {
                    int(t.get("track_id", -1)): t
                    for t in team_row.get(
                        "tracks",
                        [],
                    )
                }

                active = active_by_frame.get(
                    frame_index,
                    [],
                )

                for event in active:
                    track = tracks.get(
                        event.primary_track_id
                    )
                    if track is None:
                        continue

                    colour = (
                        (0, 0, 255)
                        if event.severity == "HIGH"
                        else (0, 165, 255)
                    )

                    draw_circle(
                        frame,
                        track,
                        (
                            f"{event.event_id} "
                            f"{event.error_type}"
                        ),
                        colour,
                    )

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, 92),
                    (18, 18, 18),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        f"Error Detection v1.1 | "
                        f"sequenced events={len(active)}"
                    ),
                    (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )

                if active:
                    summary = " | ".join(
                        (
                            f"{e.event_id} "
                            f"{e.error_type}:"
                            f"{e.severity}"
                        )
                        for e in active[:3]
                    )
                else:
                    summary = "No sustained tactical error event"

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

                meta = frame_meta.get(
                    frame_index,
                    {},
                )

                cv2.putText(
                    frame,
                    (
                        f"attack={meta.get('attacking_team')} "
                        f"owner={meta.get('possessor_track_id')} "
                        f"dir={meta.get('attack_direction')}"
                    ),
                    (18, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (145, 145, 145),
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
                            "meta": meta,
                            "candidate_events": [
                                event.__dict__
                                for event in frame_events.get(
                                    frame_index,
                                    [],
                                )
                            ],
                            "active_sequenced_event_ids": [
                                event.event_id
                                for event in active
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

    stats = Counter(
        event.error_type
        for event in timeline
    )

    print("=" * 92)
    print("DONE - Error Detection v1.1 SEQUENCED")
    print(f"Frames processed       : {len(frames)}")
    print(f"Sequenced events       : {len(timeline)}")
    print(f"LATE_PRESSURE events   : {stats['LATE_PRESSURE']}")
    print(f"UNMARKED_RUNNER events : {stats['UNMARKED_RUNNER']}")
    print(f"FREE_PASSING_LANE evts : {stats['FREE_PASSING_LANE']}")
    print("Timeline:")
    for event in timeline:
        print(
            f"  {event.event_id} "
            f"{event.error_type:<18} "
            f"frames={event.start_frame}..{event.end_frame} "
            f"peak={event.peak_frame} "
            f"team={event.attacking_team} "
            f"severity={event.severity}"
        )
    print(f"Video output           : {output}")
    print(f"Frame JSONL            : {jsonl_out}")
    print(f"Timeline JSON          : {timeline_out}")
    print("=" * 92)


if __name__ == "__main__":
    main()
