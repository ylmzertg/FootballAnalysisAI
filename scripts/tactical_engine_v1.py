
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from core.tactical_engine import (
    TEAM_A,
    TEAM_B,
    TacticalConfig,
    TacticalEngine,
    TacticalPlayer,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Pressure + Passing Lanes Tactical Engine v1"
    )
    p.add_argument("--source", default=r"input\input.mp4")
    p.add_argument(
        "--team-jsonl",
        default=r"output\team_classification_v24_pnl_exact.jsonl",
    )
    p.add_argument(
        "--possession-jsonl",
        default=r"output\possession_v1.jsonl",
    )
    p.add_argument("--output", default=r"output\tactical_v1.mp4")
    p.add_argument("--jsonl", default=r"output\tactical_v1.jsonl")
    p.add_argument("--max-frames", type=int, default=-1)

    p.add_argument("--pressure-high-m", type=float, default=2.0)
    p.add_argument("--pressure-medium-m", type=float, default=3.5)
    p.add_argument("--lane-half-width-m", type=float, default=1.25)
    p.add_argument("--max-pass-distance-m", type=float, default=35.0)
    p.add_argument("--draw-open-lanes", type=int, default=4)
    return p.parse_args()


def read_jsonl(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rows[int(item["frame_index"])] = item
    return rows


def parse_players(team_row: dict) -> list[TacticalPlayer]:
    out = []

    for tr in team_row.get("tracks", []):
        pitch = tr.get("pitch_xy")
        if not pitch or len(pitch) < 2:
            continue

        assignment = tr.get("team_v24") or {}
        team = str(assignment.get("team", "UNKNOWN"))
        role = str(assignment.get("role", "PLAYER")).upper()

        # Hard role guard: detector/source role has veto power over a noisy
        # temporal team-classification assignment. A detector-labelled referee
        # must never become a passing option or team-shape player.
        source_class = str(tr.get("class_name", "")).strip().lower()
        source_role = str(tr.get("role_hint", "")).strip().upper()

        if team not in {TEAM_A, TEAM_B}:
            continue

        if (
            role in {"REFEREE", "OUTSIDE_PITCH"}
            or source_class == "referee"
            or source_role == "REFEREE"
        ):
            continue

        tid = int(tr.get("track_id", -1))
        if tid < 0:
            continue

        out.append(
            TacticalPlayer(
                track_id=tid,
                team=team,
                role=role,
                pitch_xy=(float(pitch[0]), float(pitch[1])),
            )
        )

    return out


def frame_track_map(team_row: dict):
    result = {}

    for tr in team_row.get("tracks", []):
        tid = int(tr.get("track_id", -1))
        if tid < 0:
            continue
        result[tid] = tr

    return result


def foot_image_xy(track: dict):
    bbox = track.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (int(round((x1 + x2) / 2.0)), int(round(y2)))


def draw_pitch_panel(
    frame,
    players,
    possessor,
    pressure,
    lanes,
):
    panel_w = min(420, max(300, frame.shape[1] // 3))
    panel_h = int(round(panel_w * 68.0 / 105.0))
    margin = 14

    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (38, 104, 45)

    def p(x, y):
        px = margin + int(round((x / 105.0) * (panel_w - 2 * margin)))
        py = margin + int(round((y / 68.0) * (panel_h - 2 * margin)))
        return px, py

    white = (230, 230, 230)

    cv2.rectangle(
        panel,
        p(0, 0),
        p(105, 68),
        white,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        panel,
        p(52.5, 0),
        p(52.5, 68),
        white,
        1,
        cv2.LINE_AA,
    )

    # Passing lanes.
    receiver_map = {pl.track_id: pl for pl in players}

    for lane in lanes:
        receiver = receiver_map.get(lane.receiver_track_id)
        if receiver is None:
            continue

        color = (
            (80, 230, 80)
            if lane.status == "OPEN"
            else (70, 70, 230)
        )
        thickness = 2 if lane.status == "OPEN" else 1

        cv2.line(
            panel,
            p(*possessor.pitch_xy),
            p(*receiver.pitch_xy),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Players.
    for pl in players:
        color = (
            (255, 90, 40)
            if pl.team == TEAM_A
            else (40, 70, 255)
        )

        radius = 7 if pl.track_id == possessor.track_id else 5

        cv2.circle(
            panel,
            p(*pl.pitch_xy),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )

    # Possessor pressure rings (3m / 5m) approximately in panel x-scale.
    center = p(*possessor.pitch_xy)
    scale = (panel_w - 2 * margin) / 105.0

    cv2.circle(
        panel,
        center,
        max(2, int(round(3.0 * scale))),
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        panel,
        center,
        max(3, int(round(5.0 * scale))),
        (0, 150, 255),
        1,
        cv2.LINE_AA,
    )

    x0 = frame.shape[1] - panel_w - 8
    y0 = frame.shape[0] - panel_h - 8

    if x0 >= 0 and y0 >= 0:
        roi = frame[
            y0:y0 + panel_h,
            x0:x0 + panel_w,
        ]
        cv2.addWeighted(
            panel,
            0.92,
            roi,
            0.08,
            0,
            roi,
        )


def draw_image_lanes(
    frame,
    possessor_id,
    team_row,
    lanes,
    max_open_lanes,
):
    tracks = frame_track_map(team_row)

    possessor_track = tracks.get(possessor_id)
    if possessor_track is None:
        return

    start = foot_image_xy(possessor_track)
    if start is None:
        return

    open_drawn = 0

    for lane in lanes:
        if lane.status == "OPEN":
            if open_drawn >= max_open_lanes:
                continue
            open_drawn += 1

        receiver_track = tracks.get(lane.receiver_track_id)
        if receiver_track is None:
            continue

        end = foot_image_xy(receiver_track)
        if end is None:
            continue

        color = (
            (60, 230, 60)
            if lane.status == "OPEN"
            else (60, 60, 220)
        )

        thickness = 2 if lane.status == "OPEN" else 1

        cv2.line(
            frame,
            start,
            end,
            color,
            thickness,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    team_path = resolve_project_path(args.team_jsonl)
    possession_path = resolve_project_path(args.possession_jsonl)
    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)

    for p in (source, team_path, possession_path):
        if not p.exists():
            raise FileNotFoundError(p)

    team_rows = read_jsonl(team_path)
    possession_rows = read_jsonl(possession_path)

    common_frames = sorted(
        set(team_rows) & set(possession_rows)
    )

    if args.max_frames >= 0:
        common_frames = common_frames[:args.max_frames]

    if not common_frames:
        raise RuntimeError(
            "No common frames between team and possession JSONL."
        )

    engine = TacticalEngine(
        TacticalConfig(
            pressure_high_distance_m=args.pressure_high_m,
            pressure_medium_distance_m=args.pressure_medium_m,
            lane_half_width_m=args.lane_half_width_m,
            max_pass_distance_m=args.max_pass_distance_m,
            max_open_lanes=max(1, args.draw_open_lanes),
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create video: {output}"
        )

    stats = Counter()
    frame_set = set(common_frames)
    last_frame = max(common_frames)

    print("=" * 86)
    print(
        "FootballAnalysisAI - Tactical Engine v1 "
        "(Pressure + Passing Lanes)"
    )
    print(f"Source      : {source}")
    print(f"Team JSONL  : {team_path}")
    print(f"Possession  : {possession_path}")
    print(f"Frames      : {len(common_frames)}")
    print("=" * 86)

    frame_index = 0

    with jsonl_out.open(
        "w",
        encoding="utf-8",
    ) as out:
        try:
            while frame_index <= last_frame:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_index not in frame_set:
                    frame_index += 1
                    continue

                team_row = team_rows[frame_index]
                possession_row = possession_rows[frame_index]

                players = parse_players(team_row)
                player_map = {
                    pl.track_id: pl
                    for pl in players
                }

                possession = (
                    possession_row.get("possession")
                    or {}
                )

                possessor_id = possession.get(
                    "possessor_track_id"
                )
                state = str(
                    possession.get("state", "UNKNOWN")
                )

                pressure = None
                lanes = []
                best_open = []

                possessor = (
                    player_map.get(int(possessor_id))
                    if possessor_id is not None
                    else None
                )

                if (
                    possessor is not None
                    and state in {TEAM_A, TEAM_B}
                ):
                    teammates = [
                        pl for pl in players
                        if pl.team == possessor.team
                    ]
                    opponents = [
                        pl for pl in players
                        if pl.team != possessor.team
                    ]

                    pressure = engine.pressure(
                        possessor,
                        opponents,
                    )
                    lanes = engine.passing_lanes(
                        possessor,
                        teammates,
                        opponents,
                    )
                    best_open = engine.best_open_lanes(
                        lanes
                    )

                    stats[
                        f"pressure:{pressure.level}"
                    ] += 1
                    stats["open_lanes"] += sum(
                        1
                        for x in lanes
                        if x.status == "OPEN"
                    )
                    stats["blocked_lanes"] += sum(
                        1
                        for x in lanes
                        if x.status == "BLOCKED"
                    )
                    stats["tactical_frames"] += 1

                    draw_image_lanes(
                        frame,
                        possessor.track_id,
                        team_row,
                        lanes,
                        max_open_lanes=max(
                            1,
                            args.draw_open_lanes,
                        ),
                    )

                    draw_pitch_panel(
                        frame,
                        players,
                        possessor,
                        pressure,
                        best_open,
                    )

                else:
                    stats["no_possessor"] += 1

                # Header
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, 110),
                    (18, 18, 18),
                    -1,
                )

                if pressure is not None:
                    pressure_text = (
                        f"Pressure: {pressure.level} | "
                        f"nearest={pressure.nearest_opponent_distance_m:.2f}m | "
                        f"3m={pressure.opponents_within_3m} "
                        f"5m={pressure.opponents_within_5m}"
                    )

                    lane_text = (
                        f"Passing lanes: "
                        f"OPEN={sum(1 for x in lanes if x.status == 'OPEN')} "
                        f"BLOCKED={sum(1 for x in lanes if x.status == 'BLOCKED')}"
                    )
                else:
                    pressure_text = "Pressure: N/A"
                    lane_text = "Passing lanes: N/A"

                cv2.putText(
                    frame,
                    f"Tactical Engine v1 | Possession={state} | Owner={possessor_id}",
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    pressure_text,
                    (18, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 215, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    lane_text,
                    (18, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (100, 230, 100),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(frame)

                payload = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(
                        frame_index / fps,
                        5,
                    ),
                    "possession_state": state,
                    "possessor_track_id": possessor_id,
                    "pressure": (
                        {
                            "level": pressure.level,
                            "nearest_opponent_id": (
                                pressure.nearest_opponent_id
                            ),
                            "nearest_opponent_distance_m": (
                                round(
                                    pressure.nearest_opponent_distance_m,
                                    4,
                                )
                                if pressure.nearest_opponent_distance_m
                                is not None
                                else None
                            ),
                            "opponents_within_3m": (
                                pressure.opponents_within_3m
                            ),
                            "opponents_within_5m": (
                                pressure.opponents_within_5m
                            ),
                        }
                        if pressure is not None
                        else None
                    ),
                    "passing_lanes": [
                        {
                            "receiver_track_id": x.receiver_track_id,
                            "receiver_xy": [
                                round(x.receiver_xy[0], 4),
                                round(x.receiver_xy[1], 4),
                            ],
                            "distance_m": round(
                                x.distance_m,
                                4,
                            ),
                            "status": x.status,
                            "blocker_track_ids": list(
                                x.blocker_track_ids
                            ),
                            "nearest_blocker_clearance_m": (
                                round(
                                    x.nearest_blocker_clearance_m,
                                    4,
                                )
                                if x.nearest_blocker_clearance_m
                                is not None
                                else None
                            ),
                            "score": round(
                                x.score,
                                5,
                            ),
                        }
                        for x in lanes
                    ],
                    "best_open_receiver_ids": [
                        x.receiver_track_id
                        for x in best_open
                    ],
                }

                out.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                frame_index += 1

        finally:
            cap.release()
            writer.release()

    tactical_frames = stats["tactical_frames"]

    print("=" * 86)
    print("DONE - Tactical Engine v1")
    print(f"Frames with possessor : {tactical_frames}")
    print(f"No possessor frames   : {stats['no_possessor']}")
    print(f"HIGH pressure frames  : {stats['pressure:HIGH']}")
    print(f"MEDIUM pressure frames: {stats['pressure:MEDIUM']}")
    print(f"LOW pressure frames   : {stats['pressure:LOW']}")
    print(f"NONE pressure frames  : {stats['pressure:NONE']}")
    print(f"Open passing lanes    : {stats['open_lanes']}")
    print(f"Blocked passing lanes : {stats['blocked_lanes']}")
    print(f"Video output          : {output}")
    print(f"JSONL output          : {jsonl_out}")
    print("=" * 86)


if __name__ == "__main__":
    main()
