from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from core.possession_events import (
    CONTROL,
    CONTROL_GAP,
    PASS_FLIGHT,
    TEAM_FLIGHT,
    CONTESTED_FLIGHT,
    RAW_LOOSE,
    RAW_UNKNOWN,
    TEAM_A,
    TEAM_B,
    LOOSE,
    UNKNOWN,
    ControlFrame,
    PossessionEventConfig,
    PossessionEventReconstructor,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Possession v1.2 event reconstruction"
    )
    p.add_argument("--source", default=r"input\input.mp4")
    p.add_argument(
        "--possession-jsonl",
        default=r"output\possession_v11.jsonl",
    )
    p.add_argument(
        "--output",
        default=r"output\possession_v12_events.mp4",
    )
    p.add_argument(
        "--jsonl",
        default=r"output\possession_v12_events.jsonl",
    )
    p.add_argument(
        "--max-bridge-gap",
        type=int,
        default=30,
    )
    p.add_argument(
        "--max-unresolved-flight",
        type=int,
        default=18,
    )
    p.add_argument(
        "--min-ball-motion-px",
        type=float,
        default=8.0,
    )
    p.add_argument(
        "--max-missing-ball-ratio",
        type=float,
        default=0.45,
    )
    return p.parse_args()


def read_rows(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    rows.sort(
        key=lambda x: int(
            x["frame_index"]
        )
    )
    return rows


def to_control_frame(row: dict) -> ControlFrame:
    possession = (
        row.get("possession")
        or {}
    )
    ball = (
        row.get("ball")
        or {}
    )

    image_xy = ball.get("image_xy")

    return ControlFrame(
        frame_index=int(
            row["frame_index"]
        ),
        state=str(
            possession.get(
                "state",
                UNKNOWN,
            )
        ),
        possessor_track_id=(
            int(
                possession[
                    "possessor_track_id"
                ]
            )
            if possession.get(
                "possessor_track_id"
            )
            is not None
            else None
        ),
        possessor_team=(
            str(
                possession.get(
                    "possessor_team"
                )
            )
            if possession.get(
                "possessor_team"
            )
            in {TEAM_A, TEAM_B}
            else None
        ),
        ball_image_xy=(
            (
                float(image_xy[0]),
                float(image_xy[1]),
            )
            if image_xy is not None
            else None
        ),
        ball_detected=bool(
            ball.get(
                "detected",
                False,
            )
        ),
        ball_predicted=bool(
            ball.get(
                "predicted",
                False,
            )
        ),
    )


def main():
    args = parse_args()

    source = resolve_project_path(
        args.source
    )
    possession_path = resolve_project_path(
        args.possession_jsonl
    )
    output = resolve_project_path(
        args.output
    )
    jsonl_out = resolve_project_path(
        args.jsonl
    )

    for p in (
        source,
        possession_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    raw_rows = read_rows(
        possession_path
    )

    controls = [
        to_control_frame(row)
        for row in raw_rows
    ]

    reconstructor = PossessionEventReconstructor(
        PossessionEventConfig(
            max_bridge_gap_frames=max(
                1,
                args.max_bridge_gap,
            ),
            max_unresolved_flight_frames=max(
                1,
                args.max_unresolved_flight,
            ),
            min_ball_motion_px=max(
                0.0,
                args.min_ball_motion_px,
            ),
            max_missing_ball_ratio=min(
                1.0,
                max(
                    0.0,
                    args.max_missing_ball_ratio,
                ),
            ),
        )
    )

    events = reconstructor.reconstruct(
        controls
    )

    if len(events) != len(raw_rows):
        raise RuntimeError(
            "Event reconstruction frame count mismatch."
        )

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
        (width, height),
    )

    row_by_frame = {
        int(r["frame_index"]): r
        for r in raw_rows
    }
    event_by_frame = {
        e.frame_index: e
        for e in events
    }

    stats = Counter()
    last_frame = max(
        event_by_frame
    )
    frame_index = 0

    colors = {
        TEAM_A: (255, 90, 40),
        TEAM_B: (40, 70, 255),
        LOOSE: (0, 215, 255),
        UNKNOWN: (150, 150, 150),
    }

    with jsonl_out.open(
        "w",
        encoding="utf-8",
    ) as out:
        try:
            while frame_index <= last_frame:
                ok, frame = cap.read()

                if not ok:
                    break

                event = event_by_frame.get(
                    frame_index
                )

                raw = row_by_frame.get(
                    frame_index
                )

                if event is None or raw is None:
                    frame_index += 1
                    continue

                stats[
                    f"team:{event.team_state}"
                ] += 1
                stats[
                    f"phase:{event.phase}"
                ] += 1

                color = colors.get(
                    event.team_state,
                    (200, 200, 200),
                )

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, 118),
                    (18, 18, 18),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        f"Possession v1.2 | "
                        f"{event.team_state} | "
                        f"{event.phase}"
                    ),
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"Owner={event.possessor_track_id} | "
                        f"from={event.source_owner_track_id} "
                        f"to={event.target_owner_track_id} | "
                        f"conf={event.confidence:.2f}"
                    ),
                    (18, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    event.reason,
                    (18, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

                raw_state = (
                    raw.get(
                        "possession",
                        {},
                    ).get(
                        "state",
                        UNKNOWN,
                    )
                )

                cv2.putText(
                    frame,
                    (
                        f"Raw v1.1 state: "
                        f"{raw_state}"
                    ),
                    (18, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(
                    frame
                )

                payload = {
                    "frame_index": (
                        frame_index
                    ),
                    "timestamp_seconds": round(
                        frame_index / fps,
                        5,
                    ),
                    "team_state": (
                        event.team_state
                    ),
                    "phase": event.phase,
                    "possessor_track_id": (
                        event.possessor_track_id
                    ),
                    "source_owner_track_id": (
                        event.source_owner_track_id
                    ),
                    "target_owner_track_id": (
                        event.target_owner_track_id
                    ),
                    "source_team": (
                        event.source_team
                    ),
                    "target_team": (
                        event.target_team
                    ),
                    "confidence": round(
                        event.confidence,
                        5,
                    ),
                    "reason": event.reason,
                    "raw_possession": (
                        raw.get(
                            "possession"
                        )
                    ),
                    "ball": raw.get(
                        "ball"
                    ),
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

    total = len(events)

    print("=" * 86)
    print("DONE - Possession v1.2 EVENT STATE MACHINE")
    print(f"Frames processed   : {total}")
    print(f"TEAM_A team-state  : {stats['team:TEAM_A']}")
    print(f"TEAM_B team-state  : {stats['team:TEAM_B']}")
    print(f"LOOSE              : {stats['team:LOOSE']}")
    print(f"UNKNOWN            : {stats['team:UNKNOWN']}")
    print("Phases:")
    for phase in (
        CONTROL,
        CONTROL_GAP,
        PASS_FLIGHT,
        TEAM_FLIGHT,
        CONTESTED_FLIGHT,
        RAW_LOOSE,
        RAW_UNKNOWN,
    ):
        print(
            f"  {phase:<20} "
            f"{stats[f'phase:{phase}']}"
        )
    print(f"Video output       : {output}")
    print(f"JSONL output       : {jsonl_out}")
    print("=" * 86)


if __name__ == "__main__":
    main()
