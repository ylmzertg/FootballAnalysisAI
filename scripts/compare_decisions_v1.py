from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.decision_comparison_v1 import (
    ACTUAL_PASS,
    ACTUAL_SHOT,
    ACTUAL_UNKNOWN,
    compare_decision,
)
from core.runtime_paths import resolve_project_path


PASS_PHASES = {
    "PASS_FLIGHT",
    "TEAM_FLIGHT",
    "CONTESTED_FLIGHT",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Decision Comparison v1"
    )

    p.add_argument("--incidents-json", required=True)
    p.add_argument("--pass-options-jsonl", required=True)
    p.add_argument("--possession-events-jsonl", required=True)
    p.add_argument("--shot-jsonl", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--lookahead-frames", type=int, default=45)

    return p.parse_args()


def read_jsonl(path):
    result = {}

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            result[int(row["frame_index"])] = row

    return result


def ranked_option_near_peak(
    pass_rows,
    peak_frame,
    *,
    radius=3,
):
    candidates = []

    for frame in range(
        max(0, peak_frame - radius),
        peak_frame + radius + 1,
    ):
        row = pass_rows.get(frame)

        if row is None:
            continue

        for option in row.get("options", []):
            if option.get("category") not in {"BEST", "GOOD"}:
                continue

            candidates.append(
                (
                    float(option.get("score", 0.0) or 0.0),
                    -abs(frame - peak_frame),
                    frame,
                    row.get("possessor_track_id"),
                    option,
                )
            )

    if not candidates:
        return None

    candidates.sort(reverse=True)

    _, _, frame, possessor_id, option = candidates[0]

    return {
        "frame": frame,
        "possessor_track_id": possessor_id,
        "option": option,
    }


def find_actual_action(
    *,
    start_frame,
    possessor_track_id,
    possession_rows,
    shot_rows,
    lookahead,
):
    end_frame = start_frame + lookahead

    # Shot has priority if it appears before a resolved actual pass target.
    first_shot = None

    for frame in range(start_frame, end_frame + 1):
        shot = shot_rows.get(frame, {})

        phase = str(
            shot.get(
                "phase_v16",
                shot.get("phase", ""),
            )
        )

        if phase in {"SHOT_FLIGHT", "GOAL_ATTEMPT"}:
            first_shot = frame
            break

    first_pass = None

    for frame in range(start_frame, end_frame + 1):
        row = possession_rows.get(frame, {})
        phase = str(row.get("phase", ""))

        if phase not in PASS_PHASES:
            continue

        source_owner = row.get("source_owner_track_id")
        target_owner = row.get("target_owner_track_id")

        if (
            possessor_track_id is not None
            and source_owner is not None
            and int(source_owner) != int(possessor_track_id)
        ):
            continue

        if target_owner is None:
            continue

        first_pass = (
            frame,
            int(target_owner),
        )
        break

    if (
        first_shot is not None
        and (
            first_pass is None
            or first_shot <= first_pass[0]
        )
    ):
        return (
            ACTUAL_SHOT,
            None,
            first_shot,
        )

    if first_pass is not None:
        return (
            ACTUAL_PASS,
            first_pass[1],
            first_pass[0],
        )

    return (
        ACTUAL_UNKNOWN,
        None,
        None,
    )


def main():
    args = parse_args()

    incidents = json.loads(
        resolve_project_path(
            args.incidents_json
        ).read_text(
            encoding="utf-8"
        )
    )

    pass_rows = read_jsonl(
        resolve_project_path(
            args.pass_options_jsonl
        )
    )

    possession_rows = read_jsonl(
        resolve_project_path(
            args.possession_events_jsonl
        )
    )

    shot_rows = read_jsonl(
        resolve_project_path(
            args.shot_jsonl
        )
    )

    output = resolve_project_path(
        args.output_json
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparisons = []

    for incident in incidents:
        iid = str(
            incident.get("incident_id")
        )

        peak = int(
            incident.get("peak_frame")
        )

        ranked = ranked_option_near_peak(
            pass_rows,
            peak,
        )

        if ranked is None:
            decision_frame = peak
            possessor_id = None
            best_receiver = None
            best_score = None
            best_category = None

        else:
            decision_frame = int(
                ranked["frame"]
            )
            possessor_id = ranked[
                "possessor_track_id"
            ]
            option = ranked["option"]
            best_receiver = option.get(
                "receiver_track_id"
            )
            best_score = float(
                option.get("score", 0.0)
                or 0.0
            )
            best_category = str(
                option.get("category")
            )

        (
            actual_action,
            actual_receiver,
            actual_frame,
        ) = find_actual_action(
            start_frame=decision_frame,
            possessor_track_id=possessor_id,
            possession_rows=possession_rows,
            shot_rows=shot_rows,
            lookahead=max(
                1,
                args.lookahead_frames,
            ),
        )

        comparison = compare_decision(
            incident_id=iid,
            decision_frame=decision_frame,
            possessor_track_id=(
                int(possessor_id)
                if possessor_id is not None
                else None
            ),
            best_receiver_id=(
                int(best_receiver)
                if best_receiver is not None
                else None
            ),
            best_score=best_score,
            best_category=best_category,
            actual_action=actual_action,
            actual_receiver_id=actual_receiver,
            actual_frame=actual_frame,
        )

        comparisons.append(
            comparison.__dict__
        )

    output.write_text(
        json.dumps(
            comparisons,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 92)
    print("DONE - Decision Comparison v1")
    print(f"Incidents             : {len(comparisons)}")

    for item in comparisons:
        print(
            f"  {item['incident_id']} "
            f"frame={item['decision_frame']} "
            f"best={item['best_receiver_id']} "
            f"actual={item['actual_action']}"
            f"/{item['actual_receiver_id']} "
            f"=> {item['comparison']}"
        )

    print(f"Output JSON           : {output}")
    print("=" * 92)


if __name__ == "__main__":
    main()
