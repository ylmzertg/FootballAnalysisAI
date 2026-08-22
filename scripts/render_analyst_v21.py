from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from core.analyst_renderer_v2 import (
    IncidentCandidate,
    IncidentSelectionConfig,
    select_incidents,
)
from core.runtime_paths import resolve_project_path
from core.video_i18n import (
    SUPPORTED_LANGUAGES,
    resolve_video_language,
    tr,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "FootballAnalysisAI - Analyst Renderer v2.1 "
            "dual-perspective incident video"
        )
    )

    p.add_argument(
        "--source",
        required=True,
    )
    p.add_argument(
        "--team-jsonl",
        required=True,
    )
    p.add_argument(
        "--errors-timeline-json",
        required=True,
    )
    p.add_argument(
        "--marking-jsonl",
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
        "--incidents-json",
        required=True,
    )
    p.add_argument(
        "--decision-comparison-json",
        required=True,
        help="Decision Comparison V1 JSON output.",
    )
    p.add_argument(
        "--language",
        default=None,
        help=(
            "Video language: "
            + "/".join(SUPPORTED_LANGUAGES.keys())
            + ". If omitted, interactive selection is shown."
        ),
    )

    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--storyboard-json",
        required=True,
    )

    p.add_argument(
        "--max-incidents",
        type=int,
        default=3,
    )

    p.add_argument(
        "--freeze-seconds",
        type=float,
        default=1.15,
    )

    p.add_argument(
        "--pre-roll-frames",
        type=int,
        default=8,
    )

    p.add_argument(
        "--post-roll-frames",
        type=int,
        default=25,
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

            rows[
                int(
                    row["frame_index"]
                )
            ] = row

    return rows


def read_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


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
                (x1 + x2)
                / 2.0
            )
        ),
        int(
            round(y2)
        ),
    )


def track_map(
    team_row,
):
    return {
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


def draw_circle(
    frame,
    track,
    *,
    color,
    label,
    thickness=3,
):
    bbox = track.get(
        "bbox_xyxy"
    )

    if (
        bbox is None
        or len(bbox) != 4
    ):
        return

    x1, y1, x2, y2 = [
        int(
            round(
                float(v)
            )
        )
        for v in bbox
    ]

    cx = int(
        round(
            (x1 + x2)
            / 2.0
        )
    )
    cy = int(
        round(
            (y1 + y2)
            / 2.0
        )
    )

    radius = max(
        18,
        int(
            round(
                max(
                    x2 - x1,
                    y2 - y1,
                )
                * 0.65
            )
        ),
    )

    cv2.circle(
        frame,
        (
            cx,
            cy,
        ),
        radius,
        color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                18,
                y1 - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_header(
    frame,
    *,
    title,
    line1="",
    line2="",
):
    width = frame.shape[1]

    cv2.rectangle(
        frame,
        (
            0,
            0,
        ),
        (
            width,
            116,
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
        title[:125],
        (
            18,
            30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (
            235,
            235,
            235,
        ),
        2,
        cv2.LINE_AA,
    )

    if line1:
        cv2.putText(
            frame,
            line1[:150],
            (
                18,
                64,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (
                210,
                210,
                210,
            ),
            1,
            cv2.LINE_AA,
        )

    if line2:
        cv2.putText(
            frame,
            line2[:150],
            (
                18,
                94,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                165,
                185,
                185,
            ),
            1,
            cv2.LINE_AA,
        )


def write_freeze(
    writer,
    frame,
    *,
    fps,
    seconds,
):
    copies = max(
        1,
        int(
            round(
                fps
                * max(
                    0.2,
                    seconds,
                )
            )
        ),
    )

    for _ in range(
        copies
    ):
        writer.write(
            frame
        )


def frame_at(
    cap,
    frame_index,
):
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_index),
    )

    ok, frame = cap.read()

    if not ok:
        return None

    return frame


def best_marking(
    marking_row,
):
    candidates = [
        item
        for item
        in marking_row.get(
            "marking",
            [],
        )
        if (
            item.get(
                "marking_state"
            )
            in {
                "LOOSE",
                "UNMARKED",
            }
            and item.get(
                "dangerous"
            )
        )
    ]

    candidates.sort(
        key=lambda x: float(
            x.get(
                "threat_score",
                0.0,
            )
        ),
        reverse=True,
    )

    return (
        candidates[0]
        if candidates
        else None
    )


def best_pass_option(
    pass_row,
):
    options = pass_row.get(
        "options",
        [],
    )

    candidates = [
        option
        for option
        in options
        if option.get(
            "category"
        )
        in {
            "BEST",
            "GOOD",
        }
    ]

    candidates.sort(
        key=lambda x: float(
            x.get(
                "score",
                0.0,
            )
        ),
        reverse=True,
    )

    return (
        candidates[0]
        if candidates
        else None
    )


def decision_texts(decision, language):
    if not decision:
        return (
            tr(language, "no_ranked_option"),
            tr(language, "actual_unknown"),
            tr(language, "no_clear_comparison"),
        )

    best_id = decision.get("best_receiver_id")
    best_score = decision.get("best_score")
    best_category = decision.get("best_category")

    if best_id is None:
        best_text = tr(language, "no_ranked_option")
    else:
        best_text = tr(
            language,
            "ranked_option",
            receiver_id=best_id,
            category=best_category or "",
            score=f"{float(best_score or 0):.2f}",
        )

    actual_action = str(decision.get("actual_action") or "UNKNOWN")
    actual_id = decision.get("actual_receiver_id")

    if actual_action == "PASS" and actual_id is not None:
        actual_text = tr(
            language,
            "actual_pass_to",
            receiver_id=actual_id,
        )
    elif actual_action == "PASS":
        actual_text = tr(language, "actual_pass")
    elif actual_action == "SHOT":
        actual_text = tr(language, "actual_shot")
    elif actual_action == "CARRY_OR_CONTINUE":
        actual_text = tr(language, "actual_carry")
    elif actual_action == "TURNOVER":
        actual_text = tr(language, "actual_turnover")
    else:
        actual_text = tr(language, "actual_unknown")

    comparison = str(
        decision.get("comparison")
        or "NO_CLEAR_COMPARISON"
    )

    key_by_comparison = {
        "MATCHED_BEST": "match_best",
        "CHOSE_ALTERNATIVE": "chose_alternative",
        "SHOT_OVER_PASS": "shot_over_pass",
        "TURNOVER_OVER_PASS": "turnover_over_pass",
        "NO_CLEAR_COMPARISON": "no_clear_comparison",
    }

    comparison_key = key_by_comparison.get(
        comparison,
        "no_clear_comparison",
    )

    comparison_text = tr(
        language,
        comparison_key,
        best_id=best_id,
        actual_id=actual_id,
    )

    return best_text, actual_text, comparison_text


def incident_candidate(
    item,
):
    return IncidentCandidate(
        incident_id=str(
            item.get(
                "incident_id"
            )
        ),
        attacking_team=str(
            item.get(
                "attacking_team"
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
        attack_merit_level=str(
            item.get(
                "attack_merit_level"
            )
        ),
        defense_vulnerability_level=str(
            item.get(
                "defense_vulnerability_level"
            )
        ),
        shot_detected=bool(
            item.get(
                "shot_detected"
            )
        ),
        error_types=tuple(
            item.get(
                "error_types",
                [],
            )
        ),
    )


def main():
    args = parse_args()

    language = resolve_video_language(
        args.language,
        interactive=True,
        default="tr",
    )

    source = resolve_project_path(
        args.source
    )

    team_rows = read_jsonl(
        resolve_project_path(
            args.team_jsonl
        )
    )

    marking_rows = read_jsonl(
        resolve_project_path(
            args.marking_jsonl
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

    errors = read_json(
        resolve_project_path(
            args.errors_timeline_json
        )
    )

    incident_payload = read_json(
        resolve_project_path(
            args.incidents_json
        )
    )

    decision_payload = read_json(
        resolve_project_path(
            args.decision_comparison_json
        )
    )

    decision_by_id = {
        str(item.get("incident_id")): item
        for item in decision_payload
    }

    output = resolve_project_path(
        args.output
    )

    storyboard_out = (
        resolve_project_path(
            args.storyboard_json
        )
    )

    payload_by_id = {
        str(
            item.get(
                "incident_id"
            )
        ): item
        for item in incident_payload
    }

    candidates = [
        incident_candidate(
            item
        )
        for item
        in incident_payload
    ]

    selected = select_incidents(
        candidates,
        IncidentSelectionConfig(
            max_incidents=max(
                1,
                args.max_incidents,
            )
        ),
    )

    error_by_id = {
        str(
            item.get(
                "event_id"
            )
        ): item
        for item
        in errors
    }

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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    storyboard_out.parent.mkdir(
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

    storyboard = []

    print(f"[V2.1] language={language}")
    print(f"[V2.1] decision comparisons={len(decision_by_id)}")

    for selected_incident in selected:
        incident = payload_by_id[
            selected_incident.incident_id
        ]

        peak = selected_incident.peak_frame

        decision = decision_by_id.get(
            selected_incident.incident_id,
            {},
        )

        team_row = team_rows.get(
            peak,
            {},
        )

        tracks = track_map(
            team_row
        )

        mark = best_marking(
            marking_rows.get(
                peak,
                {},
            )
        )

        pass_option = best_pass_option(
            pass_rows.get(
                peak,
                {},
            )
        )

        error_items = [
            error_by_id.get(
                str(event_id)
            )
            for event_id
            in incident.get(
                "error_event_ids",
                [],
            )
        ]

        error_items = [
            x
            for x in error_items
            if x is not None
        ]

        # ------------------------------------------------------------
        # A) DEFENSE VIEW
        # ------------------------------------------------------------
        frame = frame_at(
            cap,
            peak,
        )

        if frame is not None:
            for error in error_items:
                tid = error.get(
                    "primary_track_id"
                )

                if tid is None:
                    continue

                track = tracks.get(
                    int(tid)
                )

                if track is None:
                    continue

                draw_circle(
                    frame,
                    track,
                    color=(
                        0,
                        0,
                        255,
                    ),
                    label=str(
                        error.get(
                            "error_type",
                            "ERROR",
                        )
                    ),
                )

            if mark is not None:
                attacker = tracks.get(
                    int(
                        mark.get(
                            "attacker_track_id"
                        )
                    )
                )

                if attacker is not None:
                    draw_circle(
                        frame,
                        attacker,
                        color=(
                            0,
                            165,
                            255,
                        ),
                        label=(
                            f"{mark.get('marking_state')} "
                            f"{float(mark.get('nearest_defender_distance_m', 0)):.1f}m"
                        ),
                    )

            defense_text = (
                incident.get(
                    "defense_view",
                    [],
                )
            )

            draw_header(
                frame,
                title=(
                    f"{selected_incident.incident_id} | "
                    f"{tr(language, 'defense_view')}"
                ),
                line1=(
                    defense_text[0]
                    if defense_text
                    else (
                        "Savunma yerlesimindeki "
                        "zafiyetler inceleniyor."
                    )
                ),
                line2=(
                    f"{tr(language, 'defense_vulnerability')}: "
                    f"{selected_incident.defense_vulnerability_level}"
                ),
            )

            write_freeze(
                writer,
                frame,
                fps=fps,
                seconds=args.freeze_seconds,
            )

        # ------------------------------------------------------------
        # B) ATTACK VIEW
        # ------------------------------------------------------------
        frame = frame_at(
            cap,
            peak,
        )

        if frame is not None:
            owner_id = (
                pass_rows.get(
                    peak,
                    {},
                ).get(
                    "possessor_track_id"
                )
            )

            if (
                pass_option is not None
                and owner_id
                is not None
            ):
                owner_track = tracks.get(
                    int(
                        owner_id
                    )
                )

                receiver_track = tracks.get(
                    int(
                        pass_option.get(
                            "receiver_track_id"
                        )
                    )
                )

                start = (
                    foot_xy(
                        owner_track
                    )
                    if owner_track
                    is not None
                    else None
                )

                end = (
                    foot_xy(
                        receiver_track
                    )
                    if receiver_track
                    is not None
                    else None
                )

                if (
                    start is not None
                    and end is not None
                ):
                    cv2.arrowedLine(
                        frame,
                        start,
                        end,
                        (
                            60,
                            240,
                            60,
                        ),
                        4,
                        cv2.LINE_AA,
                        tipLength=0.08,
                    )

                    draw_circle(
                        frame,
                        receiver_track,
                        color=(
                            60,
                            240,
                            60,
                        ),
                        label=(
                            f"{pass_option.get('category')} "
                            f"{float(pass_option.get('score', 0)):.2f}"
                        ),
                    )

            attack_text = (
                incident.get(
                    "attack_view",
                    [],
                )
            )

            draw_header(
                frame,
                title=(
                    f"{selected_incident.incident_id} | "
                    f"{tr(language, 'attack_view')}"
                ),
                line1=(
                    attack_text[0]
                    if attack_text
                    else (
                        "Hucumun yarattigi alan ve "
                        "pas opsiyonlari inceleniyor."
                    )
                ),
                line2=(
                    f"{tr(language, 'attack_merit')}: "
                    f"{selected_incident.attack_merit_level}"
                ),
            )

            write_freeze(
                writer,
                frame,
                fps=fps,
                seconds=args.freeze_seconds,
            )

        # ------------------------------------------------------------
        # C) DECISION MOMENT
        # ------------------------------------------------------------
        decision_frame = int(
            decision.get(
                "decision_frame",
                peak,
            )
        )

        frame = frame_at(
            cap,
            decision_frame,
        )

        if frame is not None:
            decision_team_row = team_rows.get(
                decision_frame,
                {},
            )
            decision_tracks = track_map(
                decision_team_row
            )

            possessor_id = decision.get(
                "possessor_track_id"
            )
            best_receiver_id = decision.get(
                "best_receiver_id"
            )
            actual_receiver_id = decision.get(
                "actual_receiver_id"
            )

            possessor_track = (
                decision_tracks.get(int(possessor_id))
                if possessor_id is not None
                else None
            )

            best_track = (
                decision_tracks.get(int(best_receiver_id))
                if best_receiver_id is not None
                else None
            )

            actual_track = (
                decision_tracks.get(int(actual_receiver_id))
                if actual_receiver_id is not None
                else None
            )

            if possessor_track is not None:
                draw_circle(
                    frame,
                    possessor_track,
                    color=(255, 210, 40),
                    label="BALL",
                )

            if best_track is not None:
                draw_circle(
                    frame,
                    best_track,
                    color=(60, 240, 60),
                    label="MODEL",
                )

                start = (
                    foot_xy(possessor_track)
                    if possessor_track is not None
                    else None
                )
                end = foot_xy(best_track)

                if start is not None and end is not None:
                    cv2.arrowedLine(
                        frame,
                        start,
                        end,
                        (60, 240, 60),
                        4,
                        cv2.LINE_AA,
                        tipLength=0.08,
                    )

            if (
                actual_track is not None
                and actual_receiver_id != best_receiver_id
            ):
                draw_circle(
                    frame,
                    actual_track,
                    color=(0, 165, 255),
                    label="ACTUAL",
                )

            (
                best_text,
                actual_text,
                comparison_text,
            ) = decision_texts(
                decision,
                language,
            )

            draw_header(
                frame,
                title=(
                    f"{selected_incident.incident_id} | "
                    f"{tr(language, 'decision_moment')}"
                ),
                line1=best_text,
                line2=actual_text,
            )

            cv2.rectangle(
                frame,
                (0, 116),
                (width, 150),
                (28, 28, 28),
                -1,
            )

            cv2.putText(
                frame,
                comparison_text[:150],
                (18, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            write_freeze(
                writer,
                frame,
                fps=fps,
                seconds=max(
                    1.0,
                    args.freeze_seconds,
                ),
            )

        # ------------------------------------------------------------
        # D) ALTERNATIVE VIEW
        # ------------------------------------------------------------
        frame = frame_at(
            cap,
            peak,
        )

        if frame is not None:
            alternatives = (
                incident.get(
                    "alternative_view",
                    [],
                )
            )

            draw_header(
                frame,
                title=(
                    f"{selected_incident.incident_id} | "
                    f"{tr(language, 'alternative_defense')}"
                ),
                line1=(
                    alternatives[0]
                    if alternatives
                    else (
                        "Savunmanin olasi alternatif "
                        "reaksiyonu."
                    )
                ),
                line2=tr(
                    language,
                    "tactical_alternative_note",
                ),
            )

            write_freeze(
                writer,
                frame,
                fps=fps,
                seconds=max(
                    0.75,
                    args.freeze_seconds
                    * 0.85,
                ),
            )

        # ------------------------------------------------------------
        # E) REAL ACTION
        # ------------------------------------------------------------
        play_start = max(
            0,
            peak
            - max(
                0,
                args.pre_roll_frames,
            ),
        )

        play_end = min(
            total_frames - 1,
            selected_incident.end_frame
            + max(
                0,
                args.post_roll_frames,
            ),
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            play_start,
        )

        for source_frame in range(
            play_start,
            play_end + 1,
        ):
            ok, action_frame = (
                cap.read()
            )

            if not ok:
                break

            shot_row = (
                shot_rows.get(
                    source_frame,
                    {},
                )
            )

            shot_phase = str(
                shot_row.get(
                    "phase_v16",
                    shot_row.get(
                        "phase",
                        "",
                    ),
                )
            )

            draw_header(
                action_frame,
                title=(
                    f"{selected_incident.incident_id} | "
                    f"{tr(language, 'real_action')}"
                ),
                line1=(
                    f"Kaynak frame: {source_frame} | "
                    f"Attack={selected_incident.attacking_team}"
                ),
                line2=(
                    f"Outcome: {shot_phase}"
                    if shot_phase
                    in {
                        "SHOT_FLIGHT",
                        "GOAL_ATTEMPT",
                    }
                    else (
                        tr(language, "position_continues")
                    )
                ),
            )

            writer.write(
                action_frame
            )

        storyboard.append(
            {
                "incident_id": (
                    selected_incident.incident_id
                ),
                "peak_frame": peak,
                "play_start_frame": (
                    play_start
                ),
                "play_end_frame": (
                    play_end
                ),
                "attack_merit_level": (
                    selected_incident.attack_merit_level
                ),
                "defense_vulnerability_level": (
                    selected_incident.defense_vulnerability_level
                ),
                "shot_detected": (
                    selected_incident.shot_detected
                ),
                "error_types": list(
                    selected_incident.error_types
                ),
                "best_pass_receiver_id": (
                    pass_option.get(
                        "receiver_track_id"
                    )
                    if pass_option
                    is not None
                    else None
                ),
                "marking_target_id": (
                    mark.get(
                        "attacker_track_id"
                    )
                    if mark
                    is not None
                    else None
                ),
            }
        )

    cap.release()
    writer.release()

    storyboard_out.write_text(
        json.dumps(
            storyboard,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 92)
    print(
        "DONE - Analyst Renderer v2 "
        "DUAL PERSPECTIVE"
    )
    print(
        f"Input incidents       : "
        f"{len(candidates)}"
    )
    print(
        f"Selected incidents    : "
        f"{len(selected)}"
    )

    for incident in selected:
        print(
            f"  {incident.incident_id} "
            f"peak={incident.peak_frame} "
            f"attack={incident.attacking_team} "
            f"attack_merit="
            f"{incident.attack_merit_level} "
            f"def_vuln="
            f"{incident.defense_vulnerability_level} "
            f"shot={incident.shot_detected}"
        )

    print(
        f"Video output          : "
        f"{output}"
    )
    print(
        f"Storyboard JSON       : "
        f"{storyboard_out}"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
