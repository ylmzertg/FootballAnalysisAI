from __future__ import annotations

import sys
from pathlib import Path

# Always expose the FootballAnalysisAI project root when this file is launched
# as: python scripts\\classify_teams_v21.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time
from collections import Counter, defaultdict
import cv2

from core.team_classifier_v2 import (
    GOALKEEPER,
    PLAYER,
    REFEREE,
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    DetectionObservation,
    TeamAssignment,
    TeamClassifierV2,
    TeamClassifierV2Config,
)


ROLE_HINTS = {
    "player": PLAYER,
    "goalkeeper": GOALKEEPER,
    "referee": REFEREE,
}

TEAM_COLORS = {
    TEAM_A: (255, 90, 40),       # BGR
    TEAM_B: (40, 70, 255),
    UNKNOWN: (160, 160, 160),
}

ROLE_COLORS = {
    REFEREE: (0, 215, 255),
    GOALKEEPER: (220, 90, 220),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - Team Classification V2.1 role-guard harness"
    )
    p.add_argument(
        "--source",
        default=r"E:\Youtube\SporAnimasyon\SporAnimasyonCalisma\data\input\input.mp4",
    )
    p.add_argument(
        "--tracking",
        default=r"output\player_tracking.jsonl",
    )
    p.add_argument(
        "--output",
        default=r"output\team_classification_v22.mp4",
    )
    p.add_argument(
        "--jsonl",
        default=r"output\team_classification_v22.jsonl",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="First comparison run. Use -1 for full video.",
    )
    p.add_argument(
        "--embedding-stride",
        type=int,
        default=5,
        help="Deep embedding refresh cadence. GTX 1050 default: 5.",
    )
    p.add_argument(
        "--bootstrap-min-samples",
        type=int,
        default=30,
    )
    return p.parse_args()


def read_jsonl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows[int(item["frame_index"])] = item
    return rows


def make_observation(track: dict) -> DetectionObservation | None:
    try:
        track_id = int(track.get("track_id", -1))
        bbox = track.get("bbox_xyxy")
        if track_id < 0 or not bbox or len(bbox) != 4:
            return None

        class_name = str(track.get("class_name", "player")).lower()
        role_hint = ROLE_HINTS.get(class_name, PLAYER)

        return DetectionObservation(
            track_id=track_id,
            bbox_xyxy=tuple(float(v) for v in bbox),
            confidence=float(track.get("confidence", 1.0)),
            pitch_xy=None,  # PnLCalib pitch coordinates enter in Radar v4 integration.
            role_hint=role_hint,
        )
    except Exception:
        return None


def assignment_to_dict(a: TeamAssignment) -> dict:
    return {
        "track_id": int(a.track_id),
        "team": a.team,
        "role": a.role,
        "confidence": round(float(a.confidence), 5),
        "raw_team": a.raw_team,
        "raw_confidence": round(float(a.raw_confidence), 5),
        "id_switch_suspected": bool(a.id_switch_suspected),
        "reason": a.reason,
    }


def draw_label(frame, bbox, assignment: TeamAssignment):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if assignment.role in ROLE_COLORS:
        color = ROLE_COLORS[assignment.role]
    else:
        color = TEAM_COLORS.get(assignment.team, TEAM_COLORS[UNKNOWN])

    thickness = 3 if assignment.id_switch_suspected else 2

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        cv2.LINE_AA,
    )

    team_text = assignment.team
    if assignment.role == REFEREE:
        team_text = "REF"
    elif assignment.role == GOALKEEPER:
        team_text = f"GK/{assignment.team}"

    text = (
        f"ID {assignment.track_id} | "
        f"{team_text} | {assignment.confidence:.2f}"
    )

    if assignment.id_switch_suspected:
        text += " | SWITCH?"

    (tw, th), base = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        1,
    )

    label_y = max(th + base + 4, y1)
    cv2.rectangle(
        frame,
        (x1, label_y - th - base - 5),
        (min(w - 1, x1 + tw + 6), label_y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 3, label_y - base - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    source = Path(args.source)
    tracking_path = Path(args.tracking)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl)

    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")
    if not tracking_path.exists():
        raise FileNotFoundError(f"Tracking JSONL not found: {tracking_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    tracking = read_jsonl(tracking_path)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    config = TeamClassifierV2Config(
        embedding_stride=max(1, args.embedding_stride),
        bootstrap_min_samples=max(8, args.bootstrap_min_samples),
        use_deep_embedding=True,
        embedding_device="auto",
    )
    classifier = TeamClassifierV2(config)

    print("=" * 82)
    print("FootballAnalysisAI - Team Classifier V2.2 REF-CONSENSUS")
    print("Harness version    : 2.2-ref-consensus-final")
    print(f"Source            : {source}")
    print(f"Tracking          : {tracking_path}")
    print(f"Output            : {output_path}")
    print(f"JSONL             : {jsonl_path}")
    print(f"Embedding backend : {classifier.embedding_backend}")
    print(f"Embedding stride  : {config.embedding_stride}")
    print(f"Bootstrap samples : {config.bootstrap_min_samples}")
    print("PnL pitch_xy      : not supplied in this harness (Radar v4 stage)")
    print("=" * 82)

    frame_index = 0
    started = time.perf_counter()

    stats = Counter()
    unique_ids: set[int] = set()
    per_track_roles = defaultdict(Counter)
    per_track_teams = defaultdict(Counter)

    with jsonl_path.open("w", encoding="utf-8") as out:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if args.max_frames >= 0 and frame_index >= args.max_frames:
                    break

                tracks = tracking.get(
                    frame_index,
                    {"tracks": []},
                ).get("tracks", [])

                observations: list[DetectionObservation] = []
                track_by_id: dict[int, dict] = {}

                for tr in tracks:
                    obs = make_observation(tr)
                    if obs is None:
                        continue
                    observations.append(obs)
                    track_by_id[obs.track_id] = tr
                    unique_ids.add(obs.track_id)

                # Bootstrap only from normal outfield players. This prevents
                # goalkeeper/referee kits from contaminating the two team clusters.
                if not classifier.is_ready:
                    classify_input = [
                        o for o in observations
                        if o.role_hint == PLAYER
                    ]
                else:
                    classify_input = observations

                assignments = classifier.classify_frame(
                    frame,
                    classify_input,
                    frame_index,
                )

                by_id = {a.track_id: a for a in assignments}

                # During bootstrap, keep role detections visible even though they
                # are deliberately excluded from team clustering.
                if not classifier.is_ready:
                    for obs in observations:
                        if obs.track_id in by_id:
                            continue
                        if obs.role_hint == REFEREE:
                            by_id[obs.track_id] = TeamAssignment(
                                track_id=obs.track_id,
                                team=UNKNOWN,
                                role=REFEREE,
                                confidence=max(0.60, obs.confidence),
                                raw_team=UNKNOWN,
                                raw_confidence=0.0,
                                reason="bootstrap_role_hint",
                            )
                        elif obs.role_hint == GOALKEEPER:
                            # Do not display a definitive GK role until PnLCalib
                            # supplies pitch coordinates. A detector GK hint is only
                            # a candidate at this stage.
                            by_id[obs.track_id] = TeamAssignment(
                                track_id=obs.track_id,
                                team=UNKNOWN,
                                role=PLAYER,
                                confidence=0.0,
                                raw_team=UNKNOWN,
                                raw_confidence=0.0,
                                reason="goalkeeper_hint_waiting_for_pitch",
                            )

                assignments_for_frame = []
                team_counts = Counter()

                for obs in observations:
                    a = by_id.get(obs.track_id)
                    if a is None:
                        continue

                    tr = track_by_id.get(obs.track_id)
                    if tr is not None:
                        draw_label(frame, tr["bbox_xyxy"], a)

                    data = assignment_to_dict(a)
                    source_track = dict(tr) if tr is not None else {"track_id": obs.track_id}
                    source_track["team_v2"] = data
                    assignments_for_frame.append(source_track)

                    stats["assignments"] += 1
                    if a.team == TEAM_A:
                        stats["team_a"] += 1
                        team_counts[TEAM_A] += 1
                    elif a.team == TEAM_B:
                        stats["team_b"] += 1
                        team_counts[TEAM_B] += 1
                    else:
                        stats["unknown"] += 1
                    if a.role == REFEREE:
                        stats["referee"] += 1
                    if a.role == GOALKEEPER:
                        stats["goalkeeper"] += 1
                    if a.id_switch_suspected:
                        stats["id_switch_suspected"] += 1

                    per_track_roles[a.track_id][a.role] += 1
                    per_track_teams[a.track_id][a.team] += 1

                debug = classifier.debug_state()

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width, 88),
                    (18, 18, 18),
                    -1,
                )

                ready_text = "READY" if classifier.is_ready else "BOOTSTRAP"
                ready_color = (0, 220, 0) if classifier.is_ready else (0, 190, 255)

                cv2.putText(
                    frame,
                    (
                        f"Team Classifier V2.2 | {ready_text} | "
                        f"{classifier.embedding_backend}"
                    ),
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.64,
                    ready_color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"Frame {frame_index + 1} | tracks={len(observations)} | "
                        f"A={team_counts[TEAM_A]} B={team_counts[TEAM_B]} | "
                        f"bootstrap={debug['bootstrap_samples']} | "
                        f"switches={stats['id_switch_suspected']}"
                    ),
                    (18, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "GK team assignment waits for PnLCalib pitch_xy in Radar v4",
                    (18, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(frame)

                payload = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 5),
                    "classifier": debug,
                    "tracks": assignments_for_frame,
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")

                frame_index += 1
                stats["frames"] += 1

                if frame_index == 1 or frame_index % 25 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"Processed {frame_index}"
                        f"/{total_frames if total_frames > 0 else '?'}"
                        f" | ready={classifier.is_ready}"
                        f" | A={stats['team_a']}"
                        f" | B={stats['team_b']}"
                        f" | unknown={stats['unknown']}"
                        f" | switch?={stats['id_switch_suspected']}"
                        f" | {frame_index / max(elapsed, 1e-6):.2f} FPS"
                    )

        finally:
            cap.release()
            writer.release()

    elapsed = time.perf_counter() - started

    print("=" * 82)
    print("DONE - Team Classifier V2.2 REF-CONSENSUS")
    print(f"Frames processed       : {stats['frames']}")
    print(f"Unique track IDs       : {len(unique_ids)}")
    print(f"Assignments            : {stats['assignments']}")
    print(f"TEAM_A samples         : {stats['team_a']}")
    print(f"TEAM_B samples         : {stats['team_b']}")
    print(f"UNKNOWN samples        : {stats['unknown']}")
    print(f"Referee samples        : {stats['referee']}")
    print(f"Goalkeeper samples     : {stats['goalkeeper']}")
    print(f"ID-switch suspicions   : {stats['id_switch_suspected']}")
    print(f"Embedding backend      : {classifier.embedding_backend}")
    print(f"Classifier ready       : {classifier.is_ready}")
    print(f"Elapsed                : {elapsed:.2f} s")
    print(f"Average FPS            : {stats['frames'] / max(elapsed, 1e-6):.2f}")

    referee_tracks = []
    for track_id, counts in per_track_roles.items():
        ref_frames = counts.get(REFEREE, 0)
        if ref_frames > 0:
            referee_tracks.append((ref_frames, track_id, counts, per_track_teams[track_id]))
    referee_tracks.sort(reverse=True)

    if referee_tracks:
        print("Referee track summary  :")
        for ref_frames, track_id, role_counts, team_counts in referee_tracks[:8]:
            print(
                f"  ID {track_id:>3} | REF={ref_frames:>3} | "
                f"roles={dict(role_counts)} | teams={dict(team_counts)}"
            )
    else:
        print("Referee track summary  : none")

    print(f"Video output           : {output_path.resolve()}")
    print(f"JSONL output           : {jsonl_path.resolve()}")
    print("=" * 82)


if __name__ == "__main__":
    main()
