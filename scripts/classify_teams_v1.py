from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from core.team_classifier import TeamColorClusterer, TeamClassifierConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - automatic Team A/B classification"
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
        "--scenes",
        default=r"output\scene_labels_v2.jsonl",
    )
    p.add_argument(
        "--output",
        default=r"output\team_classification.mp4",
    )
    p.add_argument(
        "--jsonl",
        default=r"output\team_classification.jsonl",
    )
    p.add_argument(
        "--profile",
        default=r"output\team_profile.json",
    )
    p.add_argument(
        "--max-samples-per-track",
        type=int,
        default=12,
    )
    p.add_argument(
        "--min-samples-per-track",
        type=int,
        default=2,
    )
    return p.parse_args()


def read_jsonl(path: Path) -> dict[int, dict]:
    rows = {}

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            rows[int(item["frame_index"])] = item

    return rows


def team_label(team_id: int | None) -> str:
    if team_id is None:
        return "UNKNOWN"
    return "TEAM_A" if team_id == 0 else "TEAM_B"


def team_display_color(team_id: int | None):
    # Display colors intentionally fixed and high-contrast.
    # Representative kit colors are still saved separately in team_profile.json.
    if team_id == 0:
        return (255, 80, 80)   # BGR
    if team_id == 1:
        return (80, 220, 255)  # BGR
    return (170, 170, 170)


def bbox_center(track: dict):
    bbox = track.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = map(float, bbox)

    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ],
        dtype=np.float32,
    )


def assign_goalkeeper_team(
    goalkeeper: dict,
    classified_players: list[dict],
) -> int | None:
    g_center = bbox_center(goalkeeper)

    if g_center is None:
        return None

    distances = {
        0: [],
        1: [],
    }

    for player in classified_players:
        team_id = player.get("team_id")

        if team_id not in (0, 1):
            continue

        p_center = bbox_center(player)

        if p_center is None:
            continue

        distance = float(np.linalg.norm(g_center - p_center))
        distances[team_id].append(distance)

    scores = {}

    for team_id, values in distances.items():
        if not values:
            scores[team_id] = float("inf")
            continue

        values.sort()
        scores[team_id] = float(np.mean(values[:3]))

    if not np.isfinite(min(scores.values())):
        return None

    return min(scores, key=scores.get)


def draw_ellipse(
    frame,
    track,
    color,
    label,
):
    bbox = track.get("bbox_xyxy")

    if not bbox or len(bbox) != 4:
        return

    x1, y1, x2, y2 = map(int, bbox)

    foot_x = int(round((x1 + x2) / 2))
    foot_y = y2

    width = max(10, int((x2 - x1) * 0.55))

    cv2.ellipse(
        frame,
        (foot_x, foot_y),
        (width, max(5, width // 3)),
        0,
        0,
        360,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        label,
        (max(0, foot_x - 24), max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        2,
        cv2.LINE_AA,
    )


def collect_training_samples(
    *,
    source: Path,
    tracking: dict[int, dict],
    scenes: dict[int, dict],
    classifier: TeamColorClusterer,
):
    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")

    frame_index = 0
    collected = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        scene = scenes.get(frame_index)

        if scene is not None:
            if not bool(scene.get("analysis_enabled", False)):
                frame_index += 1
                continue

        tracks = tracking.get(
            frame_index,
            {"tracks": []},
        ).get("tracks", [])

        for tr in tracks:
            if tr.get("class_name") != "player":
                continue

            bbox = tr.get("bbox_xyxy")

            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(float, bbox)

            if (
                y2 - y1 < classifier.config.min_bbox_height
                or x2 - x1 < classifier.config.min_bbox_width
            ):
                continue

            track_id = int(tr.get("track_id", -1))

            if track_id < 0:
                continue

            crop = classifier.crop_torso(frame, bbox)

            if classifier.add_sample(track_id, crop):
                collected += 1

        frame_index += 1

    cap.release()

    return collected


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]

    source = Path(args.source)
    tracking_path = (
        Path(args.tracking)
        if Path(args.tracking).is_absolute()
        else project_root / args.tracking
    )
    scenes_path = (
        Path(args.scenes)
        if Path(args.scenes).is_absolute()
        else project_root / args.scenes
    )
    output_path = (
        Path(args.output)
        if Path(args.output).is_absolute()
        else project_root / args.output
    )
    jsonl_path = (
        Path(args.jsonl)
        if Path(args.jsonl).is_absolute()
        else project_root / args.jsonl
    )
    profile_path = (
        Path(args.profile)
        if Path(args.profile).is_absolute()
        else project_root / args.profile
    )

    if not source.exists():
        raise FileNotFoundError(source)

    if not tracking_path.exists():
        raise FileNotFoundError(tracking_path)

    tracking = read_jsonl(tracking_path)
    scenes = read_jsonl(scenes_path)

    config = TeamClassifierConfig(
        max_samples_per_track=args.max_samples_per_track,
        min_samples_per_track=args.min_samples_per_track,
    )

    classifier = TeamColorClusterer(config)

    print("=" * 76)
    print("FootballAnalysisAI - Team A/B Classification")
    print(f"Source   : {source}")
    print(f"Tracking : {tracking_path}")
    print(f"Scenes   : {scenes_path if scenes_path.exists() else 'not used'}")
    print("=" * 76)

    print("[1/3] Collecting torso samples...")

    collected = collect_training_samples(
        source=source,
        tracking=tracking,
        scenes=scenes,
        classifier=classifier,
    )

    print(
        f"Collected {collected} crops from "
        f"{len(classifier.track_features)} track IDs."
    )

    print("[2/3] Fitting PCA + KMeans...")

    classifier.fit()

    print("Track assignments:")

    for track_id, team_id in sorted(classifier.track_team.items()):
        print(
            f"  ID {track_id:>3} -> "
            f"{team_label(team_id)}"
        )

    for team_id in (0, 1):
        print(
            f"{team_label(team_id)} representative BGR: "
            f"{classifier.team_colors_bgr.get(team_id)}"
        )

    print("[3/3] Rendering classified video...")

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(source)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    team_counts = defaultdict(int)

    with jsonl_path.open("w", encoding="utf-8") as out_jsonl:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            tracks = tracking.get(
                frame_index,
                {"tracks": []},
            ).get("tracks", [])

            output_tracks = []
            classified_players = []
            unknown_player_candidates = []

            # First pass: regular players.
            for tr in tracks:
                item = dict(tr)
                class_name = item.get("class_name")

                if class_name == "player":
                    track_id = int(item.get("track_id", -1))
                    team_id = classifier.get_track_team(track_id)

                    # Track may be too short to have entered the fit.
                    if team_id is None:
                        bbox = item.get("bbox_xyxy")

                        if bbox:
                            crop = classifier.crop_torso(frame, bbox)
                            team_id = classifier.predict_crop(crop)

                    item["team_id"] = team_id
                    item["team_name"] = team_label(team_id)

                    if team_id in (0, 1):
                        classified_players.append(item)
                        team_counts[team_id] += 1

                output_tracks.append(item)

            # Second pass: goalkeepers, using nearest classified outfield players.
            for item in output_tracks:
                if item.get("class_name") != "goalkeeper":
                    continue

                team_id = assign_goalkeeper_team(
                    item,
                    classified_players,
                )

                item["team_id"] = team_id
                item["team_name"] = team_label(team_id)

            # Referees remain team-less.
            for item in output_tracks:
                if item.get("class_name") == "referee":
                    item["team_id"] = None
                    item["team_name"] = "REFEREE"

            # Draw.
            for item in output_tracks:
                class_name = item.get("class_name")
                track_id = int(item.get("track_id", -1))

                if class_name == "referee":
                    color = (0, 255, 255)
                    label = f"REF {track_id}"
                else:
                    team_id = item.get("team_id")
                    color = team_display_color(team_id)

                    if class_name == "goalkeeper":
                        label = (
                            f"GK {track_id} "
                            f"{team_label(team_id)}"
                        )
                    else:
                        label = (
                            f"{track_id} "
                            f"{team_label(team_id)}"
                        )

                draw_ellipse(
                    frame,
                    item,
                    color,
                    label,
                )

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 56),
                (18, 18, 18),
                -1,
            )

            cv2.putText(
                frame,
                "TEAM_A / TEAM_B AUTO CLASSIFICATION",
                (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

            payload = {
                "frame_index": frame_index,
                "timestamp_seconds": round(
                    frame_index / fps,
                    5,
                ),
                "tracks": output_tracks,
            }

            out_jsonl.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )

            frame_index += 1

            if frame_index == 1 or frame_index % 25 == 0:
                print(
                    f"Rendered {frame_index}/{total_frames}"
                )

    cap.release()
    writer.release()

    profile = {
        "engine": "color_histogram_pca_kmeans_v1",
        "team_names": {
            "0": "TEAM_A",
            "1": "TEAM_B",
        },
        "representative_colors_bgr": {
            str(team_id): list(
                classifier.team_colors_bgr.get(
                    team_id,
                    (128, 128, 128),
                )
            )
            for team_id in (0, 1)
        },
        "track_team_assignments": {
            str(track_id): team_label(team_id)
            for track_id, team_id in sorted(
                classifier.track_team.items()
            )
        },
        "note": (
            "TEAM_A / TEAM_B labels are cluster identities, "
            "not club names yet."
        ),
    }

    profile_path.write_text(
        json.dumps(
            profile,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 76)
    print("DONE")
    print(f"Frames       : {frame_index}")
    print(f"Video        : {output_path.resolve()}")
    print(f"Team JSONL   : {jsonl_path.resolve()}")
    print(f"Team profile : {profile_path.resolve()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
