import numpy as np

from core.team_classifier_v2 import (
    DetectionObservation,
    PLAYER,
    REFEREE,
    TeamClassifierV2,
    TeamClassifierV2Config,
)


def _scene(include_referee: bool = False):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (45, 115, 45)

    observations = []
    tid = 1
    for team_idx in range(2):
        for row in range(5):
            x = 60 + team_idx * 320 + (row % 2) * 35
            y = 45 + row * 55
            colour = (0, 0, 220) if team_idx == 0 else (220, 40, 20)
            frame[y:y + 44, x:x + 22] = colour
            observations.append(
                DetectionObservation(
                    tid,
                    (x, y, x + 22, y + 44),
                    pitch_xy=(15 + team_idx * 75, 10 + row * 10),
                )
            )
            tid += 1

    if include_referee:
        frame[185:233, 300:324] = (25, 25, 25)
        observations.append(
            DetectionObservation(
                100,
                (300, 185, 324, 233),
                pitch_xy=(52.5, 34.0),
            )
        )

    return frame, observations


def test_referee_candidate_does_not_spread_to_normal_players():
    cfg = TeamClassifierV2Config(
        use_deep_embedding=False,
        bootstrap_min_samples=18,
        bootstrap_samples_per_track=3,
        embedding_stride=1,
        referee_min_frames=4,
        role_outlier_similarity=0.55,
        role_outlier_colour_similarity=0.70,
    )
    clf = TeamClassifierV2(cfg)

    # Bootstrap the two teams first.
    for frame_idx in range(4):
        frame, obs = _scene(False)
        assignments = clf.classify_frame(frame, obs, frame_idx)

    assert clf.is_ready

    # Introduce one persistent appearance outlier long enough to become referee.
    for frame_idx in range(4, 11):
        frame, obs = _scene(True)
        assignments = clf.classify_frame(frame, obs, frame_idx)

    referee = next(a for a in assignments if a.track_id == 100)
    assert referee.role == REFEREE

    normal_players = [a for a in assignments if a.track_id != 100]
    assert normal_players
    assert all(a.role == PLAYER for a in normal_players)
    assert all(a.team != "UNKNOWN" for a in normal_players)
