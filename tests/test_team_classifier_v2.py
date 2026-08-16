import numpy as np

from core.team_classifier_v2 import (
    DetectionObservation,
    GOALKEEPER,
    REFEREE,
    TEAM_A,
    TEAM_B,
    TeamClassifierV2,
    TeamClassifierV2Config,
)


def _frame_and_obs(switched_track=None):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (45, 115, 45)
    obs = []
    tid = 1
    # BGR: red team on left half, blue team on right half.
    for team_idx in range(2):
        for row in range(5):
            x = 60 + team_idx * 320 + (row % 2) * 35
            y = 45 + row * 55
            color = (0, 0, 220) if team_idx == 0 else (220, 40, 20)
            if switched_track == tid:
                color = (220, 40, 20) if team_idx == 0 else (0, 0, 220)
            frame[y:y+44, x:x+22] = color
            obs.append(DetectionObservation(tid, (x, y, x+22, y+44), pitch_xy=(15 + team_idx*75, 10+row*10)))
            tid += 1
    return frame, obs


def test_bootstrap_and_id_switch_recovery():
    cfg = TeamClassifierV2Config(
        use_deep_embedding=False,
        bootstrap_min_samples=18,
        bootstrap_samples_per_track=3,
        embedding_stride=1,
        referee_min_frames=999,
        role_outlier_similarity=-1.0,
    )
    clf = TeamClassifierV2(cfg)

    last = None
    for frame_idx in range(4):
        frame, obs = _frame_and_obs()
        last = clf.classify_frame(frame, obs, frame_idx)

    assert clf.is_ready
    labels = {a.track_id: a.team for a in last}
    left = {labels[i] for i in range(1, 6)}
    right = {labels[i] for i in range(6, 11)}
    assert len(left) == 1
    assert len(right) == 1
    assert left != right

    original = labels[1]
    opposite = TEAM_B if original == TEAM_A else TEAM_A
    switched_seen = False
    final = None
    for frame_idx in range(4, 11):
        frame, obs = _frame_and_obs(switched_track=1)
        final = clf.classify_frame(frame, obs, frame_idx)
        a1 = next(a for a in final if a.track_id == 1)
        switched_seen = switched_seen or a1.id_switch_suspected

    assert switched_seen
    assert next(a for a in final if a.track_id == 1).team == opposite


def test_goalkeeper_and_referee_role_corrections():
    cfg = TeamClassifierV2Config(
        use_deep_embedding=False,
        bootstrap_min_samples=18,
        bootstrap_samples_per_track=3,
        embedding_stride=1,
        referee_min_frames=3,
        role_outlier_similarity=0.55,
        goalkeeper_goal_line_distance_m=19.0,
    )
    clf = TeamClassifierV2(cfg)

    # Bootstrap only from the ten outfield synthetic players.
    for frame_idx in range(4):
        frame, obs = _frame_and_obs()
        last = clf.classify_frame(frame, obs, frame_idx)
    assert clf.is_ready
    left_team = next(a.team for a in last if a.track_id == 1)

    keeper_assignment = None
    referee_assignment = None
    for frame_idx in range(4, 9):
        frame, obs = _frame_and_obs()
        # Distinct yellow keeper near the left goal.
        frame[130:178, 20:44] = (0, 230, 230)
        obs.append(DetectionObservation(99, (20, 130, 44, 178), pitch_xy=(4.0, 34.0)))
        # Distinct black referee in central pitch area.
        frame[185:233, 300:324] = (25, 25, 25)
        obs.append(DetectionObservation(100, (300, 185, 324, 233), pitch_xy=(52.5, 34.0)))
        assignments = clf.classify_frame(frame, obs, frame_idx)
        keeper_assignment = next(a for a in assignments if a.track_id == 99)
        referee_assignment = next(a for a in assignments if a.track_id == 100)

    assert keeper_assignment.role == GOALKEEPER
    assert keeper_assignment.team == left_team
    assert referee_assignment.role == REFEREE
