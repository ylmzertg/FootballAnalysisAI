import numpy as np

from core.team_classifier_v23 import (
    DetectionObservation,
    PLAYER,
    REFEREE,
    TeamClassifierV2,
    TeamClassifierV2Config,
)


def _base_scene():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (45, 115, 45)

    obs = []
    tid = 1

    for team_idx in range(2):
        for row in range(5):
            x = 60 + team_idx * 320 + (row % 2) * 35
            y = 45 + row * 55
            colour = (0, 0, 220) if team_idx == 0 else (220, 40, 20)

            frame[y:y + 44, x:x + 22] = colour

            obs.append(
                DetectionObservation(
                    tid,
                    (x, y, x + 22, y + 44),
                    confidence=0.95,
                    pitch_xy=(15 + team_idx * 75, 10 + row * 10),
                    role_hint=PLAYER,
                )
            )
            tid += 1

    return frame, obs


def _classifier():
    return TeamClassifierV2(
        TeamClassifierV2Config(
            use_deep_embedding=False,
            bootstrap_min_samples=18,
            bootstrap_samples_per_track=3,
            embedding_stride=1,
            referee_min_frames=3,
            referee_hint_min_frames=3,
            referee_hint_min_total_frames=4,
            referee_hint_min_ratio=0.75,
            referee_hint_min_novelty=0.10,
            referee_min_novelty=0.20,
            referee_candidate_prototype_guard_similarity=0.75,
            role_outlier_colour_similarity=0.75,
        )
    )


def test_v23_off_pitch_referee_hints_never_become_match_referee():
    clf = _classifier()

    for frame_idx in range(4):
        frame, obs = _base_scene()
        clf.classify_frame(frame, obs, frame_idx)

    assert clf.is_ready

    roles = []

    for frame_idx in range(4, 18):
        frame, obs = _base_scene()

        # Yellow sideline official / staff.
        # PnLCalib spatial coordinate says the foot point is OUTSIDE the
        # canonical 105x68 pitch.
        frame[185:233, 300:324] = (0, 230, 230)

        obs.append(
            DetectionObservation(
                200,
                (300, 185, 324, 233),
                confidence=0.98,
                pitch_xy=(109.0, 34.0),
                role_hint=REFEREE,
            )
        )

        assignments = clf.classify_frame(frame, obs, frame_idx)
        roles.append(
            next(a for a in assignments if a.track_id == 200).role
        )

    assert REFEREE not in roles
    assert not clf.track_states[200].referee_trusted


def test_v23_intermittent_on_pitch_referee_hints_can_become_trusted():
    clf = _classifier()

    # V2.3 on-pitch referee consensus. The real match referee may flicker
    # between detector labels PLAYER and REFEREE, so on-pitch temporal evidence
    # is allowed to override global class purity.
    clf.config.referee_on_pitch_hint_min_frames = 3
    clf.config.referee_on_pitch_min_total_frames = 10
    clf.config.referee_on_pitch_min_novelty = 0.10

    for frame_idx in range(4):
        frame, obs = _base_scene()
        clf.classify_frame(frame, obs, frame_idx)

    assert clf.is_ready

    roles = []
    ref_frames = {4, 8, 13}

    for frame_idx in range(4, 14):
        frame, obs = _base_scene()

        # Distinct yellow referee who is physically inside the pitch.
        frame[185:233, 300:324] = (0, 230, 230)

        hint = REFEREE if frame_idx in ref_frames else PLAYER

        obs.append(
            DetectionObservation(
                201,
                (300, 185, 324, 233),
                confidence=0.98,
                pitch_xy=(52.5, 34.0),
                role_hint=hint,
            )
        )

        assignments = clf.classify_frame(frame, obs, frame_idx)

        roles.append(
            next(a for a in assignments if a.track_id == 201).role
        )

    state = clf.track_states[201]

    assert state.referee_hint_frames == 3
    assert state.referee_on_pitch_hint_frames == 3
    assert state.on_pitch_seen_frames == 10
    assert state.referee_trusted
    assert roles[-1] == REFEREE
