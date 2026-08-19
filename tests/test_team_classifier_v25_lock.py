from core.team_classifier_v25 import (
    TEAM_A,
    TEAM_B,
    TeamClassifierV2,
    TeamClassifierV2Config,
    _Vote,
)


def test_stable_team_lock_prevents_flicker():
    clf = TeamClassifierV2(
        TeamClassifierV2Config(
            use_deep_embedding=False,
            team_lock_min_votes=5,
            team_lock_min_ratio=0.70,
            team_lock_min_margin=0.15,
        )
    )
    state = clf._state(20)

    for frame in range(5):
        state.votes.append(_Vote(TEAM_A, 0.90, frame))

    team, _ = clf._temporal_team(state, TEAM_A, 0.90, 4)
    assert team == TEAM_A
    assert state.stable_team == TEAM_A

    for frame in range(5, 9):
        state.votes.append(_Vote(TEAM_B, 0.95, frame))
        team, _ = clf._temporal_team(state, TEAM_B, 0.95, frame)
        assert team == TEAM_A


def test_credible_id_switch_releases_stable_team():
    clf = TeamClassifierV2(
        TeamClassifierV2Config(use_deep_embedding=False)
    )
    state = clf._state(7)
    state.stable_team = TEAM_A
    state.stable_team_confidence = 0.92
    state.stable_team_locked_frame = 10

    clf._handle_possible_switch(state, strength=1.05)

    assert state.stable_team is None
    assert state.stable_team_confidence == 0.0


def test_soft_noise_below_switch_strength_keeps_lock():
    clf = TeamClassifierV2(
        TeamClassifierV2Config(use_deep_embedding=False)
    )
    state = clf._state(8)
    state.stable_team = TEAM_B
    state.stable_team_confidence = 0.88

    clf._handle_possible_switch(state, strength=0.90)

    assert state.stable_team == TEAM_B
