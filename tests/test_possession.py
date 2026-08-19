
from core.possession import (
    LOOSE,
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    PlayerPossessionCandidate,
    PossessionConfig,
    PossessionEstimator,
)


def p(track_id, team, x, y, role="PLAYER"):
    return PlayerPossessionCandidate(
        track_id=track_id,
        team=team,
        role=role,
        pitch_xy=(x,y),
    )


def test_acquire_requires_confirmation():
    e = PossessionEstimator(PossessionConfig(confirm_frames=2))
    players = [p(1, TEAM_A, 10,10)]
    r0 = e.update((10.5,10), players, ball_detected=True, ball_predicted=False)
    assert r0.state == LOOSE
    r1 = e.update((10.4,10), players, ball_detected=True, ball_predicted=False)
    assert r1.state == TEAM_A
    assert r1.possessor_track_id == 1


def test_owner_hysteresis_prevents_small_switch():
    e = PossessionEstimator(PossessionConfig(confirm_frames=1, switch_margin_m=0.75))
    r0 = e.update(
        (10,10),
        [p(1,TEAM_A,10.5,10), p(2,TEAM_B,12,10)],
        ball_detected=True,
        ball_predicted=False,
    )
    assert r0.state == TEAM_A

    r1 = e.update(
        (10,10),
        [p(1,TEAM_A,11.0,10), p(2,TEAM_B,10.6,10)],
        ball_detected=True,
        ball_predicted=False,
    )
    assert r1.state == TEAM_A


def test_clear_switch_can_change_team():
    e = PossessionEstimator(
        PossessionConfig(confirm_frames=2, switch_margin_m=0.5)
    )
    e.update(
        (10,10),
        [p(1,TEAM_A,10.3,10)],
        ball_detected=True,
        ball_predicted=False,
    )
    e.update(
        (10,10),
        [p(1,TEAM_A,10.3,10)],
        ball_detected=True,
        ball_predicted=False,
    )

    r2 = e.update(
        (20,20),
        [p(1,TEAM_A,24,20), p(2,TEAM_B,20.4,20)],
        ball_detected=True,
        ball_predicted=False,
    )
    assert r2.state in {TEAM_A, LOOSE}

    r3 = e.update(
        (20,20),
        [p(1,TEAM_A,24,20), p(2,TEAM_B,20.3,20)],
        ball_detected=True,
        ball_predicted=False,
    )
    assert r3.state == TEAM_B
    assert r3.possessor_track_id == 2


def test_short_missing_ball_gap_holds_owner():
    e = PossessionEstimator(
        PossessionConfig(confirm_frames=1, hold_missing_ball_frames=2)
    )
    r0 = e.update(
        (10,10),
        [p(1,TEAM_A,10.2,10)],
        ball_detected=True,
        ball_predicted=False,
    )
    assert r0.state == TEAM_A

    r1 = e.update(
        None,
        [p(1,TEAM_A,10.2,10)],
        ball_detected=False,
        ball_predicted=False,
    )
    assert r1.state == TEAM_A

    r2 = e.update(
        None,
        [p(1,TEAM_A,10.2,10)],
        ball_detected=False,
        ball_predicted=False,
    )
    assert r2.state == TEAM_A

    r3 = e.update(
        None,
        [p(1,TEAM_A,10.2,10)],
        ball_detected=False,
        ball_predicted=False,
    )
    assert r3.state == UNKNOWN
