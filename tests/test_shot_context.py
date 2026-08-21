from core.shot_context import (
    PLUS_X,
    MINUS_X,
    RAW_LOOSE,
    SHOT_FLIGHT,
    GOAL_ATTEMPT,
    ATTACKING_FLIGHT,
    ShotFrame,
    ShotContextClassifier,
    ShotContextConfig,
)


def f(i, x, y=34.0, team="TEAM_A"):
    return ShotFrame(
        frame_index=i,
        phase=RAW_LOOSE,
        team_state="LOOSE",
        source_team=team,
        source_owner_track_id=1,
        ball_pitch_xy=(x, y),
    )


def test_plus_x_goal_attempt():
    c = ShotContextClassifier(
        ShotContextConfig(
            min_forward_progress_m=8.0,
            goal_attempt_distance_m=12.0,
        )
    )
    frames = [f(0, 80), f(1, 90), f(2, 98)]
    r = c.classify_run(frames, PLUS_X)
    assert r.classification == GOAL_ATTEMPT


def test_minus_x_shot_flight():
    c = ShotContextClassifier(
        ShotContextConfig(
            min_forward_progress_m=8.0,
            goal_attempt_distance_m=8.0,
            shot_goal_distance_m=25.0,
        )
    )
    frames = [
        f(0, 40, team="TEAM_B"),
        f(1, 30, team="TEAM_B"),
        f(2, 20, team="TEAM_B"),
    ]
    r = c.classify_run(frames, MINUS_X)
    assert r.classification == SHOT_FLIGHT


def test_small_progress_is_not_forced_to_shot():
    c = ShotContextClassifier()
    frames = [f(0, 50), f(1, 52), f(2, 54.5)]
    r = c.classify_run(frames, PLUS_X)
    assert r.classification in {ATTACKING_FLIGHT, RAW_LOOSE}
