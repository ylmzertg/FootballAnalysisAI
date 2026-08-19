from core.possession_events import (
    TEAM_A,
    TEAM_B,
    LOOSE,
    CONTROL,
    CONTROL_GAP,
    PASS_FLIGHT,
    CONTESTED_FLIGHT,
    ControlFrame,
    PossessionEventConfig,
    PossessionEventReconstructor,
)


def c(
    frame,
    state,
    owner,
    team,
    x,
    y,
):
    return ControlFrame(
        frame_index=frame,
        state=state,
        possessor_track_id=owner,
        possessor_team=team,
        ball_image_xy=(x, y),
        ball_detected=True,
        ball_predicted=False,
    )


def loose(frame, x, y):
    return ControlFrame(
        frame_index=frame,
        state=LOOSE,
        possessor_track_id=None,
        possessor_team=None,
        ball_image_xy=(x, y),
        ball_detected=True,
        ball_predicted=False,
    )


def test_same_owner_gap_becomes_control_gap():
    r = PossessionEventReconstructor()

    frames = [
        c(0, TEAM_A, 1, TEAM_A, 10, 10),
        loose(1, 11, 10),
        loose(2, 12, 10),
        c(3, TEAM_A, 1, TEAM_A, 13, 10),
    ]

    out = r.reconstruct(frames)

    assert out[0].phase == CONTROL
    assert out[1].phase == CONTROL_GAP
    assert out[2].phase == CONTROL_GAP
    assert out[1].team_state == TEAM_A


def test_same_team_different_owner_motion_becomes_pass():
    r = PossessionEventReconstructor(
        PossessionEventConfig(
            min_ball_motion_px=5.0,
        )
    )

    frames = [
        c(0, TEAM_A, 1, TEAM_A, 10, 10),
        loose(1, 15, 10),
        loose(2, 20, 10),
        c(3, TEAM_A, 2, TEAM_A, 25, 10),
    ]

    out = r.reconstruct(frames)

    assert out[1].phase == PASS_FLIGHT
    assert out[2].phase == PASS_FLIGHT
    assert out[1].team_state == TEAM_A
    assert out[1].possessor_track_id is None


def test_team_change_gap_stays_contested():
    r = PossessionEventReconstructor()

    frames = [
        c(0, TEAM_A, 1, TEAM_A, 10, 10),
        loose(1, 15, 10),
        loose(2, 20, 10),
        c(3, TEAM_B, 9, TEAM_B, 25, 10),
    ]

    out = r.reconstruct(frames)

    assert out[1].phase == CONTESTED_FLIGHT
    assert out[1].team_state == LOOSE
