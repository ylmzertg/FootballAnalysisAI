from core.possession import (
    LOOSE,
    TEAM_A,
    TEAM_B,
    PlayerPossessionCandidate,
    PossessionConfig,
    PossessionEstimator,
)


def p(
    tid,
    team,
    pitch_x,
    pitch_y,
    *,
    foot=(100, 100),
    h=100,
):
    return PlayerPossessionCandidate(
        track_id=tid,
        team=team,
        role="PLAYER",
        pitch_xy=(pitch_x, pitch_y),
        image_foot_xy=foot,
        bbox_height_px=h,
    )


def test_image_space_can_acquire_when_pitch_projection_is_bad():
    e = PossessionEstimator(
        PossessionConfig(
            confirm_frames=1,
            acquire_distance_m=2.0,
            acquire_image_ratio=0.75,
        )
    )

    result = e.update(
        (40, 40),
        [
            p(
                1,
                TEAM_A,
                10,
                10,
                foot=(105, 100),
                h=100,
            )
        ],
        ball_detected=True,
        ball_predicted=False,
        ball_image_xy=(100, 100),
    )

    assert result.state == TEAM_A
    assert result.possessor_track_id == 1
    assert result.control_source == "BOTH"


def test_image_only_control_when_ball_pitch_is_missing():
    e = PossessionEstimator(
        PossessionConfig(
            confirm_frames=1,
        )
    )

    result = e.update(
        None,
        [
            p(
                7,
                TEAM_B,
                60,
                40,
                foot=(210, 200),
                h=100,
            )
        ],
        ball_detected=True,
        ball_predicted=False,
        ball_image_xy=(205, 200),
    )

    assert result.state == TEAM_B
    assert result.possessor_track_id == 7
    assert result.control_source == "IMAGE"


def test_image_ratio_is_scale_normalized():
    e = PossessionEstimator(
        PossessionConfig(
            confirm_frames=1,
            acquire_image_ratio=0.75,
        )
    )

    near_small = p(
        1,
        TEAM_A,
        10,
        10,
        foot=(100, 100),
        h=40,
    )
    near_large = p(
        2,
        TEAM_B,
        20,
        20,
        foot=(130, 100),
        h=120,
    )

    result = e.update(
        None,
        [near_small, near_large],
        ball_detected=True,
        ball_predicted=False,
        ball_image_xy=(100, 100),
    )

    assert result.possessor_track_id == 1


def test_role_filter_still_excludes_referee():
    e = PossessionEstimator(
        PossessionConfig(confirm_frames=1)
    )

    referee = PlayerPossessionCandidate(
        track_id=99,
        team=TEAM_A,
        role="REFEREE",
        pitch_xy=(10, 10),
        image_foot_xy=(100, 100),
        bbox_height_px=100,
    )

    player = p(
        2,
        TEAM_B,
        20,
        20,
        foot=(120, 100),
        h=100,
    )

    result = e.update(
        None,
        [referee, player],
        ball_detected=True,
        ball_predicted=False,
        ball_image_xy=(100, 100),
    )

    assert result.possessor_track_id != 99
