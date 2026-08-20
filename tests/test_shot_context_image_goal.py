import numpy as np

from core.shot_context_image_goal import (
    PLUS_X,
    MINUS_X,
    RAW_LOOSE,
    SHOT_FLIGHT,
    GOAL_ATTEMPT,
    GoalApproachSample,
    ImageGoalShotClassifier,
    ImageGoalShotConfig,
    goal_image_geometry,
)


def sample(frame, d):
    return GoalApproachSample(
        frame_index=frame,
        normalized_goal_distance=d,
        ball_xy=(0.0, 0.0),
        goal_center_xy=(0.0, 0.0),
        goal_mouth_width_px=100.0,
    )


def test_goal_attempt_from_strong_image_goal_closing():
    c = ImageGoalShotClassifier(
        ImageGoalShotConfig(
            goal_attempt_min_closing_units=1.3,
            goal_attempt_max_closest_units=1.8,
            goal_attempt_min_approach_fraction=0.45,
        )
    )

    r = c.classify(
        start_frame=0,
        end_frame=4,
        team="TEAM_A",
        samples=[
            sample(0, 4.0),
            sample(1, 3.2),
            sample(2, 2.4),
            sample(3, 1.7),
            sample(4, 1.2),
        ],
        fallback_phase=RAW_LOOSE,
    )

    assert r.classification == GOAL_ATTEMPT


def test_shot_flight_without_goal_attempt():
    c = ImageGoalShotClassifier()

    r = c.classify(
        start_frame=0,
        end_frame=3,
        team="TEAM_A",
        samples=[
            sample(0, 6.0),
            sample(1, 5.0),
            sample(2, 4.3),
            sample(3, 3.8),
        ],
        fallback_phase=RAW_LOOSE,
    )

    assert r.classification == SHOT_FLIGHT


def test_identity_homography_projects_plus_x_goal_to_pitch_coordinates():
    H = np.eye(3, dtype=np.float64)

    g = goal_image_geometry(
        H,
        PLUS_X,
    )

    assert g is not None
    assert abs(g.center_xy[0] - 105.0) < 1e-6
    assert abs(g.center_xy[1] - 34.0) < 1e-6
    assert g.mouth_width_px > 7.0
