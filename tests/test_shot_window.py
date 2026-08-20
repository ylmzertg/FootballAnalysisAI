from core.shot_window import (
    SHOT_FLIGHT, GOAL_ATTEMPT,
    LocalGoalSample, LocalShotWindowDetector
)

def s(values, start=0):
    return [LocalGoalSample(start+i, v) for i,v in enumerate(values)]

def test_five_frame_local_shot_window():
    best = LocalShotWindowDetector().best_window(
        s([7.960, 7.420, 6.980, 6.500, 6.380], 201)
    )
    assert best is not None
    assert best.classification == SHOT_FLIGHT
    assert (best.start_frame, best.end_frame) == (201, 205)

def test_goal_attempt_requires_absolute_proximity():
    best = LocalShotWindowDetector().best_window(
        s([3.8, 3.0, 2.3, 1.7, 1.3])
    )
    assert best is not None
    assert best.classification == GOAL_ATTEMPT

def test_noise_does_not_create_shot():
    best = LocalShotWindowDetector().best_window(
        s([7.0, 7.1, 6.95, 7.08, 7.0, 6.98, 7.05, 7.02])
    )
    assert best is None
