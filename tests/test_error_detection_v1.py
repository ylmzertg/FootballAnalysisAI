from core.error_detection_v1 import (
    FREE_PASSING_LANE,
    LATE_PRESSURE,
    TEAM_A,
    TEAM_B,
    UNMARKED_RUNNER,
    ErrorDetectorV1,
    PlayerState,
    TacticalLane,
)


def p(tid, team, x, y):
    return PlayerState(
        track_id=tid,
        team=team,
        pitch_xy=(x, y),
    )


def test_late_pressure_detected():
    detector = ErrorDetectorV1()
    events = detector.detect(
        frame_index=10,
        attacking_team=TEAM_A,
        possessor_track_id=1,
        players=[
            p(1, TEAM_A, 70, 34),
            p(2, TEAM_B, 76, 34),
            p(3, TEAM_B, 82, 40),
        ],
        attack_direction="PLUS_X",
    )
    assert any(e.error_type == LATE_PRESSURE for e in events)


def test_unmarked_runner_detected():
    detector = ErrorDetectorV1()
    events = detector.detect(
        frame_index=20,
        attacking_team=TEAM_A,
        possessor_track_id=1,
        players=[
            p(1, TEAM_A, 55, 34),
            p(7, TEAM_A, 68, 20),
            p(10, TEAM_B, 60, 45),
            p(11, TEAM_B, 75, 35),
        ],
        attack_direction="PLUS_X",
    )
    assert any(e.error_type == UNMARKED_RUNNER for e in events)


def test_open_advanced_lane_detected():
    detector = ErrorDetectorV1()
    events = detector.detect(
        frame_index=30,
        attacking_team=TEAM_A,
        possessor_track_id=1,
        players=[
            p(1, TEAM_A, 50, 34),
            p(7, TEAM_A, 70, 20),
            p(10, TEAM_B, 60, 45),
            p(11, TEAM_B, 72, 35),
        ],
        attack_direction="PLUS_X",
        passing_lanes=[
            TacticalLane(
                receiver_track_id=7,
                status="OPEN",
                distance_m=24.0,
                blocker_track_ids=(),
            )
        ],
    )
    assert any(e.error_type == FREE_PASSING_LANE for e in events)


def test_no_error_when_possessor_missing():
    detector = ErrorDetectorV1()
    events = detector.detect(
        frame_index=40,
        attacking_team=TEAM_A,
        possessor_track_id=None,
        players=[],
        attack_direction="PLUS_X",
    )
    assert events == []
