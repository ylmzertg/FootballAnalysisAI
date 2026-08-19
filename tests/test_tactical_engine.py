
from core.tactical_engine import (
    TEAM_A,
    TEAM_B,
    TacticalConfig,
    TacticalEngine,
    TacticalPlayer,
    point_segment_metrics,
)


def p(tid, team, x, y):
    return TacticalPlayer(
        track_id=tid,
        team=team,
        role="PLAYER",
        pitch_xy=(x, y),
    )


def test_pressure_high_when_opponent_is_close():
    engine = TacticalEngine()

    result = engine.pressure(
        p(1, TEAM_A, 10, 10),
        [
            p(10, TEAM_B, 11.0, 10),
            p(11, TEAM_B, 20, 20),
        ],
    )

    assert result.level == "HIGH"
    assert result.nearest_opponent_id == 10
    assert result.opponents_within_3m == 1


def test_pressure_medium_with_multiple_outer_opponents():
    engine = TacticalEngine()

    result = engine.pressure(
        p(1, TEAM_A, 10, 10),
        [
            p(10, TEAM_B, 14, 10),
            p(11, TEAM_B, 10, 14.5),
        ],
    )

    assert result.level == "MEDIUM"
    assert result.opponents_within_5m == 2


def test_passing_lane_blocked_by_defender_in_corridor():
    engine = TacticalEngine(
        TacticalConfig(
            lane_half_width_m=1.25,
        )
    )

    possessor = p(1, TEAM_A, 0, 0)
    receiver = p(2, TEAM_A, 10, 0)
    defender = p(10, TEAM_B, 5, 0.7)

    lanes = engine.passing_lanes(
        possessor,
        [possessor, receiver],
        [defender],
    )

    assert len(lanes) == 1
    assert lanes[0].status == "BLOCKED"
    assert lanes[0].blocker_track_ids == (10,)


def test_opponent_beyond_receiver_does_not_block():
    engine = TacticalEngine()

    possessor = p(1, TEAM_A, 0, 0)
    receiver = p(2, TEAM_A, 10, 0)
    defender = p(10, TEAM_B, 12, 0)

    lanes = engine.passing_lanes(
        possessor,
        [possessor, receiver],
        [defender],
    )

    assert lanes[0].status == "OPEN"


def test_point_segment_projection_parameter():
    clearance, t = point_segment_metrics(
        (5, 2),
        (0, 0),
        (10, 0),
    )

    assert abs(clearance - 2.0) < 1e-6
    assert abs(t - 0.5) < 1e-6
