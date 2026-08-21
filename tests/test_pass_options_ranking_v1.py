from core.pass_options_ranking_v1 import (
    BEST,
    BLOCKED,
    GOOD,
    RISKY,
    PassLane,
    PassOptionsRankerV1,
    PassPlayer,
)


def p(tid, team, x, y):
    return PassPlayer(
        track_id=tid,
        team=team,
        pitch_xy=(x, y),
    )


def lane(
    receiver,
    status="OPEN",
    distance=15.0,
    clearance=4.0,
    score=0.8,
):
    return PassLane(
        receiver_track_id=receiver,
        status=status,
        distance_m=distance,
        nearest_blocker_clearance_m=clearance,
        blocker_track_ids=(),
        tactical_score=score,
    )


def test_progressive_open_option_becomes_best():
    ranker = PassOptionsRankerV1()

    options = ranker.rank(
        possessor=p(1, "TEAM_A", 50, 34),
        players=[
            p(1, "TEAM_A", 50, 34),
            p(2, "TEAM_A", 70, 30),
            p(3, "TEAM_A", 45, 40),
            p(9, "TEAM_B", 60, 50),
            p(10, "TEAM_B", 85, 40),
        ],
        lanes=[
            lane(2, distance=20, clearance=5.0, score=0.9),
            lane(3, distance=8, clearance=5.0, score=0.9),
        ],
        attack_direction="PLUS_X",
    )

    assert options[0].receiver_track_id == 2
    assert options[0].category == BEST


def test_blocked_lane_is_never_best():
    ranker = PassOptionsRankerV1()

    options = ranker.rank(
        possessor=p(1, "TEAM_A", 50, 34),
        players=[
            p(1, "TEAM_A", 50, 34),
            p(2, "TEAM_A", 70, 30),
            p(9, "TEAM_B", 60, 32),
        ],
        lanes=[
            PassLane(
                receiver_track_id=2,
                status="BLOCKED",
                distance_m=20,
                nearest_blocker_clearance_m=0.8,
                blocker_track_ids=(9,),
                tactical_score=0.95,
            )
        ],
        attack_direction="PLUS_X",
    )

    assert options[0].category == BLOCKED


def test_tightly_marked_receiver_is_penalized():
    ranker = PassOptionsRankerV1()

    options = ranker.rank(
        possessor=p(1, "TEAM_A", 50, 34),
        players=[
            p(1, "TEAM_A", 50, 34),
            p(2, "TEAM_A", 68, 30),
            p(3, "TEAM_A", 66, 50),
            p(9, "TEAM_B", 68.5, 30.5),
            p(10, "TEAM_B", 85, 55),
        ],
        lanes=[
            lane(2, distance=18, clearance=4.0, score=0.9),
            lane(3, distance=18, clearance=4.0, score=0.9),
        ],
        attack_direction="PLUS_X",
    )

    by_id = {
        option.receiver_track_id: option
        for option in options
    }

    assert by_id[3].score > by_id[2].score


def test_only_one_best_option():
    ranker = PassOptionsRankerV1()

    options = ranker.rank(
        possessor=p(1, "TEAM_A", 40, 34),
        players=[
            p(1, "TEAM_A", 40, 34),
            p(2, "TEAM_A", 60, 20),
            p(3, "TEAM_A", 62, 45),
            p(9, "TEAM_B", 80, 60),
        ],
        lanes=[
            lane(2, distance=20, clearance=5, score=0.95),
            lane(3, distance=22, clearance=5, score=0.95),
        ],
        attack_direction="PLUS_X",
    )

    assert sum(
        1
        for option in options
        if option.category == BEST
    ) <= 1
