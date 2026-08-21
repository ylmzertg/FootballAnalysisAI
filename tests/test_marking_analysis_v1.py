from core.marking_analysis_v1 import (
    MARKED,
    UNMARKED,
    MarkingAnalyzerV1,
    MarkingPlayer,
)


def p(tid, team, x, y):
    return MarkingPlayer(
        track_id=tid,
        team=team,
        pitch_xy=(x, y),
    )


def test_unmarked_advanced_player_is_high_threat():
    analyzer = MarkingAnalyzerV1()

    results = analyzer.assess(
        possessor=p(1, "TEAM_A", 50, 34),
        players=[
            p(1, "TEAM_A", 50, 34),
            p(7, "TEAM_A", 70, 25),
            p(9, "TEAM_B", 58, 50),
            p(10, "TEAM_B", 82, 40),
        ],
        attack_direction="PLUS_X",
    )

    r = next(
        x
        for x in results
        if x.attacker_track_id == 7
    )

    assert r.marking_state == UNMARKED
    assert r.dangerous is True


def test_close_defender_marks_player():
    analyzer = MarkingAnalyzerV1()

    results = analyzer.assess(
        possessor=p(1, "TEAM_A", 50, 34),
        players=[
            p(1, "TEAM_A", 50, 34),
            p(7, "TEAM_A", 62, 30),
            p(9, "TEAM_B", 64, 30),
        ],
        attack_direction="PLUS_X",
    )

    r = results[0]

    assert r.marking_state in {
        "TIGHT",
        MARKED,
    }
