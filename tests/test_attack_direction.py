
from core.attack_direction import (
    TEAM_A, ATTACK_PLUS_X, ATTACK_MINUS_X, UNKNOWN,
    DirectionPlayer, AttackDirectionResolver, DefensiveLineEstimator,
)

def p(tid,team,role,x,y):
    return DirectionPlayer(tid,team,role,(x,y))

def test_goalkeeper_near_left_goal_means_plus_x_attack():
    r = AttackDirectionResolver().resolve(
        TEAM_A,[p(1,TEAM_A,"GOALKEEPER",3,34)]
    )
    assert r.direction == ATTACK_PLUS_X
    assert r.source == "trusted_goalkeeper"

def test_goalkeeper_near_right_goal_means_minus_x_attack():
    r = AttackDirectionResolver().resolve(
        TEAM_A,[p(1,TEAM_A,"GOALKEEPER",102,34)]
    )
    assert r.direction == ATTACK_MINUS_X

def test_no_goalkeeper_means_unknown():
    r = AttackDirectionResolver().resolve(
        TEAM_A,[p(2,TEAM_A,"PLAYER",30,30)]
    )
    assert r.direction == UNKNOWN

def test_override_is_authoritative():
    r = AttackDirectionResolver().resolve(
        TEAM_A,[],override="left"
    )
    assert r.direction == ATTACK_MINUS_X
    assert r.confidence == 1.0

def test_defensive_line_uses_own_goal_side():
    players = [
        p(2,TEAM_A,"PLAYER",10,10),
        p(3,TEAM_A,"PLAYER",12,20),
        p(4,TEAM_A,"PLAYER",13,30),
        p(5,TEAM_A,"PLAYER",35,40),
    ]
    line = DefensiveLineEstimator().estimate(
        TEAM_A,players,ATTACK_PLUS_X
    )
    assert line.line_x is not None
    assert line.line_x < 20
    assert len(line.member_track_ids) >= 2
