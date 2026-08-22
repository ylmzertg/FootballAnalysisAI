from core.decision_comparison_v1 import (
    ACTUAL_PASS,
    ACTUAL_SHOT,
    ACTUAL_TURNOVER,
    CHOSE_ALTERNATIVE,
    MATCHED_BEST,
    SHOT_OVER_PASS,
    TURNOVER_OVER_PASS,
    compare_decision,
)


def test_actual_pass_matches_best():
    result = compare_decision(
        incident_id="INC-1",
        decision_frame=90,
        possessor_track_id=4,
        best_receiver_id=7,
        best_score=0.8,
        best_category="BEST",
        actual_action=ACTUAL_PASS,
        actual_receiver_id=7,
        actual_frame=94,
    )

    assert result.comparison == MATCHED_BEST


def test_actual_pass_can_choose_alternative():
    result = compare_decision(
        incident_id="INC-1",
        decision_frame=90,
        possessor_track_id=4,
        best_receiver_id=7,
        best_score=0.8,
        best_category="BEST",
        actual_action=ACTUAL_PASS,
        actual_receiver_id=9,
        actual_frame=94,
    )

    assert result.comparison == CHOSE_ALTERNATIVE


def test_shot_can_be_compared_against_pass_option():
    result = compare_decision(
        incident_id="INC-1",
        decision_frame=90,
        possessor_track_id=4,
        best_receiver_id=7,
        best_score=0.8,
        best_category="BEST",
        actual_action=ACTUAL_SHOT,
        actual_receiver_id=None,
        actual_frame=95,
    )

    assert result.comparison == SHOT_OVER_PASS

def test_turnover_is_not_treated_as_pass():
    result = compare_decision(
        incident_id="INC-TURNOVER",
        decision_frame=90,
        possessor_track_id=4,
        best_receiver_id=7,
        best_score=0.8,
        best_category="BEST",
        actual_action=ACTUAL_TURNOVER,
        actual_receiver_id=12,
        actual_frame=94,
    )

    assert result.comparison == TURNOVER_OVER_PASS
    assert result.actual_action == ACTUAL_TURNOVER


def test_find_actual_action_marks_opponent_control_as_turnover():
    from scripts.compare_decisions_v1 import find_actual_action

    possession_rows = {
        90: {
            "phase": "PASS_FLIGHT",
            "source_owner_track_id": 4,
            "target_owner_track_id": 12,
            "source_team": "TEAM_A",
            "target_team": "TEAM_B",
        }
    }

    action, receiver, frame = find_actual_action(
        start_frame=90,
        possessor_track_id=4,
        possession_rows=possession_rows,
        shot_rows={},
        lookahead=10,
    )

    assert action == ACTUAL_TURNOVER
    assert receiver == 12
    assert frame == 90

