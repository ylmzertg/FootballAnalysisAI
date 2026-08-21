from core.decision_comparison_v1 import (
    ACTUAL_PASS,
    ACTUAL_SHOT,
    CHOSE_ALTERNATIVE,
    MATCHED_BEST,
    SHOT_OVER_PASS,
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
