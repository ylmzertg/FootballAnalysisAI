from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


ACTUAL_PASS = "PASS"
ACTUAL_SHOT = "SHOT"
ACTUAL_CARRY = "CARRY_OR_CONTINUE"
ACTUAL_UNKNOWN = "UNKNOWN"

MATCHED_BEST = "MATCHED_BEST"
CHOSE_ALTERNATIVE = "CHOSE_ALTERNATIVE"
SHOT_OVER_PASS = "SHOT_OVER_PASS"
NO_CLEAR_COMPARISON = "NO_CLEAR_COMPARISON"


@dataclass(frozen=True)
class DecisionComparison:
    incident_id: str
    decision_frame: int
    possessor_track_id: Optional[int]

    best_receiver_id: Optional[int]
    best_score: Optional[float]
    best_category: Optional[str]

    actual_action: str
    actual_receiver_id: Optional[int]
    actual_frame: Optional[int]

    comparison: str
    explanation: str


def compare_decision(
    *,
    incident_id: str,
    decision_frame: int,
    possessor_track_id: Optional[int],
    best_receiver_id: Optional[int],
    best_score: Optional[float],
    best_category: Optional[str],
    actual_action: str,
    actual_receiver_id: Optional[int],
    actual_frame: Optional[int],
) -> DecisionComparison:
    if best_receiver_id is None:
        comparison = NO_CLEAR_COMPARISON
        explanation = (
            "Karar anında güvenilir bir BEST/GOOD pas alternatifi bulunamadı."
        )

    elif actual_action == ACTUAL_SHOT:
        comparison = SHOT_OVER_PASS
        explanation = (
            f"Sistem ID {best_receiver_id} yönünü güçlü pas alternatifi olarak "
            f"görürken top sahibi şutu tercih etti."
        )

    elif (
        actual_action == ACTUAL_PASS
        and actual_receiver_id == best_receiver_id
    ):
        comparison = MATCHED_BEST
        explanation = (
            f"Gerçek pas ID {actual_receiver_id} yönüne oynandı ve sistemin "
            f"en yüksek sıralı seçeneğiyle eşleşti."
        )

    elif (
        actual_action == ACTUAL_PASS
        and actual_receiver_id is not None
    ):
        comparison = CHOSE_ALTERNATIVE
        explanation = (
            f"Sistem ID {best_receiver_id} yönünü daha yüksek sıralarken "
            f"gerçekte ID {actual_receiver_id} tercih edildi."
        )

    else:
        comparison = NO_CLEAR_COMPARISON
        explanation = (
            "Gerçek aksiyon güvenilir biçimde pas veya şut olarak "
            "karşılaştırılamadı."
        )

    return DecisionComparison(
        incident_id=incident_id,
        decision_frame=decision_frame,
        possessor_track_id=possessor_track_id,
        best_receiver_id=best_receiver_id,
        best_score=best_score,
        best_category=best_category,
        actual_action=actual_action,
        actual_receiver_id=actual_receiver_id,
        actual_frame=actual_frame,
        comparison=comparison,
        explanation=explanation,
    )
