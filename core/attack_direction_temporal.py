from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
PLUS_X = "PLUS_X"
MINUS_X = "MINUS_X"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DirectionEvidence:
    team: str
    direction: str
    confidence: float
    source: str
    frame_index: int


@dataclass(frozen=True)
class DirectionConsensus:
    team: str
    direction: str
    confidence: float
    source: str
    evidence_frames: int
    plus_x_frames: int
    minus_x_frames: int


@dataclass
class TemporalDirectionConfig:
    min_evidence_frames: int = 8
    min_consensus_ratio: float = 0.80
    allow_opponent_inference: bool = True


def opposite(direction: str) -> str:
    if direction == PLUS_X:
        return MINUS_X
    if direction == MINUS_X:
        return PLUS_X
    return UNKNOWN


class TemporalAttackDirectionResolver:
    def __init__(self, config: TemporalDirectionConfig | None = None):
        self.config = config or TemporalDirectionConfig()

    def _direct_consensus(
        self,
        team: str,
        evidence: Iterable[DirectionEvidence],
    ) -> DirectionConsensus:
        rows = [
            e for e in evidence
            if e.team == team and e.direction in {PLUS_X, MINUS_X}
        ]

        counts = Counter(e.direction for e in rows)
        plus = counts[PLUS_X]
        minus = counts[MINUS_X]
        total = plus + minus

        if total < self.config.min_evidence_frames:
            return DirectionConsensus(
                team=team,
                direction=UNKNOWN,
                confidence=0.0,
                source="insufficient_direct_evidence",
                evidence_frames=total,
                plus_x_frames=plus,
                minus_x_frames=minus,
            )

        best_direction = PLUS_X if plus >= minus else MINUS_X
        best_count = max(plus, minus)
        ratio = best_count / max(total, 1)

        if ratio < self.config.min_consensus_ratio:
            return DirectionConsensus(
                team=team,
                direction=UNKNOWN,
                confidence=ratio,
                source="ambiguous_direct_evidence",
                evidence_frames=total,
                plus_x_frames=plus,
                minus_x_frames=minus,
            )

        mean_conf = (
            sum(
                float(e.confidence)
                for e in rows
                if e.direction == best_direction
            )
            / best_count
        )

        confidence = min(
            0.99,
            0.65 * ratio + 0.35 * mean_conf,
        )

        return DirectionConsensus(
            team=team,
            direction=best_direction,
            confidence=confidence,
            source="temporal_goalkeeper_consensus",
            evidence_frames=total,
            plus_x_frames=plus,
            minus_x_frames=minus,
        )

    def resolve_pair(
        self,
        evidence: Iterable[DirectionEvidence],
    ) -> dict[str, DirectionConsensus]:
        evidence = list(evidence)

        a = self._direct_consensus(TEAM_A, evidence)
        b = self._direct_consensus(TEAM_B, evidence)

        if not self.config.allow_opponent_inference:
            return {TEAM_A: a, TEAM_B: b}

        if a.direction != UNKNOWN and b.direction == UNKNOWN:
            b = DirectionConsensus(
                team=TEAM_B,
                direction=opposite(a.direction),
                confidence=max(0.50, a.confidence * 0.85),
                source="opponent_direction_inference",
                evidence_frames=b.evidence_frames,
                plus_x_frames=b.plus_x_frames,
                minus_x_frames=b.minus_x_frames,
            )

        elif b.direction != UNKNOWN and a.direction == UNKNOWN:
            a = DirectionConsensus(
                team=TEAM_A,
                direction=opposite(b.direction),
                confidence=max(0.50, b.confidence * 0.85),
                source="opponent_direction_inference",
                evidence_frames=a.evidence_frames,
                plus_x_frames=a.plus_x_frames,
                minus_x_frames=a.minus_x_frames,
            )

        # If both resolve directly but contradict football geometry (same
        # attacking direction), refuse to invent certainty.
        elif (
            a.direction != UNKNOWN
            and b.direction != UNKNOWN
            and a.direction == b.direction
        ):
            a = DirectionConsensus(
                team=TEAM_A,
                direction=UNKNOWN,
                confidence=0.0,
                source="pair_conflict",
                evidence_frames=a.evidence_frames,
                plus_x_frames=a.plus_x_frames,
                minus_x_frames=a.minus_x_frames,
            )
            b = DirectionConsensus(
                team=TEAM_B,
                direction=UNKNOWN,
                confidence=0.0,
                source="pair_conflict",
                evidence_frames=b.evidence_frames,
                plus_x_frames=b.plus_x_frames,
                minus_x_frames=b.minus_x_frames,
            )

        return {TEAM_A: a, TEAM_B: b}
