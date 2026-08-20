from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ResolverPlayer:
    track_id: int
    team: str
    role: str
    pitch_xy: tuple[float, float]


@dataclass(frozen=True)
class GoalkeeperTeamEvidence:
    frame_index: int
    goalkeeper_track_id: int
    goalkeeper_x: float
    assigned_team: str
    confidence: float
    neighbor_count: int
    team_a_weight: float
    team_b_weight: float


@dataclass(frozen=True)
class GoalkeeperTeamConsensus:
    goalkeeper_track_id: int
    resolved_team: str
    confidence: float
    evidence_frames: int
    team_a_frames: int
    team_b_frames: int
    median_goalkeeper_x: float


@dataclass
class GoalkeeperTeamResolverConfig:
    neighbor_radius_m: float = 28.0
    min_neighbors: int = 2
    min_weight_margin_ratio: float = 0.18
    min_evidence_frames: int = 4
    min_consensus_ratio: float = 0.65


def distance(a, b):
    return hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


class GoalkeeperTeamResolver:
    """
    Resolves goalkeeper team independently of the goalkeeper's own noisy
    TEAM_A/TEAM_B label.

    Evidence comes from nearby non-referee, non-goalkeeper outfield players in
    calibrated pitch coordinates. This avoids circularly trusting a wrongly
    assigned goalkeeper team label when attack direction is inferred.
    """

    def __init__(self, config: GoalkeeperTeamResolverConfig | None = None):
        self.config = config or GoalkeeperTeamResolverConfig()

    def frame_evidence(
        self,
        frame_index: int,
        goalkeeper: ResolverPlayer,
        players: Iterable[ResolverPlayer],
    ) -> Optional[GoalkeeperTeamEvidence]:
        neighbors = []

        for p in players:
            if p.track_id == goalkeeper.track_id:
                continue
            if str(p.role).upper() in {"REFEREE", "GOALKEEPER"}:
                continue
            if p.team not in {TEAM_A, TEAM_B}:
                continue

            d = distance(goalkeeper.pitch_xy, p.pitch_xy)
            if d <= self.config.neighbor_radius_m:
                neighbors.append((d, p))

        if len(neighbors) < self.config.min_neighbors:
            return None

        weights = {TEAM_A: 0.0, TEAM_B: 0.0}

        for d, p in neighbors:
            # Nearby defenders carry more evidence than distant midfielders.
            w = 1.0 / max(1.5, d)
            weights[p.team] += w

        total = weights[TEAM_A] + weights[TEAM_B]
        if total <= 0:
            return None

        best_team = TEAM_A if weights[TEAM_A] >= weights[TEAM_B] else TEAM_B
        other_team = TEAM_B if best_team == TEAM_A else TEAM_A

        margin_ratio = (
            weights[best_team] - weights[other_team]
        ) / total

        if margin_ratio < self.config.min_weight_margin_ratio:
            return None

        confidence = min(
            0.98,
            0.55 + 0.45 * margin_ratio,
        )

        return GoalkeeperTeamEvidence(
            frame_index=frame_index,
            goalkeeper_track_id=goalkeeper.track_id,
            goalkeeper_x=float(goalkeeper.pitch_xy[0]),
            assigned_team=best_team,
            confidence=confidence,
            neighbor_count=len(neighbors),
            team_a_weight=weights[TEAM_A],
            team_b_weight=weights[TEAM_B],
        )

    def consensus(
        self,
        evidence: Iterable[GoalkeeperTeamEvidence],
    ) -> dict[int, GoalkeeperTeamConsensus]:
        by_gk = defaultdict(list)

        for e in evidence:
            by_gk[e.goalkeeper_track_id].append(e)

        result = {}

        for gk_id, rows in by_gk.items():
            counts = Counter(
                e.assigned_team
                for e in rows
            )

            a = counts[TEAM_A]
            b = counts[TEAM_B]
            total = a + b

            if total < self.config.min_evidence_frames:
                resolved = UNKNOWN
                confidence = 0.0
            else:
                resolved = TEAM_A if a >= b else TEAM_B
                best = max(a, b)
                ratio = best / total

                if ratio < self.config.min_consensus_ratio:
                    resolved = UNKNOWN
                    confidence = ratio
                else:
                    mean_conf = sum(
                        e.confidence
                        for e in rows
                        if e.assigned_team == resolved
                    ) / best
                    confidence = min(
                        0.99,
                        0.60 * ratio + 0.40 * mean_conf,
                    )

            xs = sorted(
                float(e.goalkeeper_x)
                for e in rows
            )
            median_x = xs[len(xs)//2]

            result[gk_id] = GoalkeeperTeamConsensus(
                goalkeeper_track_id=gk_id,
                resolved_team=resolved,
                confidence=confidence,
                evidence_frames=total,
                team_a_frames=a,
                team_b_frames=b,
                median_goalkeeper_x=median_x,
            )

        return result
