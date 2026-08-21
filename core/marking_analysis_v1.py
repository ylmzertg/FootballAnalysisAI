from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"

TIGHT = "TIGHT"
MARKED = "MARKED"
LOOSE = "LOOSE"
UNMARKED = "UNMARKED"


@dataclass(frozen=True)
class MarkingPlayer:
    track_id: int
    team: str
    pitch_xy: tuple[float, float]
    role: str = "PLAYER"


@dataclass(frozen=True)
class MarkingAssessment:
    attacker_track_id: int
    nearest_defender_track_id: Optional[int]
    nearest_defender_distance_m: Optional[float]
    marking_state: str
    forward_progress_m: float
    goal_distance_m: Optional[float]
    distance_from_possessor_m: float
    dangerous: bool
    threat_score: float
    reason: str


@dataclass
class MarkingConfig:
    tight_distance_m: float = 2.0
    marked_distance_m: float = 4.0
    loose_distance_m: float = 6.0

    max_relevant_ball_distance_m: float = 34.0
    dangerous_goal_distance_m: float = 32.0
    useful_forward_progress_m: float = 22.0


def distance(a, b) -> float:
    return hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def forward_progress(
    start_x: float,
    end_x: float,
    direction: str,
) -> float:
    if direction == "PLUS_X":
        return float(end_x) - float(start_x)
    if direction == "MINUS_X":
        return float(start_x) - float(end_x)
    return 0.0


def goal_distance(
    x: float,
    direction: str,
) -> Optional[float]:
    if direction == "PLUS_X":
        return max(0.0, 105.0 - float(x))
    if direction == "MINUS_X":
        return max(0.0, float(x))
    return None


class MarkingAnalyzerV1:
    """
    Player-centric marking analysis.

    Important distinction:
    UNMARKED does not automatically mean "defensive error".
    A player is marked as a tactical threat only when his location is relevant
    to the current possession and attacking direction.
    """

    def __init__(
        self,
        config: MarkingConfig | None = None,
    ):
        self.config = config or MarkingConfig()

    def assess(
        self,
        *,
        possessor: MarkingPlayer,
        players: Iterable[MarkingPlayer],
        attack_direction: str,
    ) -> list[MarkingAssessment]:
        players = list(players)

        attackers = [
            p
            for p in players
            if (
                p.team == possessor.team
                and p.track_id != possessor.track_id
                and p.role not in {
                    "GOALKEEPER",
                    "REFEREE",
                    "OUTSIDE_PITCH",
                }
            )
        ]

        defenders = [
            p
            for p in players
            if (
                p.team != possessor.team
                and p.role not in {
                    "REFEREE",
                    "OUTSIDE_PITCH",
                }
            )
        ]

        if not defenders:
            return []

        result = []
        cfg = self.config

        for attacker in attackers:
            ball_distance = distance(
                possessor.pitch_xy,
                attacker.pitch_xy,
            )

            if (
                ball_distance
                > cfg.max_relevant_ball_distance_m
            ):
                continue

            nearest = min(
                defenders,
                key=lambda d: distance(
                    attacker.pitch_xy,
                    d.pitch_xy,
                ),
            )

            nearest_distance = distance(
                attacker.pitch_xy,
                nearest.pitch_xy,
            )

            if nearest_distance <= cfg.tight_distance_m:
                state = TIGHT
            elif nearest_distance <= cfg.marked_distance_m:
                state = MARKED
            elif nearest_distance <= cfg.loose_distance_m:
                state = LOOSE
            else:
                state = UNMARKED

            progress = forward_progress(
                possessor.pitch_xy[0],
                attacker.pitch_xy[0],
                attack_direction,
            )

            gd = goal_distance(
                attacker.pitch_xy[0],
                attack_direction,
            )

            dangerous = bool(
                (
                    gd is not None
                    and gd
                    <= cfg.dangerous_goal_distance_m
                )
                or progress >= 8.0
            )

            # Explainable threat score, not xT.
            marking_score = min(
                1.0,
                nearest_distance / 8.0,
            )
            progress_score = min(
                1.0,
                max(0.0, progress)
                / cfg.useful_forward_progress_m,
            )
            goal_score = (
                min(
                    1.0,
                    max(
                        0.0,
                        cfg.dangerous_goal_distance_m
                        - gd,
                    )
                    / cfg.dangerous_goal_distance_m,
                )
                if gd is not None
                else 0.0
            )

            support_score = max(
                0.0,
                1.0
                - ball_distance
                / cfg.max_relevant_ball_distance_m,
            )

            threat = (
                0.38 * marking_score
                + 0.27 * progress_score
                + 0.20 * goal_score
                + 0.15 * support_score
            )

            if state == TIGHT:
                reason = (
                    f"ID {attacker.track_id} yakın markajda "
                    f"({nearest_distance:.1f} m)."
                )
            elif state == MARKED:
                reason = (
                    f"ID {attacker.track_id} markaj altında "
                    f"({nearest_distance:.1f} m)."
                )
            elif state == LOOSE:
                reason = (
                    f"ID {attacker.track_id} gevşek markajda; "
                    f"en yakın savunmacı {nearest_distance:.1f} m."
                )
            else:
                reason = (
                    f"ID {attacker.track_id} markajsız; "
                    f"en yakın savunmacı {nearest_distance:.1f} m."
                )

            result.append(
                MarkingAssessment(
                    attacker_track_id=attacker.track_id,
                    nearest_defender_track_id=nearest.track_id,
                    nearest_defender_distance_m=round(
                        nearest_distance,
                        4,
                    ),
                    marking_state=state,
                    forward_progress_m=round(
                        progress,
                        4,
                    ),
                    goal_distance_m=(
                        round(gd, 4)
                        if gd is not None
                        else None
                    ),
                    distance_from_possessor_m=round(
                        ball_distance,
                        4,
                    ),
                    dangerous=dangerous,
                    threat_score=round(
                        float(threat),
                        5,
                    ),
                    reason=reason,
                )
            )

        return sorted(
            result,
            key=lambda a: (
                a.dangerous,
                a.threat_score,
            ),
            reverse=True,
        )
