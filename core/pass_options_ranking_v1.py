from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
UNKNOWN = "UNKNOWN"

BEST = "BEST"
GOOD = "GOOD"
RISKY = "RISKY"
BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class PassPlayer:
    track_id: int
    team: str
    pitch_xy: tuple[float, float]
    role: str = "PLAYER"


@dataclass(frozen=True)
class PassLane:
    receiver_track_id: int
    status: str
    distance_m: float
    nearest_blocker_clearance_m: Optional[float]
    blocker_track_ids: tuple[int, ...]
    tactical_score: float


@dataclass(frozen=True)
class PassOption:
    receiver_track_id: int
    category: str
    score: float
    rank: int
    pass_distance_m: float
    forward_progress_m: float
    goal_progress_m: float
    receiver_space_m: Optional[float]
    lane_clearance_m: Optional[float]
    explanation: str
    reasons: tuple[str, ...]


@dataclass
class PassRankingConfig:
    max_options: int = 6

    # Component normalization
    useful_forward_progress_m: float = 22.0
    useful_goal_progress_m: float = 22.0
    useful_receiver_space_m: float = 7.0
    useful_lane_clearance_m: float = 5.0

    ideal_pass_distance_m: float = 16.0
    pass_distance_tolerance_m: float = 20.0

    # Category thresholds
    best_min_score: float = 0.66
    good_min_score: float = 0.50
    risky_min_score: float = 0.25

    # Hard risk signals
    tight_receiver_space_m: float = 2.4
    narrow_lane_clearance_m: float = 2.0
    very_long_pass_m: float = 30.0
    backward_penalty_start_m: float = -2.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
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


class PassOptionsRankerV1:
    """
    Explainable pass-option ranking.

    Inputs are deliberately simple and auditable:
    - Tactical Engine lane geometry;
    - attacking direction;
    - receiver separation from defenders;
    - progress toward opponent goal;
    - pass distance.

    This is NOT an expected-threat model yet. It is the first transparent
    ranking layer for analyst/telestration output.
    """

    def __init__(
        self,
        config: PassRankingConfig | None = None,
    ):
        self.config = config or PassRankingConfig()

    def _receiver_space(
        self,
        receiver: PassPlayer,
        defenders: list[PassPlayer],
    ) -> Optional[float]:
        if not defenders:
            return None

        return min(
            distance(
                receiver.pitch_xy,
                defender.pitch_xy,
            )
            for defender in defenders
        )

    def _distance_quality(
        self,
        pass_distance: float,
    ) -> float:
        cfg = self.config

        delta = abs(
            pass_distance
            - cfg.ideal_pass_distance_m
        )

        return clamp01(
            1.0
            - delta
            / max(
                cfg.pass_distance_tolerance_m,
                1e-6,
            )
        )

    def _base_score(
        self,
        *,
        lane: PassLane,
        forward_m: float,
        goal_progress_m: float,
        receiver_space_m: Optional[float],
    ) -> float:
        cfg = self.config

        tactical = clamp01(
            lane.tactical_score
        )

        forward_score = clamp01(
            max(0.0, forward_m)
            / cfg.useful_forward_progress_m
        )

        goal_score = clamp01(
            max(0.0, goal_progress_m)
            / cfg.useful_goal_progress_m
        )

        if receiver_space_m is None:
            space_score = 0.55
        else:
            space_score = clamp01(
                receiver_space_m
                / cfg.useful_receiver_space_m
            )

        if lane.nearest_blocker_clearance_m is None:
            clearance_score = 1.0
        else:
            clearance_score = clamp01(
                lane.nearest_blocker_clearance_m
                / cfg.useful_lane_clearance_m
            )

        distance_score = self._distance_quality(
            lane.distance_m
        )

        # Transparent weighted score.
        score = (
            0.28 * tactical
            + 0.22 * goal_score
            + 0.16 * forward_score
            + 0.16 * space_score
            + 0.10 * clearance_score
            + 0.08 * distance_score
        )

        # Backward passes can still be useful, but should not rank above an
        # equally safe progressive option in this analyst layer.
        if (
            forward_m
            < cfg.backward_penalty_start_m
        ):
            score *= 0.72

        if (
            receiver_space_m is not None
            and receiver_space_m
            < cfg.tight_receiver_space_m
        ):
            score *= 0.72

        if (
            lane.nearest_blocker_clearance_m
            is not None
            and lane.nearest_blocker_clearance_m
            < cfg.narrow_lane_clearance_m
        ):
            score *= 0.78

        if (
            lane.distance_m
            > cfg.very_long_pass_m
        ):
            score *= 0.82

        return clamp01(
            score
        )

    def rank(
        self,
        *,
        possessor: PassPlayer,
        players: Iterable[PassPlayer],
        lanes: Iterable[PassLane],
        attack_direction: str,
    ) -> list[PassOption]:
        players = list(players)
        lanes = list(lanes)

        if (
            possessor.team
            not in {
                TEAM_A,
                TEAM_B,
            }
            or attack_direction
            not in {
                "PLUS_X",
                "MINUS_X",
            }
        ):
            return []

        teammates = {
            p.track_id: p
            for p in players
            if (
                p.team == possessor.team
                and p.track_id
                != possessor.track_id
                and p.role
                not in {
                    "REFEREE",
                    "GOALKEEPER",
                    "OUTSIDE_PITCH",
                }
            )
        }

        defenders = [
            p
            for p in players
            if (
                p.team
                != possessor.team
                and p.team
                in {
                    TEAM_A,
                    TEAM_B,
                }
                and p.role
                not in {
                    "REFEREE",
                    "OUTSIDE_PITCH",
                }
            )
        ]

        start_goal_distance = goal_distance(
            possessor.pitch_xy[0],
            attack_direction,
        )

        raw_options = []

        for lane in lanes:
            receiver = teammates.get(
                lane.receiver_track_id
            )

            if receiver is None:
                continue

            forward_m = forward_progress(
                possessor.pitch_xy[0],
                receiver.pitch_xy[0],
                attack_direction,
            )

            receiver_goal_distance = goal_distance(
                receiver.pitch_xy[0],
                attack_direction,
            )

            if (
                start_goal_distance
                is None
                or receiver_goal_distance
                is None
            ):
                goal_progress_m = 0.0
            else:
                goal_progress_m = (
                    start_goal_distance
                    - receiver_goal_distance
                )

            receiver_space_m = (
                self._receiver_space(
                    receiver,
                    defenders,
                )
            )

            reasons = []

            if lane.status != "OPEN":
                category = BLOCKED
                score = clamp01(
                    lane.tactical_score
                    * 0.18
                )

                reasons.append(
                    "pas koridorunda rakip engeli var"
                )

                if lane.blocker_track_ids:
                    reasons.append(
                        "bloklayan oyuncu: "
                        + ", ".join(
                            str(x)
                            for x
                            in lane.blocker_track_ids[:3]
                        )
                    )

            else:
                score = self._base_score(
                    lane=lane,
                    forward_m=forward_m,
                    goal_progress_m=goal_progress_m,
                    receiver_space_m=receiver_space_m,
                )

                category = RISKY  # final category assigned after ranking

                if goal_progress_m >= 8.0:
                    reasons.append(
                        f"kaleye {goal_progress_m:.1f} m ilerletiyor"
                    )

                elif forward_m > 0:
                    reasons.append(
                        f"ileri yönlü {forward_m:.1f} m progresyon"
                    )

                else:
                    reasons.append(
                        "geri / yatay güvenlik pası"
                    )

                if (
                    receiver_space_m
                    is not None
                ):
                    if (
                        receiver_space_m
                        >= 5.0
                    ):
                        reasons.append(
                            f"alıcı çevresinde {receiver_space_m:.1f} m boşluk"
                        )

                    elif (
                        receiver_space_m
                        < self.config.tight_receiver_space_m
                    ):
                        reasons.append(
                            f"alıcı yakın baskıda ({receiver_space_m:.1f} m)"
                        )

                if (
                    lane.nearest_blocker_clearance_m
                    is not None
                ):
                    if (
                        lane.nearest_blocker_clearance_m
                        >= 3.5
                    ):
                        reasons.append(
                            "pas koridoru geniş"
                        )

                    elif (
                        lane.nearest_blocker_clearance_m
                        < self.config.narrow_lane_clearance_m
                    ):
                        reasons.append(
                            "koridor dar"
                        )

                if (
                    lane.distance_m
                    > self.config.very_long_pass_m
                ):
                    reasons.append(
                        "uzun pas riski"
                    )

            raw_options.append(
                {
                    "receiver": receiver,
                    "lane": lane,
                    "score": score,
                    "category": category,
                    "forward_m": forward_m,
                    "goal_progress_m": goal_progress_m,
                    "receiver_space_m": receiver_space_m,
                    "reasons": reasons,
                }
            )

        # OPEN options before blocked, then score.
        raw_options.sort(
            key=lambda item: (
                0
                if item["lane"].status
                == "OPEN"
                else 1,
                -float(item["score"]),
            )
        )

        # Exactly one BEST at most.
        best_given = False

        result: list[PassOption] = []

        for index, item in enumerate(
            raw_options[: self.config.max_options],
            start=1,
        ):
            lane = item["lane"]
            score = float(
                item["score"]
            )

            if (
                lane.status
                != "OPEN"
            ):
                category = BLOCKED

            elif (
                not best_given
                and score
                >= self.config.best_min_score
            ):
                category = BEST
                best_given = True

            elif (
                score
                >= self.config.good_min_score
            ):
                category = GOOD

            else:
                category = RISKY

            reasons = tuple(
                item["reasons"]
            )

            explanation = (
                f"ID {lane.receiver_track_id}: "
                f"{category} pas seçeneği; "
                + (
                    "; ".join(reasons)
                    if reasons
                    else "geometrik değerlendirme"
                )
                + "."
            )

            result.append(
                PassOption(
                    receiver_track_id=lane.receiver_track_id,
                    category=category,
                    score=round(score, 5),
                    rank=index,
                    pass_distance_m=round(
                        float(lane.distance_m),
                        4,
                    ),
                    forward_progress_m=round(
                        float(item["forward_m"]),
                        4,
                    ),
                    goal_progress_m=round(
                        float(item["goal_progress_m"]),
                        4,
                    ),
                    receiver_space_m=(
                        round(
                            float(item["receiver_space_m"]),
                            4,
                        )
                        if item["receiver_space_m"]
                        is not None
                        else None
                    ),
                    lane_clearance_m=(
                        round(
                            float(
                                lane.nearest_blocker_clearance_m
                            ),
                            4,
                        )
                        if lane.nearest_blocker_clearance_m
                        is not None
                        else None
                    ),
                    explanation=explanation,
                    reasons=reasons,
                )
            )

        return result
