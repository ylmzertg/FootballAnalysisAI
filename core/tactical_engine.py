
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional


TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"


@dataclass(frozen=True)
class TacticalPlayer:
    track_id: int
    team: str
    role: str
    pitch_xy: tuple[float, float]


@dataclass(frozen=True)
class PressureResult:
    level: str
    nearest_opponent_id: Optional[int]
    nearest_opponent_distance_m: Optional[float]
    opponents_within_3m: int
    opponents_within_5m: int


@dataclass(frozen=True)
class PassingLaneResult:
    receiver_track_id: int
    receiver_xy: tuple[float, float]
    distance_m: float
    status: str
    blocker_track_ids: tuple[int, ...]
    nearest_blocker_clearance_m: Optional[float]
    score: float


@dataclass
class TacticalConfig:
    pressure_high_distance_m: float = 2.0
    pressure_medium_distance_m: float = 3.5
    pressure_count_near_m: float = 3.0
    pressure_count_outer_m: float = 5.0

    lane_half_width_m: float = 1.25
    lane_endpoint_margin: float = 0.10
    min_pass_distance_m: float = 2.0
    max_pass_distance_m: float = 35.0
    max_open_lanes: int = 5


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def point_segment_metrics(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    """
    Return (perpendicular_distance_m, t) where t is the projection parameter
    along start -> end. t=0 at start, t=1 at end.
    """
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    px, py = float(point[0]), float(point[1])

    vx, vy = ex - sx, ey - sy
    denom = vx * vx + vy * vy

    if denom <= 1e-12:
        return distance(point, start), 0.0

    t = ((px - sx) * vx + (py - sy) * vy) / denom
    cx = sx + t * vx
    cy = sy + t * vy
    return hypot(px - cx, py - cy), t


class TacticalEngine:
    def __init__(self, config: TacticalConfig | None = None):
        self.config = config or TacticalConfig()

    def pressure(
        self,
        possessor: TacticalPlayer,
        opponents: Iterable[TacticalPlayer],
    ) -> PressureResult:
        ranked = sorted(
            (
                (distance(possessor.pitch_xy, op.pitch_xy), op)
                for op in opponents
                if op.team != possessor.team
            ),
            key=lambda x: x[0],
        )

        if not ranked:
            return PressureResult(
                level="NONE",
                nearest_opponent_id=None,
                nearest_opponent_distance_m=None,
                opponents_within_3m=0,
                opponents_within_5m=0,
            )

        nearest_d, nearest = ranked[0]

        within_3 = sum(
            1 for d, _ in ranked
            if d <= self.config.pressure_count_near_m
        )
        within_5 = sum(
            1 for d, _ in ranked
            if d <= self.config.pressure_count_outer_m
        )

        if (
            nearest_d <= self.config.pressure_high_distance_m
            or within_3 >= 2
        ):
            level = "HIGH"
        elif (
            nearest_d <= self.config.pressure_medium_distance_m
            or within_5 >= 2
        ):
            level = "MEDIUM"
        else:
            level = "LOW"

        return PressureResult(
            level=level,
            nearest_opponent_id=nearest.track_id,
            nearest_opponent_distance_m=nearest_d,
            opponents_within_3m=within_3,
            opponents_within_5m=within_5,
        )

    def passing_lanes(
        self,
        possessor: TacticalPlayer,
        teammates: Iterable[TacticalPlayer],
        opponents: Iterable[TacticalPlayer],
    ) -> list[PassingLaneResult]:
        teammates = [
            p for p in teammates
            if p.team == possessor.team and p.track_id != possessor.track_id
        ]
        opponents = [
            p for p in opponents
            if p.team != possessor.team
        ]

        lanes: list[PassingLaneResult] = []

        for receiver in teammates:
            pass_distance = distance(
                possessor.pitch_xy,
                receiver.pitch_xy,
            )
            if pass_distance < self.config.min_pass_distance_m:
                continue
            if pass_distance > self.config.max_pass_distance_m:
                continue

            blockers: list[tuple[float, TacticalPlayer]] = []

            for op in opponents:
                clearance, t = point_segment_metrics(
                    op.pitch_xy,
                    possessor.pitch_xy,
                    receiver.pitch_xy,
                )

                if not (
                    self.config.lane_endpoint_margin
                    <= t
                    <= 1.0 - self.config.lane_endpoint_margin
                ):
                    continue

                if clearance <= self.config.lane_half_width_m:
                    blockers.append((clearance, op))

            blockers.sort(key=lambda x: x[0])

            if blockers:
                status = "BLOCKED"
                blocker_ids = tuple(p.track_id for _, p in blockers)
                nearest_clearance = blockers[0][0]
            else:
                status = "OPEN"
                blocker_ids = ()
                nearest_clearance = self._nearest_corridor_clearance(
                    possessor,
                    receiver,
                    opponents,
                )

            score = self._lane_score(
                pass_distance=pass_distance,
                status=status,
                nearest_clearance=nearest_clearance,
            )

            lanes.append(
                PassingLaneResult(
                    receiver_track_id=receiver.track_id,
                    receiver_xy=receiver.pitch_xy,
                    distance_m=pass_distance,
                    status=status,
                    blocker_track_ids=blocker_ids,
                    nearest_blocker_clearance_m=nearest_clearance,
                    score=score,
                )
            )

        # Open lanes first, then higher score.
        lanes.sort(
            key=lambda x: (
                0 if x.status == "OPEN" else 1,
                -x.score,
                x.distance_m,
            )
        )
        return lanes

    def best_open_lanes(
        self,
        lanes: Iterable[PassingLaneResult],
    ) -> list[PassingLaneResult]:
        return [
            lane
            for lane in lanes
            if lane.status == "OPEN"
        ][:max(1, self.config.max_open_lanes)]

    def _nearest_corridor_clearance(
        self,
        possessor: TacticalPlayer,
        receiver: TacticalPlayer,
        opponents: list[TacticalPlayer],
    ) -> Optional[float]:
        clearances = []

        for op in opponents:
            clearance, t = point_segment_metrics(
                op.pitch_xy,
                possessor.pitch_xy,
                receiver.pitch_xy,
            )

            if (
                self.config.lane_endpoint_margin
                <= t
                <= 1.0 - self.config.lane_endpoint_margin
            ):
                clearances.append(clearance)

        return min(clearances) if clearances else None

    def _lane_score(
        self,
        pass_distance: float,
        status: str,
        nearest_clearance: Optional[float],
    ) -> float:
        # Neutral geometric quality score. Attacking direction is intentionally
        # not assumed in v1.
        distance_score = max(
            0.0,
            1.0 - pass_distance / max(self.config.max_pass_distance_m, 1e-6),
        )

        if nearest_clearance is None:
            clearance_score = 1.0
        else:
            clearance_score = min(
                1.0,
                nearest_clearance / max(3.5, self.config.lane_half_width_m),
            )

        score = 0.45 * distance_score + 0.55 * clearance_score

        if status == "BLOCKED":
            score *= 0.20

        return max(0.0, min(1.0, score))
