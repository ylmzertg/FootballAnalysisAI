from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
UNKNOWN = "UNKNOWN"

LATE_PRESSURE = "LATE_PRESSURE"
UNMARKED_RUNNER = "UNMARKED_RUNNER"
FREE_PASSING_LANE = "FREE_PASSING_LANE"


@dataclass(frozen=True)
class PlayerState:
    track_id: int
    team: str
    pitch_xy: tuple[float, float]
    role: str = "PLAYER"


@dataclass(frozen=True)
class TacticalLane:
    receiver_track_id: int
    status: str
    distance_m: float
    blocker_track_ids: tuple[int, ...]


@dataclass(frozen=True)
class ErrorEvent:
    frame_index: int
    error_type: str
    attacking_team: str
    defending_team: str
    severity: str
    primary_track_id: Optional[int]
    secondary_track_id: Optional[int]
    metric_value: Optional[float]
    explanation: str
    evidence: dict


@dataclass
class ErrorDetectionConfig:
    late_pressure_distance_m: float = 4.5
    severe_pressure_distance_m: float = 6.0
    unmarked_distance_m: float = 4.5
    severe_unmarked_distance_m: float = 6.0
    min_runner_forward_progress_m: float = 3.0
    max_runner_distance_from_ball_m: float = 32.0
    free_lane_min_distance_m: float = 7.0
    free_lane_severe_distance_m: float = 14.0
    dangerous_goal_distance_m: float = 38.0


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def opponent(team: str) -> str:
    if team == TEAM_A:
        return TEAM_B
    if team == TEAM_B:
        return TEAM_A
    return UNKNOWN


def forward_progress(start_x: float, end_x: float, direction: str) -> float:
    if direction == "PLUS_X":
        return float(end_x) - float(start_x)
    if direction == "MINUS_X":
        return float(start_x) - float(end_x)
    return 0.0


def goal_distance(x: float, direction: str) -> Optional[float]:
    if direction == "PLUS_X":
        return max(0.0, 105.0 - float(x))
    if direction == "MINUS_X":
        return max(0.0, float(x))
    return None


class ErrorDetectorV1:
    def __init__(self, config: ErrorDetectionConfig | None = None):
        self.config = config or ErrorDetectionConfig()

    def detect(
        self,
        *,
        frame_index: int,
        attacking_team: str,
        possessor_track_id: Optional[int],
        players: Iterable[PlayerState],
        attack_direction: str,
        passing_lanes: Iterable[TacticalLane] = (),
    ) -> list[ErrorEvent]:
        players = list(players)

        if (
            attacking_team not in {TEAM_A, TEAM_B}
            or possessor_track_id is None
            or attack_direction not in {"PLUS_X", "MINUS_X"}
        ):
            return []

        possessor = next(
            (p for p in players if p.track_id == possessor_track_id),
            None,
        )
        if possessor is None:
            return []

        defending_team = opponent(attacking_team)

        attackers = [
            p for p in players
            if p.team == attacking_team
            and p.role not in {"REFEREE", "OUTSIDE_PITCH"}
        ]
        defenders = [
            p for p in players
            if p.team == defending_team
            and p.role not in {"REFEREE", "OUTSIDE_PITCH"}
        ]
        if not defenders:
            return []

        events: list[ErrorEvent] = []

        # 1) Late pressure
        nearest_defender = min(
            defenders,
            key=lambda p: distance(possessor.pitch_xy, p.pitch_xy),
        )
        nearest_distance = distance(
            possessor.pitch_xy,
            nearest_defender.pitch_xy,
        )
        possessor_goal_distance = goal_distance(
            possessor.pitch_xy[0],
            attack_direction,
        )
        dangerous = bool(
            possessor_goal_distance is not None
            and possessor_goal_distance <= self.config.dangerous_goal_distance_m
        )

        if nearest_distance >= self.config.late_pressure_distance_m:
            severity = (
                "HIGH"
                if (
                    nearest_distance >= self.config.severe_pressure_distance_m
                    or dangerous
                )
                else "MEDIUM"
            )
            events.append(
                ErrorEvent(
                    frame_index=frame_index,
                    error_type=LATE_PRESSURE,
                    attacking_team=attacking_team,
                    defending_team=defending_team,
                    severity=severity,
                    primary_track_id=nearest_defender.track_id,
                    secondary_track_id=possessor.track_id,
                    metric_value=nearest_distance,
                    explanation=(
                        f"Top sahibine en yakın savunmacı "
                        f"{nearest_distance:.1f} m uzakta; baskı gecikmiş görünüyor."
                    ),
                    evidence={
                        "possessor_track_id": possessor.track_id,
                        "nearest_defender_track_id": nearest_defender.track_id,
                        "nearest_defender_distance_m": round(nearest_distance, 4),
                        "possessor_goal_distance_m": (
                            round(possessor_goal_distance, 4)
                            if possessor_goal_distance is not None
                            else None
                        ),
                        "dangerous_zone": dangerous,
                    },
                )
            )

        # 2) Unmarked forward runner
        runner_candidates = []

        for attacker in attackers:
            if attacker.track_id == possessor.track_id:
                continue

            progress = forward_progress(
                possessor.pitch_xy[0],
                attacker.pitch_xy[0],
                attack_direction,
            )
            if progress < self.config.min_runner_forward_progress_m:
                continue

            ball_distance = distance(
                possessor.pitch_xy,
                attacker.pitch_xy,
            )
            if ball_distance > self.config.max_runner_distance_from_ball_m:
                continue

            nearest_marker = min(
                defenders,
                key=lambda p: distance(attacker.pitch_xy, p.pitch_xy),
            )
            marker_distance = distance(
                attacker.pitch_xy,
                nearest_marker.pitch_xy,
            )

            if marker_distance < self.config.unmarked_distance_m:
                continue

            runner_candidates.append(
                (
                    marker_distance,
                    progress,
                    ball_distance,
                    attacker,
                    nearest_marker,
                )
            )

        if runner_candidates:
            runner_candidates.sort(
                key=lambda x: (x[0], x[1]),
                reverse=True,
            )
            (
                marker_distance,
                progress,
                ball_distance,
                runner,
                nearest_marker,
            ) = runner_candidates[0]

            events.append(
                ErrorEvent(
                    frame_index=frame_index,
                    error_type=UNMARKED_RUNNER,
                    attacking_team=attacking_team,
                    defending_team=defending_team,
                    severity=(
                        "HIGH"
                        if marker_distance >= self.config.severe_unmarked_distance_m
                        else "MEDIUM"
                    ),
                    primary_track_id=runner.track_id,
                    secondary_track_id=nearest_marker.track_id,
                    metric_value=marker_distance,
                    explanation=(
                        f"İleri koşu yapan ID {runner.track_id}, en yakın "
                        f"savunmacıdan {marker_distance:.1f} m uzakta."
                    ),
                    evidence={
                        "runner_track_id": runner.track_id,
                        "nearest_defender_track_id": nearest_marker.track_id,
                        "nearest_defender_distance_m": round(marker_distance, 4),
                        "forward_progress_m": round(progress, 4),
                        "distance_from_possessor_m": round(ball_distance, 4),
                    },
                )
            )

        # 3) Free passing lane
        lane_by_receiver = {
            lane.receiver_track_id: lane
            for lane in passing_lanes
        }

        lane_candidates = []

        for receiver in attackers:
            if receiver.track_id == possessor.track_id:
                continue

            lane = lane_by_receiver.get(receiver.track_id)
            if (
                lane is None
                or lane.status != "OPEN"
                or lane.distance_m < self.config.free_lane_min_distance_m
            ):
                continue

            progress = forward_progress(
                possessor.pitch_xy[0],
                receiver.pitch_xy[0],
                attack_direction,
            )
            if progress <= 0:
                continue

            nearest_marker_distance = min(
                distance(receiver.pitch_xy, defender.pitch_xy)
                for defender in defenders
            )

            lane_candidates.append(
                (
                    progress,
                    lane.distance_m,
                    nearest_marker_distance,
                    receiver,
                    lane,
                )
            )

        if lane_candidates:
            lane_candidates.sort(
                key=lambda x: (x[0], x[2]),
                reverse=True,
            )
            (
                progress,
                lane_distance,
                marker_distance,
                receiver,
                lane,
            ) = lane_candidates[0]

            events.append(
                ErrorEvent(
                    frame_index=frame_index,
                    error_type=FREE_PASSING_LANE,
                    attacking_team=attacking_team,
                    defending_team=defending_team,
                    severity=(
                        "HIGH"
                        if (
                            lane_distance >= self.config.free_lane_severe_distance_m
                            or marker_distance >= self.config.severe_unmarked_distance_m
                        )
                        else "MEDIUM"
                    ),
                    primary_track_id=receiver.track_id,
                    secondary_track_id=possessor.track_id,
                    metric_value=lane_distance,
                    explanation=(
                        f"ID {receiver.track_id} için ileri yönde "
                        f"açık bir pas koridoru bulunuyor."
                    ),
                    evidence={
                        "receiver_track_id": receiver.track_id,
                        "possessor_track_id": possessor.track_id,
                        "pass_distance_m": round(lane_distance, 4),
                        "forward_progress_m": round(progress, 4),
                        "receiver_nearest_defender_m": round(marker_distance, 4),
                        "blocker_track_ids": list(lane.blocker_track_ids),
                    },
                )
            )

        return events
