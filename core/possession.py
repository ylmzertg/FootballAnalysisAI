
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional


TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
LOOSE = "LOOSE"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PlayerPossessionCandidate:
    track_id: int
    team: str
    role: str
    pitch_xy: tuple[float, float]


@dataclass
class PossessionConfig:
    acquire_distance_m: float = 2.4
    release_distance_m: float = 4.0
    switch_margin_m: float = 0.75
    confirm_frames: int = 2
    hold_missing_ball_frames: int = 4
    allow_goalkeeper: bool = True


@dataclass
class PossessionResult:
    state: str
    possessor_track_id: Optional[int]
    possessor_team: Optional[str]
    distance_m: Optional[float]
    confidence: float
    nearest_track_id: Optional[int]
    nearest_team: Optional[str]
    nearest_distance_m: Optional[float]
    reason: str


class PossessionEstimator:
    """
    Temporal possession estimator.

    State changes are conservative:
    - acquisition requires a nearby player for N consecutive frames;
    - switching team/player requires confirmation and a meaningful distance advantage;
    - short ball-detection gaps retain the current owner briefly;
    - ball present but not controllably close to a player becomes LOOSE.
    """

    def __init__(self, config: PossessionConfig | None = None):
        self.config = config or PossessionConfig()

        self.current_track_id: Optional[int] = None
        self.current_team: Optional[str] = None
        self.current_distance_m: Optional[float] = None

        self.pending_track_id: Optional[int] = None
        self.pending_team: Optional[str] = None
        self.pending_count: int = 0

        self.missing_ball_frames: int = 0

    def _eligible(
        self,
        players: Iterable[PlayerPossessionCandidate],
    ) -> list[PlayerPossessionCandidate]:
        out = []
        for p in players:
            if p.team not in {TEAM_A, TEAM_B}:
                continue
            role = str(p.role).upper()
            if role in {"REFEREE", "OUTSIDE_PITCH"}:
                continue
            if role == "GOALKEEPER" and not self.config.allow_goalkeeper:
                continue
            out.append(p)
        return out

    @staticmethod
    def _distance(
        ball_xy: tuple[float, float],
        player_xy: tuple[float, float],
    ) -> float:
        return hypot(
            float(ball_xy[0]) - float(player_xy[0]),
            float(ball_xy[1]) - float(player_xy[1]),
        )

    def _nearest(
        self,
        ball_xy: tuple[float, float],
        players: list[PlayerPossessionCandidate],
    ):
        if not players:
            return None, None
        ranked = sorted(
            (
                (self._distance(ball_xy, p.pitch_xy), p)
                for p in players
            ),
            key=lambda x: x[0],
        )
        return ranked[0]

    def _find_current_distance(
        self,
        ball_xy: tuple[float, float],
        players: list[PlayerPossessionCandidate],
    ) -> Optional[float]:
        if self.current_track_id is None:
            return None
        for p in players:
            if p.track_id == self.current_track_id:
                return self._distance(ball_xy, p.pitch_xy)
        return None

    def _clear_pending(self):
        self.pending_track_id = None
        self.pending_team = None
        self.pending_count = 0

    def _set_pending(self, track_id: int, team: str):
        if self.pending_track_id == track_id and self.pending_team == team:
            self.pending_count += 1
        else:
            self.pending_track_id = track_id
            self.pending_team = team
            self.pending_count = 1

    def _commit(self, track_id: int, team: str, distance_m: float):
        self.current_track_id = track_id
        self.current_team = team
        self.current_distance_m = distance_m
        self._clear_pending()

    def update(
        self,
        ball_xy: Optional[tuple[float, float]],
        players: Iterable[PlayerPossessionCandidate],
        *,
        ball_detected: bool,
        ball_predicted: bool,
    ) -> PossessionResult:
        players = self._eligible(players)

        if ball_xy is None:
            self.missing_ball_frames += 1
            self._clear_pending()

            if (
                self.current_track_id is not None
                and self.missing_ball_frames <= self.config.hold_missing_ball_frames
            ):
                return PossessionResult(
                    state=self.current_team or UNKNOWN,
                    possessor_track_id=self.current_track_id,
                    possessor_team=self.current_team,
                    distance_m=self.current_distance_m,
                    confidence=max(
                        0.15,
                        0.65 - 0.10 * self.missing_ball_frames,
                    ),
                    nearest_track_id=None,
                    nearest_team=None,
                    nearest_distance_m=None,
                    reason="short_ball_gap_hold",
                )

            self.current_track_id = None
            self.current_team = None
            self.current_distance_m = None

            return PossessionResult(
                state=UNKNOWN,
                possessor_track_id=None,
                possessor_team=None,
                distance_m=None,
                confidence=0.0,
                nearest_track_id=None,
                nearest_team=None,
                nearest_distance_m=None,
                reason="ball_missing",
            )

        self.missing_ball_frames = 0

        nearest_distance, nearest = self._nearest(ball_xy, players)
        current_distance = self._find_current_distance(ball_xy, players)

        nearest_track_id = nearest.track_id if nearest is not None else None
        nearest_team = nearest.team if nearest is not None else None

        # Maintain current possession while the owner remains within release range.
        if self.current_track_id is not None and current_distance is not None:
            if current_distance <= self.config.release_distance_m:
                # A competitor may take over only if clearly closer.
                if (
                    nearest is not None
                    and nearest.track_id != self.current_track_id
                    and nearest_distance is not None
                    and nearest_distance <= self.config.acquire_distance_m
                    and nearest_distance + self.config.switch_margin_m
                    < current_distance
                ):
                    self._set_pending(nearest.track_id, nearest.team)
                    if self.pending_count >= self.config.confirm_frames:
                        self._commit(
                            nearest.track_id,
                            nearest.team,
                            nearest_distance,
                        )
                        return PossessionResult(
                            state=nearest.team,
                            possessor_track_id=nearest.track_id,
                            possessor_team=nearest.team,
                            distance_m=nearest_distance,
                            confidence=self._confidence(
                                nearest_distance,
                                ball_detected,
                                ball_predicted,
                            ),
                            nearest_track_id=nearest.track_id,
                            nearest_team=nearest.team,
                            nearest_distance_m=nearest_distance,
                            reason="confirmed_switch",
                        )
                else:
                    self._clear_pending()

                self.current_distance_m = current_distance
                return PossessionResult(
                    state=self.current_team or UNKNOWN,
                    possessor_track_id=self.current_track_id,
                    possessor_team=self.current_team,
                    distance_m=current_distance,
                    confidence=self._confidence(
                        current_distance,
                        ball_detected,
                        ball_predicted,
                    ),
                    nearest_track_id=nearest_track_id,
                    nearest_team=nearest_team,
                    nearest_distance_m=nearest_distance,
                    reason="owner_hysteresis",
                )

        # Current owner is gone or the ball moved beyond release range.
        if nearest is not None and nearest_distance is not None:
            if nearest_distance <= self.config.acquire_distance_m:
                self._set_pending(nearest.track_id, nearest.team)

                if self.pending_count >= self.config.confirm_frames:
                    self._commit(
                        nearest.track_id,
                        nearest.team,
                        nearest_distance,
                    )
                    return PossessionResult(
                        state=nearest.team,
                        possessor_track_id=nearest.track_id,
                        possessor_team=nearest.team,
                        distance_m=nearest_distance,
                        confidence=self._confidence(
                            nearest_distance,
                            ball_detected,
                            ball_predicted,
                        ),
                        nearest_track_id=nearest.track_id,
                        nearest_team=nearest.team,
                        nearest_distance_m=nearest_distance,
                        reason="confirmed_acquire",
                    )

                return PossessionResult(
                    state=LOOSE,
                    possessor_track_id=None,
                    possessor_team=None,
                    distance_m=None,
                    confidence=0.35,
                    nearest_track_id=nearest.track_id,
                    nearest_team=nearest.team,
                    nearest_distance_m=nearest_distance,
                    reason="acquire_pending",
                )

        self.current_track_id = None
        self.current_team = None
        self.current_distance_m = None
        self._clear_pending()

        return PossessionResult(
            state=LOOSE,
            possessor_track_id=None,
            possessor_team=None,
            distance_m=None,
            confidence=0.55 if ball_detected else 0.35,
            nearest_track_id=nearest_track_id,
            nearest_team=nearest_team,
            nearest_distance_m=nearest_distance,
            reason="no_player_in_control_radius",
        )

    def _confidence(
        self,
        distance_m: float,
        ball_detected: bool,
        ball_predicted: bool,
    ) -> float:
        max_d = max(self.config.release_distance_m, 1e-6)
        spatial = max(0.0, 1.0 - float(distance_m) / max_d)
        base = 0.45 + 0.50 * spatial
        if ball_predicted and not ball_detected:
            base *= 0.72
        return min(0.99, max(0.05, base))
