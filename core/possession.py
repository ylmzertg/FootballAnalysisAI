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
    image_foot_xy: Optional[tuple[float, float]] = None
    bbox_height_px: Optional[float] = None


@dataclass
class PossessionConfig:
    acquire_distance_m: float = 3.2
    release_distance_m: float = 5.0
    switch_margin_m: float = 0.75
    confirm_frames: int = 2
    hold_missing_ball_frames: int = 4
    allow_goalkeeper: bool = True

    # Perspective-normalized image-space fallback.
    acquire_image_ratio: float = 0.75
    release_image_ratio: float = 1.20
    min_bbox_height_px: float = 18.0
    switch_score_margin: float = 0.15


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
    image_distance_px: Optional[float] = None
    image_distance_ratio: Optional[float] = None
    control_source: str = "NONE"


@dataclass(frozen=True)
class _Match:
    player: PlayerPossessionCandidate
    pitch_distance_m: Optional[float]
    image_distance_px: Optional[float]
    image_distance_ratio: Optional[float]
    score: float
    acquire_ok: bool
    release_ok: bool
    source: str


class PossessionEstimator:
    """
    Possession v1.1.

    Combines:
      - ground-plane pitch distance when PnL projection is trustworthy;
      - perspective-normalized image foot distance as a fallback.

    This is especially useful when an airborne ball projects poorly through a
    ground-plane homography.
    """

    def __init__(self, config: PossessionConfig | None = None):
        self.config = config or PossessionConfig()

        self.current_track_id: Optional[int] = None
        self.current_team: Optional[str] = None
        self.current_distance_m: Optional[float] = None
        self.current_image_ratio: Optional[float] = None
        self.current_source: str = "NONE"

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
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _match(
        self,
        player: PlayerPossessionCandidate,
        ball_pitch_xy: Optional[tuple[float, float]],
        ball_image_xy: Optional[tuple[float, float]],
    ) -> _Match:
        pitch_d = None
        if ball_pitch_xy is not None:
            pitch_d = self._distance(ball_pitch_xy, player.pitch_xy)

        image_d = None
        image_ratio = None

        if ball_image_xy is not None and player.image_foot_xy is not None:
            image_d = self._distance(ball_image_xy, player.image_foot_xy)
            h = float(player.bbox_height_px or 0.0)
            if h >= self.config.min_bbox_height_px:
                image_ratio = image_d / h

        norms = []
        source_parts = []

        if pitch_d is not None:
            norms.append(
                pitch_d / max(self.config.acquire_distance_m, 1e-6)
            )
            source_parts.append("PITCH")

        if image_ratio is not None:
            norms.append(
                image_ratio / max(self.config.acquire_image_ratio, 1e-6)
            )
            source_parts.append("IMAGE")

        if not norms:
            score = float("inf")
            source = "NONE"
        elif len(norms) == 1:
            score = norms[0]
            source = source_parts[0]
        else:
            lo = min(norms)
            hi = max(norms)
            # Let one strong signal rescue a distorted one, but retain a small
            # disagreement penalty.
            score = lo + 0.15 * hi
            source = "BOTH"

        acquire_ok = (
            (pitch_d is not None and pitch_d <= self.config.acquire_distance_m)
            or (
                image_ratio is not None
                and image_ratio <= self.config.acquire_image_ratio
            )
        )

        release_ok = (
            (pitch_d is not None and pitch_d <= self.config.release_distance_m)
            or (
                image_ratio is not None
                and image_ratio <= self.config.release_image_ratio
            )
        )

        return _Match(
            player=player,
            pitch_distance_m=pitch_d,
            image_distance_px=image_d,
            image_distance_ratio=image_ratio,
            score=score,
            acquire_ok=acquire_ok,
            release_ok=release_ok,
            source=source,
        )

    def _ranked_matches(
        self,
        players: list[PlayerPossessionCandidate],
        ball_pitch_xy: Optional[tuple[float, float]],
        ball_image_xy: Optional[tuple[float, float]],
    ) -> list[_Match]:
        matches = [
            self._match(p, ball_pitch_xy, ball_image_xy)
            for p in players
        ]
        matches = [m for m in matches if m.score != float("inf")]
        matches.sort(key=lambda m: m.score)
        return matches

    def _current_match(self, matches: list[_Match]) -> Optional[_Match]:
        if self.current_track_id is None:
            return None
        for m in matches:
            if m.player.track_id == self.current_track_id:
                return m
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

    def _commit(self, match: _Match):
        self.current_track_id = match.player.track_id
        self.current_team = match.player.team
        self.current_distance_m = match.pitch_distance_m
        self.current_image_ratio = match.image_distance_ratio
        self.current_source = match.source
        self._clear_pending()

    def _switch_is_clear(
        self,
        candidate: _Match,
        current: _Match,
    ) -> bool:
        if (
            candidate.pitch_distance_m is not None
            and current.pitch_distance_m is not None
        ):
            return (
                candidate.pitch_distance_m + self.config.switch_margin_m
                < current.pitch_distance_m
            )

        return (
            candidate.score + self.config.switch_score_margin
            < current.score
        )

    def update(
        self,
        ball_xy: Optional[tuple[float, float]],
        players: Iterable[PlayerPossessionCandidate],
        *,
        ball_detected: bool,
        ball_predicted: bool,
        ball_image_xy: Optional[tuple[float, float]] = None,
    ) -> PossessionResult:
        players = self._eligible(players)

        # True ball missing means neither pitch nor image evidence exists.
        if ball_xy is None and ball_image_xy is None:
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
                    image_distance_ratio=self.current_image_ratio,
                    control_source=self.current_source,
                )

            self.current_track_id = None
            self.current_team = None
            self.current_distance_m = None
            self.current_image_ratio = None
            self.current_source = "NONE"

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

        matches = self._ranked_matches(
            players,
            ball_xy,
            ball_image_xy,
        )

        nearest = matches[0] if matches else None
        current = self._current_match(matches)

        nearest_track_id = nearest.player.track_id if nearest else None
        nearest_team = nearest.player.team if nearest else None
        nearest_distance_m = nearest.pitch_distance_m if nearest else None

        # Maintain current owner while either spatial signal says control is
        # still plausible.
        if current is not None and current.release_ok:
            if (
                nearest is not None
                and nearest.player.track_id != current.player.track_id
                and nearest.acquire_ok
                and self._switch_is_clear(nearest, current)
            ):
                self._set_pending(
                    nearest.player.track_id,
                    nearest.player.team,
                )

                if self.pending_count >= self.config.confirm_frames:
                    self._commit(nearest)
                    return self._result_from_match(
                        nearest,
                        ball_detected,
                        ball_predicted,
                        reason="confirmed_switch",
                        nearest=nearest,
                    )
            else:
                self._clear_pending()

            self.current_distance_m = current.pitch_distance_m
            self.current_image_ratio = current.image_distance_ratio
            self.current_source = current.source

            return self._result_from_match(
                current,
                ball_detected,
                ball_predicted,
                reason="owner_hysteresis",
                nearest=nearest,
            )

        # Acquire a new owner.
        if nearest is not None and nearest.acquire_ok:
            self._set_pending(
                nearest.player.track_id,
                nearest.player.team,
            )

            if self.pending_count >= self.config.confirm_frames:
                self._commit(nearest)
                return self._result_from_match(
                    nearest,
                    ball_detected,
                    ball_predicted,
                    reason="confirmed_acquire",
                    nearest=nearest,
                )

            return PossessionResult(
                state=LOOSE,
                possessor_track_id=None,
                possessor_team=None,
                distance_m=None,
                confidence=0.35,
                nearest_track_id=nearest.player.track_id,
                nearest_team=nearest.player.team,
                nearest_distance_m=nearest.pitch_distance_m,
                reason="acquire_pending",
                image_distance_px=nearest.image_distance_px,
                image_distance_ratio=nearest.image_distance_ratio,
                control_source=nearest.source,
            )

        self.current_track_id = None
        self.current_team = None
        self.current_distance_m = None
        self.current_image_ratio = None
        self.current_source = "NONE"
        self._clear_pending()

        return PossessionResult(
            state=LOOSE,
            possessor_track_id=None,
            possessor_team=None,
            distance_m=None,
            confidence=0.55 if ball_detected else 0.35,
            nearest_track_id=nearest_track_id,
            nearest_team=nearest_team,
            nearest_distance_m=nearest_distance_m,
            reason="no_player_in_control_radius",
            image_distance_px=nearest.image_distance_px if nearest else None,
            image_distance_ratio=nearest.image_distance_ratio if nearest else None,
            control_source=nearest.source if nearest else "NONE",
        )

    def _result_from_match(
        self,
        match: _Match,
        ball_detected: bool,
        ball_predicted: bool,
        *,
        reason: str,
        nearest: Optional[_Match],
    ) -> PossessionResult:
        return PossessionResult(
            state=match.player.team,
            possessor_track_id=match.player.track_id,
            possessor_team=match.player.team,
            distance_m=match.pitch_distance_m,
            confidence=self._confidence(
                match.score,
                ball_detected,
                ball_predicted,
            ),
            nearest_track_id=nearest.player.track_id if nearest else None,
            nearest_team=nearest.player.team if nearest else None,
            nearest_distance_m=nearest.pitch_distance_m if nearest else None,
            reason=reason,
            image_distance_px=match.image_distance_px,
            image_distance_ratio=match.image_distance_ratio,
            control_source=match.source,
        )

    @staticmethod
    def _confidence(
        score: float,
        ball_detected: bool,
        ball_predicted: bool,
    ) -> float:
        spatial = max(0.0, min(1.0, 1.0 - score / 2.0))
        base = 0.45 + 0.50 * spatial
        if ball_predicted and not ball_detected:
            base *= 0.72
        return min(0.99, max(0.05, base))
