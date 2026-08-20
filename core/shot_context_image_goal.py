from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional

import numpy as np


PLUS_X = "PLUS_X"
MINUS_X = "MINUS_X"
UNKNOWN = "UNKNOWN"

RAW_LOOSE = "RAW_LOOSE"
TEAM_FLIGHT = "TEAM_FLIGHT"
PASS_FLIGHT = "PASS_FLIGHT"
ATTACKING_FLIGHT = "ATTACKING_FLIGHT"

SHOT_FLIGHT = "SHOT_FLIGHT"
GOAL_ATTEMPT = "GOAL_ATTEMPT"

GOAL_HALF_WIDTH_M = 7.32 / 2.0


@dataclass(frozen=True)
class GoalImageGeometry:
    center_xy: tuple[float, float]
    post_a_xy: tuple[float, float]
    post_b_xy: tuple[float, float]
    mouth_width_px: float


@dataclass(frozen=True)
class GoalApproachSample:
    frame_index: int
    normalized_goal_distance: float
    ball_xy: tuple[float, float]
    goal_center_xy: tuple[float, float]
    goal_mouth_width_px: float


@dataclass(frozen=True)
class ImageGoalRunResult:
    start_frame: int
    end_frame: int
    team: Optional[str]
    classification: str
    confidence: float
    valid_samples: int
    start_goal_distance_units: Optional[float]
    closest_goal_distance_units: Optional[float]
    closing_progress_units: float
    approach_fraction: float
    closest_frame: Optional[int]
    reason: str


@dataclass
class ImageGoalShotConfig:
    min_samples: int = 3
    min_goal_mouth_px: float = 6.0

    attacking_min_closing_units: float = 0.45
    shot_min_closing_units: float = 0.90
    goal_attempt_min_closing_units: float = 1.35

    shot_max_closest_units: float = 4.0
    goal_attempt_max_closest_units: float = 1.8

    min_approach_fraction: float = 0.35
    shot_min_approach_fraction: float = 0.42
    goal_attempt_min_approach_fraction: float = 0.48


def project_pitch_to_image(
    homography_image_to_pitch: np.ndarray,
    pitch_xy: tuple[float, float],
) -> Optional[tuple[float, float]]:
    H = np.asarray(
        homography_image_to_pitch,
        dtype=np.float64,
    )

    if H.shape != (3, 3):
        return None

    try:
        inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    p = inv @ np.array(
        [
            float(pitch_xy[0]),
            float(pitch_xy[1]),
            1.0,
        ],
        dtype=np.float64,
    )

    if abs(float(p[2])) < 1e-12:
        return None

    p = p / p[2]

    if not np.isfinite(p).all():
        return None

    return float(p[0]), float(p[1])


def opponent_goal_x(direction: str) -> Optional[float]:
    if direction == PLUS_X:
        return 105.0
    if direction == MINUS_X:
        return 0.0
    return None


def goal_image_geometry(
    homography_image_to_pitch: np.ndarray,
    attack_direction: str,
) -> Optional[GoalImageGeometry]:
    gx = opponent_goal_x(attack_direction)

    if gx is None:
        return None

    center = project_pitch_to_image(
        homography_image_to_pitch,
        (gx, 34.0),
    )
    post_a = project_pitch_to_image(
        homography_image_to_pitch,
        (gx, 34.0 - GOAL_HALF_WIDTH_M),
    )
    post_b = project_pitch_to_image(
        homography_image_to_pitch,
        (gx, 34.0 + GOAL_HALF_WIDTH_M),
    )

    if (
        center is None
        or post_a is None
        or post_b is None
    ):
        return None

    width = hypot(
        float(post_a[0]) - float(post_b[0]),
        float(post_a[1]) - float(post_b[1]),
    )

    if not np.isfinite(width):
        return None

    return GoalImageGeometry(
        center_xy=center,
        post_a_xy=post_a,
        post_b_xy=post_b,
        mouth_width_px=float(width),
    )


def normalized_ball_goal_distance(
    ball_xy: tuple[float, float],
    geometry: GoalImageGeometry,
    min_goal_mouth_px: float,
) -> Optional[float]:
    if geometry.mouth_width_px < min_goal_mouth_px:
        return None

    d = hypot(
        float(ball_xy[0]) - float(geometry.center_xy[0]),
        float(ball_xy[1]) - float(geometry.center_xy[1]),
    )

    return d / geometry.mouth_width_px


class ImageGoalShotClassifier:
    """
    Shot-context classifier that avoids projecting an airborne ball onto the
    ground plane.

    Each frame:
      1. project the opponent goal mouth from pitch -> image using calibration;
      2. measure ball-image distance to that moving goal target;
      3. normalize by projected goal-mouth width.

    Because both the target and scale come from the same frame calibration,
    camera pan/zoom is substantially less harmful than raw image coordinates.
    """

    def __init__(
        self,
        config: ImageGoalShotConfig | None = None,
    ):
        self.config = config or ImageGoalShotConfig()

    def classify(
        self,
        *,
        start_frame: int,
        end_frame: int,
        team: Optional[str],
        samples: list[GoalApproachSample],
        fallback_phase: str,
    ) -> ImageGoalRunResult:
        cfg = self.config

        if len(samples) < cfg.min_samples:
            return ImageGoalRunResult(
                start_frame=start_frame,
                end_frame=end_frame,
                team=team,
                classification=fallback_phase,
                confidence=0.35,
                valid_samples=len(samples),
                start_goal_distance_units=None,
                closest_goal_distance_units=None,
                closing_progress_units=0.0,
                approach_fraction=0.0,
                closest_frame=None,
                reason="insufficient_image_goal_samples",
            )

        samples = sorted(
            samples,
            key=lambda x: x.frame_index,
        )

        distances = [
            float(s.normalized_goal_distance)
            for s in samples
        ]

        start_distance = distances[0]
        closest_distance = min(distances)
        closest_index = distances.index(
            closest_distance
        )
        closest_frame = samples[
            closest_index
        ].frame_index

        closing = max(
            0.0,
            start_distance - closest_distance,
        )

        approaching_steps = 0
        comparable_steps = 0

        for prev, cur in zip(
            distances,
            distances[1:],
        ):
            comparable_steps += 1

            # Small tolerance prevents calibration jitter from counting as
            # meaningful motion.
            if cur < prev - 0.03:
                approaching_steps += 1

        approach_fraction = (
            approaching_steps
            / comparable_steps
            if comparable_steps > 0
            else 0.0
        )

        if (
            closing
            >= cfg.goal_attempt_min_closing_units
            and closest_distance
            <= cfg.goal_attempt_max_closest_units
            and approach_fraction
            >= cfg.goal_attempt_min_approach_fraction
        ):
            classification = GOAL_ATTEMPT
            confidence = 0.86
            reason = (
                "image_ball_strongly_closes_on_goal_mouth"
            )

        elif (
            closing
            >= cfg.shot_min_closing_units
            and closest_distance
            <= cfg.shot_max_closest_units
            and approach_fraction
            >= cfg.shot_min_approach_fraction
        ):
            classification = SHOT_FLIGHT
            confidence = 0.78
            reason = (
                "image_ball_closes_toward_opponent_goal"
            )

        elif (
            closing
            >= cfg.attacking_min_closing_units
            and approach_fraction
            >= cfg.min_approach_fraction
        ):
            classification = ATTACKING_FLIGHT
            confidence = 0.64
            reason = (
                "image_ball_has_attacking_goal_progress"
            )

        else:
            classification = fallback_phase
            confidence = 0.44
            reason = (
                "no_strong_image_goal_approach"
            )

        return ImageGoalRunResult(
            start_frame=start_frame,
            end_frame=end_frame,
            team=team,
            classification=classification,
            confidence=confidence,
            valid_samples=len(samples),
            start_goal_distance_units=start_distance,
            closest_goal_distance_units=closest_distance,
            closing_progress_units=closing,
            approach_fraction=approach_fraction,
            closest_frame=closest_frame,
            reason=reason,
        )
