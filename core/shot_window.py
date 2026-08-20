from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

SHOT_FLIGHT = "SHOT_FLIGHT"
GOAL_ATTEMPT = "GOAL_ATTEMPT"
ATTACKING_FLIGHT = "ATTACKING_FLIGHT"

@dataclass(frozen=True)
class LocalGoalSample:
    frame_index: int
    normalized_goal_distance: float

@dataclass(frozen=True)
class LocalShotWindow:
    start_frame: int
    end_frame: int
    window_size: int
    sample_count: int
    start_distance: float
    closest_distance: float
    end_distance: float
    closing: float
    end_closing: float
    approach_fraction: float
    closing_per_frame: float
    classification: str
    confidence: float
    reason: str

@dataclass
class LocalShotWindowConfig:
    window_sizes: tuple[int, ...] = (5, 8, 12, 15)
    max_frame_slack: int = 4
    shot_min_closing: float = 1.20
    shot_min_end_closing: float = 0.95
    shot_min_approach_fraction: float = 0.40
    shot_min_closing_per_frame: float = 0.18
    shot_max_closest_distance: float = 8.0
    attacking_min_closing: float = 0.60
    attacking_min_approach_fraction: float = 0.35
    goal_attempt_max_closest_distance: float = 1.80
    goal_attempt_min_closing: float = 1.35
    goal_attempt_min_approach_fraction: float = 0.48

class LocalShotWindowDetector:
    def __init__(self, config: LocalShotWindowConfig | None = None):
        self.config = config or LocalShotWindowConfig()

    def _classify(self, closest, closing, end_closing, approach, closing_pf):
        c = self.config
        if (
            closest <= c.goal_attempt_max_closest_distance
            and closing >= c.goal_attempt_min_closing
            and approach >= c.goal_attempt_min_approach_fraction
        ):
            return GOAL_ATTEMPT, 0.88, "local_goal_closing_reaches_goal_attempt_zone"
        if (
            closest <= c.shot_max_closest_distance
            and closing >= c.shot_min_closing
            and end_closing >= c.shot_min_end_closing
            and approach >= c.shot_min_approach_fraction
            and closing_pf >= c.shot_min_closing_per_frame
        ):
            return SHOT_FLIGHT, 0.80, "strong_local_goal_closing_window"
        if (
            closing >= c.attacking_min_closing
            and approach >= c.attacking_min_approach_fraction
        ):
            return ATTACKING_FLIGHT, 0.62, "local_attacking_goal_progress"
        return "NONE", 0.0, "no_local_shot_window"

    def windows(self, samples: Iterable[LocalGoalSample]) -> list[LocalShotWindow]:
        samples = sorted(list(samples), key=lambda x: x.frame_index)
        out = []
        c = self.config
        for w in c.window_sizes:
            if len(samples) < w:
                continue
            for i in range(len(samples)-w+1):
                chunk = samples[i:i+w]
                if chunk[-1].frame_index - chunk[0].frame_index > w + c.max_frame_slack:
                    continue
                ds = [float(s.normalized_goal_distance) for s in chunk]
                start, end, closest = ds[0], ds[-1], min(ds)
                closing = start - closest
                end_closing = start - end
                approaching = sum(1 for a,b in zip(ds, ds[1:]) if b < a - 0.03)
                approach = approaching / max(1, len(ds)-1)
                span = max(1, chunk[-1].frame_index - chunk[0].frame_index)
                closing_pf = closing / span
                klass, conf, reason = self._classify(
                    closest, closing, end_closing, approach, closing_pf
                )
                if klass == "NONE":
                    continue
                out.append(LocalShotWindow(
                    chunk[0].frame_index, chunk[-1].frame_index, w, len(chunk),
                    start, closest, end, closing, end_closing, approach,
                    closing_pf, klass, conf, reason
                ))
        return out

    def best_window(self, samples: Iterable[LocalGoalSample]) -> Optional[LocalShotWindow]:
        windows = self.windows(samples)
        if not windows:
            return None
        rank = {GOAL_ATTEMPT: 3, SHOT_FLIGHT: 2, ATTACKING_FLIGHT: 1}
        return max(
            windows,
            key=lambda x: (
                rank.get(x.classification, 0),
                x.closing,
                x.approach_fraction,
                x.closing_per_frame,
                -x.window_size,
            )
        )
