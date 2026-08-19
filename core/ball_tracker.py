
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class BallCandidate:
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    confidence: float


@dataclass
class BallTrackResult:
    frame_index: int
    bbox_xyxy: Optional[tuple[float, float, float, float]]
    center_xy: Optional[tuple[float, float]]
    confidence: float
    detected: bool
    predicted: bool
    gap_frames: int
    candidate_count: int


@dataclass
class BallTrackerConfig:
    history_size: int = 12
    max_gap_frames: int = 5
    max_jump_px: float = 260.0
    confidence_weight: float = 90.0
    prediction_decay: float = 0.72


class BallTemporalTracker:
    def __init__(self, config: BallTrackerConfig | None = None):
        self.config = config or BallTrackerConfig()
        self._history = deque(maxlen=max(2, self.config.history_size))
        self._last_bbox = None
        self._last_frame = None

    def _predicted_center(self, frame_index: int):
        if not self._history:
            return None
        if len(self._history) == 1:
            _, x, y, _ = self._history[-1]
            return np.array([x, y], dtype=np.float64)
        f1, x1, y1, _ = self._history[-2]
        f2, x2, y2, _ = self._history[-1]
        dt = max(1, f2 - f1)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        ahead = max(0, frame_index - f2)
        return np.array([x2 + vx * ahead, y2 + vy * ahead], dtype=np.float64)

    def _predict_gap(self, frame_index: int, candidate_count: int):
        if not self._history or self._last_frame is None:
            return BallTrackResult(frame_index, None, None, 0.0, False, False, 0, candidate_count)
        gap = max(1, frame_index - self._last_frame)
        if gap > self.config.max_gap_frames:
            return BallTrackResult(frame_index, None, None, 0.0, False, False, gap, candidate_count)
        center = self._predicted_center(frame_index)
        if center is None:
            return BallTrackResult(frame_index, None, None, 0.0, False, False, gap, candidate_count)
        last_conf = float(self._history[-1][3])
        confidence = last_conf * (self.config.prediction_decay ** gap)
        bbox = None
        if self._last_bbox is not None:
            x1, y1, x2, y2 = self._last_bbox
            w, h = x2 - x1, y2 - y1
            cx, cy = float(center[0]), float(center[1])
            bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
        return BallTrackResult(
            frame_index=frame_index,
            bbox_xyxy=bbox,
            center_xy=(float(center[0]), float(center[1])),
            confidence=confidence,
            detected=False,
            predicted=True,
            gap_frames=gap,
            candidate_count=candidate_count,
        )

    def update(self, candidates: Iterable[BallCandidate], frame_index: int):
        candidates = list(candidates)
        pred = self._predicted_center(frame_index)
        gap = 0 if self._last_frame is None else max(0, frame_index - self._last_frame - 1)

        if not candidates:
            return self._predict_gap(frame_index, 0)

        if pred is None:
            selected = max(candidates, key=lambda c: c.confidence)
        else:
            def cost(c):
                d = float(np.linalg.norm(np.asarray(c.center_xy) - pred))
                return d - self.config.confidence_weight * c.confidence
            selected = min(candidates, key=cost)
            d = float(np.linalg.norm(np.asarray(selected.center_xy) - pred))
            allowed = self.config.max_jump_px * (1.0 + 0.35 * min(gap, self.config.max_gap_frames))
            if d > allowed and gap <= self.config.max_gap_frames:
                return self._predict_gap(frame_index, len(candidates))

        cx, cy = selected.center_xy
        self._history.append((frame_index, float(cx), float(cy), float(selected.confidence)))
        self._last_bbox = tuple(float(v) for v in selected.bbox_xyxy)
        self._last_frame = frame_index

        return BallTrackResult(
            frame_index=frame_index,
            bbox_xyxy=self._last_bbox,
            center_xy=(float(cx), float(cy)),
            confidence=float(selected.confidence),
            detected=True,
            predicted=False,
            gap_frames=0,
            candidate_count=len(candidates),
        )
