from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemas.match_state import BallState, CameraState, MatchFrame, PlayerState


@dataclass(slots=True)
class AdapterInfo:
    name: str
    version: str | None = None
    source: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    def __init__(self, info: AdapterInfo):
        self.info = info

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Return (is_ready, human_readable_message)."""
        raise NotImplementedError


class CalibrationAdapter(BaseAdapter):
    @abstractmethod
    def calibrate(self, frame_path: Path) -> CameraState:
        raise NotImplementedError


class DetectionAdapter(BaseAdapter):
    @abstractmethod
    def detect_players(self, frame_path: Path) -> list[PlayerState]:
        raise NotImplementedError

    @abstractmethod
    def detect_ball(self, frame_path: Path) -> BallState | None:
        raise NotImplementedError


class TrackingAdapter(BaseAdapter):
    @abstractmethod
    def track(self, video_path: Path) -> list[MatchFrame]:
        raise NotImplementedError


class TeamAdapter(BaseAdapter):
    @abstractmethod
    def assign_teams(self, frames: list[MatchFrame]) -> list[MatchFrame]:
        raise NotImplementedError


class GameStateAdapter(BaseAdapter):
    @abstractmethod
    def reconstruct(self, video_path: Path) -> list[MatchFrame]:
        """End-to-end adapter, e.g. SoccerNet GSR."""
        raise NotImplementedError
