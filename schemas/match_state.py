from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class Point2D:
    x: float
    y: float


@dataclass(slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def foot_point(self) -> Point2D:
        return Point2D(
            x=self.x + self.width / 2.0,
            y=self.y + self.height,
        )


@dataclass(slots=True)
class CameraState:
    homography: Optional[list[list[float]]] = None
    calibration_confidence: Optional[float] = None
    calibration_engine: Optional[str] = None


@dataclass(slots=True)
class PlayerState:
    track_id: int
    bbox: BoundingBox
    team_id: Optional[str] = None
    role: str = "player"
    jersey_number: Optional[int] = None
    pitch_position: Optional[Point2D] = None
    detection_confidence: float = 1.0
    tracking_confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BallState:
    bbox: Optional[BoundingBox] = None
    image_position: Optional[Point2D] = None
    pitch_position: Optional[Point2D] = None
    confidence: float = 0.0
    visible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MatchFrame:
    frame_index: int
    timestamp_seconds: float
    camera: CameraState = field(default_factory=CameraState)
    players: list[PlayerState] = field(default_factory=list)
    ball: Optional[BallState] = None
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
