from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
LOOSE = "LOOSE"
UNKNOWN = "UNKNOWN"

CONTROL = "CONTROL"
CONTROL_GAP = "CONTROL_GAP"
PASS_FLIGHT = "PASS_FLIGHT"
TEAM_FLIGHT = "TEAM_FLIGHT"
CONTESTED_FLIGHT = "CONTESTED_FLIGHT"
RAW_LOOSE = "RAW_LOOSE"
RAW_UNKNOWN = "RAW_UNKNOWN"


@dataclass(frozen=True)
class ControlFrame:
    frame_index: int
    state: str
    possessor_track_id: Optional[int]
    possessor_team: Optional[str]
    ball_image_xy: Optional[tuple[float, float]]
    ball_detected: bool
    ball_predicted: bool


@dataclass(frozen=True)
class EventFrame:
    frame_index: int
    team_state: str
    phase: str
    possessor_track_id: Optional[int]
    source_owner_track_id: Optional[int]
    target_owner_track_id: Optional[int]
    source_team: Optional[str]
    target_team: Optional[str]
    confidence: float
    reason: str


@dataclass
class PossessionEventConfig:
    max_bridge_gap_frames: int = 30
    max_unresolved_flight_frames: int = 18
    min_ball_motion_px: float = 8.0
    max_missing_ball_ratio: float = 0.45


class PossessionEventReconstructor:
    """
    Offline event reconstruction over frame-level possession.

    It does not invent per-frame player control during a pass. Instead it
    preserves *team* possession through a ball-flight phase when two confirmed
    control segments support that interpretation.
    """

    def __init__(self, config: PossessionEventConfig | None = None):
        self.config = config or PossessionEventConfig()

    @staticmethod
    def _is_control(frame: ControlFrame) -> bool:
        return (
            frame.state in {TEAM_A, TEAM_B}
            and frame.possessor_track_id is not None
            and frame.possessor_team in {TEAM_A, TEAM_B}
        )

    @staticmethod
    def _ball_motion_px(frames: list[ControlFrame]) -> float:
        points = [
            f.ball_image_xy
            for f in frames
            if f.ball_image_xy is not None
        ]
        if len(points) < 2:
            return 0.0

        total = 0.0
        prev = points[0]

        for cur in points[1:]:
            dx = float(cur[0]) - float(prev[0])
            dy = float(cur[1]) - float(prev[1])
            total += (dx * dx + dy * dy) ** 0.5
            prev = cur

        return total

    @staticmethod
    def _missing_ball_ratio(frames: list[ControlFrame]) -> float:
        if not frames:
            return 1.0

        missing = sum(
            1
            for f in frames
            if f.ball_image_xy is None
        )
        return missing / len(frames)

    def reconstruct(
        self,
        frames: list[ControlFrame],
    ) -> list[EventFrame]:
        if not frames:
            return []

        frames = sorted(
            frames,
            key=lambda x: x.frame_index,
        )

        events: list[Optional[EventFrame]] = [
            None
            for _ in frames
        ]

        # Confirmed control frames are authoritative.
        for i, frame in enumerate(frames):
            if self._is_control(frame):
                events[i] = EventFrame(
                    frame_index=frame.frame_index,
                    team_state=frame.possessor_team or frame.state,
                    phase=CONTROL,
                    possessor_track_id=frame.possessor_track_id,
                    source_owner_track_id=frame.possessor_track_id,
                    target_owner_track_id=frame.possessor_track_id,
                    source_team=frame.possessor_team,
                    target_team=frame.possessor_team,
                    confidence=0.95,
                    reason="confirmed_frame_control",
                )

        i = 0

        while i < len(frames):
            if events[i] is not None:
                i += 1
                continue

            start = i

            while (
                i < len(frames)
                and events[i] is None
            ):
                i += 1

            end = i - 1

            prev_i = start - 1
            next_i = i if i < len(frames) else None

            prev_control = (
                frames[prev_i]
                if prev_i >= 0
                and self._is_control(frames[prev_i])
                else None
            )

            next_control = (
                frames[next_i]
                if next_i is not None
                and self._is_control(frames[next_i])
                else None
            )

            gap_frames = frames[start:end + 1]

            motion = self._ball_motion_px(
                gap_frames
            )
            missing_ratio = self._missing_ball_ratio(
                gap_frames
            )

            gap_len = len(gap_frames)

            bridge_eligible = (
                gap_len
                <= self.config.max_bridge_gap_frames
                and missing_ratio
                <= self.config.max_missing_ball_ratio
            )

            # Case 1: confirmed control on both sides.
            if (
                prev_control is not None
                and next_control is not None
                and bridge_eligible
            ):
                same_team = (
                    prev_control.possessor_team
                    == next_control.possessor_team
                )
                same_owner = (
                    prev_control.possessor_track_id
                    == next_control.possessor_track_id
                )

                if same_team and same_owner:
                    phase = CONTROL_GAP
                    team_state = (
                        prev_control.possessor_team
                        or prev_control.state
                    )
                    possessor = (
                        prev_control.possessor_track_id
                    )
                    reason = (
                        "same_owner_control_gap"
                    )
                    confidence = 0.82

                elif same_team:
                    phase = (
                        PASS_FLIGHT
                        if motion
                        >= self.config.min_ball_motion_px
                        else CONTROL_GAP
                    )
                    team_state = (
                        prev_control.possessor_team
                        or prev_control.state
                    )
                    possessor = None
                    reason = (
                        "same_team_control_to_control"
                        if phase == PASS_FLIGHT
                        else "same_team_low_motion_gap"
                    )
                    confidence = (
                        0.88
                        if phase == PASS_FLIGHT
                        else 0.72
                    )

                else:
                    phase = CONTESTED_FLIGHT
                    team_state = LOOSE
                    possessor = None
                    reason = (
                        "control_changes_team"
                    )
                    confidence = 0.72

                for j in range(start, end + 1):
                    events[j] = EventFrame(
                        frame_index=frames[j].frame_index,
                        team_state=team_state,
                        phase=phase,
                        possessor_track_id=possessor,
                        source_owner_track_id=(
                            prev_control.possessor_track_id
                        ),
                        target_owner_track_id=(
                            next_control.possessor_track_id
                        ),
                        source_team=(
                            prev_control.possessor_team
                        ),
                        target_team=(
                            next_control.possessor_team
                        ),
                        confidence=confidence,
                        reason=reason,
                    )

                continue

            # Case 2: unresolved flight after a confirmed owner.
            if (
                prev_control is not None
                and gap_len
                <= self.config.max_unresolved_flight_frames
                and missing_ratio
                <= self.config.max_missing_ball_ratio
                and motion
                >= self.config.min_ball_motion_px
            ):
                team_state = (
                    prev_control.possessor_team
                    or prev_control.state
                )

                for j in range(start, end + 1):
                    events[j] = EventFrame(
                        frame_index=frames[j].frame_index,
                        team_state=team_state,
                        phase=TEAM_FLIGHT,
                        possessor_track_id=None,
                        source_owner_track_id=(
                            prev_control.possessor_track_id
                        ),
                        target_owner_track_id=None,
                        source_team=(
                            prev_control.possessor_team
                        ),
                        target_team=None,
                        confidence=0.62,
                        reason="unresolved_short_team_flight",
                    )

                continue

            # Case 3: retain raw semantics.
            for j in range(start, end + 1):
                raw = frames[j]
                if raw.state == UNKNOWN:
                    phase = RAW_UNKNOWN
                    team_state = UNKNOWN
                else:
                    phase = RAW_LOOSE
                    team_state = LOOSE

                events[j] = EventFrame(
                    frame_index=raw.frame_index,
                    team_state=team_state,
                    phase=phase,
                    possessor_track_id=None,
                    source_owner_track_id=(
                        prev_control.possessor_track_id
                        if prev_control
                        else None
                    ),
                    target_owner_track_id=(
                        next_control.possessor_track_id
                        if next_control
                        else None
                    ),
                    source_team=(
                        prev_control.possessor_team
                        if prev_control
                        else None
                    ),
                    target_team=(
                        next_control.possessor_team
                        if next_control
                        else None
                    ),
                    confidence=0.40,
                    reason="raw_frame_state_preserved",
                )

        return [
            e
            for e in events
            if e is not None
        ]
