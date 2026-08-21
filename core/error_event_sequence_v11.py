from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.error_detection_v1 import ErrorEvent


@dataclass(frozen=True)
class SequencedErrorEvent:
    event_id: str
    error_type: str
    attacking_team: str
    defending_team: str
    primary_track_id: int | None
    secondary_track_id: int | None
    start_frame: int
    end_frame: int
    peak_frame: int
    duration_frames: int
    severity: str
    peak_metric_value: float | None
    explanation: str
    evidence: dict


@dataclass
class ErrorSequenceConfig:
    merge_gap_frames: int = 3
    min_duration_late_pressure: int = 2
    min_duration_unmarked_runner: int = 3
    min_duration_free_lane: int = 3


def _severity_rank(value: str) -> int:
    return {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
    }.get(str(value).upper(), 0)


def _minimum_duration(error_type: str, cfg: ErrorSequenceConfig) -> int:
    if error_type == "LATE_PRESSURE":
        return cfg.min_duration_late_pressure
    if error_type == "UNMARKED_RUNNER":
        return cfg.min_duration_unmarked_runner
    if error_type == "FREE_PASSING_LANE":
        return cfg.min_duration_free_lane
    return 2


def sequence_error_events(
    frame_events: dict[int, list[ErrorEvent]],
    config: ErrorSequenceConfig | None = None,
) -> list[SequencedErrorEvent]:
    cfg = config or ErrorSequenceConfig()

    grouped: dict[
        tuple[str, str, int | None],
        list[ErrorEvent],
    ] = {}

    for frame_index in sorted(frame_events):
        for event in frame_events[frame_index]:
            key = (
                event.error_type,
                event.attacking_team,
                event.primary_track_id,
            )
            grouped.setdefault(key, []).append(event)

    sequenced: list[SequencedErrorEvent] = []
    serial = 1

    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda e: e.frame_index)

        runs: list[list[ErrorEvent]] = []
        current: list[ErrorEvent] = []

        for event in rows:
            if not current:
                current = [event]
                continue

            if (
                event.frame_index
                - current[-1].frame_index
                <= cfg.merge_gap_frames + 1
            ):
                current.append(event)
            else:
                runs.append(current)
                current = [event]

        if current:
            runs.append(current)

        for run in runs:
            error_type = run[0].error_type

            if len(run) < _minimum_duration(error_type, cfg):
                continue

            # Higher metric means more severe for all V1 error types:
            # defender distance / marking distance / open-lane distance.
            peak = max(
                run,
                key=lambda e: (
                    float(e.metric_value)
                    if e.metric_value is not None
                    else -1.0,
                    _severity_rank(e.severity),
                ),
            )

            severity = max(
                (e.severity for e in run),
                key=_severity_rank,
            )

            start_frame = run[0].frame_index
            end_frame = run[-1].frame_index

            sequenced.append(
                SequencedErrorEvent(
                    event_id=f"ERR-{serial:04d}",
                    error_type=error_type,
                    attacking_team=peak.attacking_team,
                    defending_team=peak.defending_team,
                    primary_track_id=peak.primary_track_id,
                    secondary_track_id=peak.secondary_track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    peak_frame=peak.frame_index,
                    duration_frames=end_frame - start_frame + 1,
                    severity=severity,
                    peak_metric_value=peak.metric_value,
                    explanation=peak.explanation,
                    evidence=peak.evidence,
                )
            )
            serial += 1

    return sorted(
        sequenced,
        key=lambda e: (
            e.start_frame,
            e.peak_frame,
            e.error_type,
        ),
    )
