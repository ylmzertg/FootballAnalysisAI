from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IncidentCandidate:
    incident_id: str
    attacking_team: str
    start_frame: int
    end_frame: int
    peak_frame: int
    attack_merit_level: str
    defense_vulnerability_level: str
    shot_detected: bool
    error_types: tuple[str, ...]


@dataclass
class IncidentSelectionConfig:
    max_incidents: int = 3
    overlap_suppression_ratio: float = 0.55


def level_score(value: str) -> float:
    return {
        "LOW": 0.25,
        "MEDIUM": 0.60,
        "HIGH": 1.00,
    }.get(str(value).upper(), 0.0)


def incident_score(incident: IncidentCandidate) -> float:
    score = (
        1.25 * level_score(incident.attack_merit_level)
        + 1.25 * level_score(incident.defense_vulnerability_level)
        + 0.15 * min(3, len(incident.error_types))
    )

    if incident.shot_detected:
        score += 2.5

    return float(score)


def overlap_ratio(
    a: IncidentCandidate,
    b: IncidentCandidate,
) -> float:
    start = max(
        a.start_frame,
        b.start_frame,
    )
    end = min(
        a.end_frame,
        b.end_frame,
    )

    if end < start:
        return 0.0

    overlap = end - start + 1

    duration_a = max(
        1,
        a.end_frame - a.start_frame + 1,
    )
    duration_b = max(
        1,
        b.end_frame - b.start_frame + 1,
    )

    return overlap / min(
        duration_a,
        duration_b,
    )


def select_incidents(
    incidents: Iterable[IncidentCandidate],
    config: IncidentSelectionConfig | None = None,
) -> list[IncidentCandidate]:
    cfg = config or IncidentSelectionConfig()

    ranked = sorted(
        list(incidents),
        key=lambda x: (
            incident_score(x),
            x.shot_detected,
            -x.peak_frame,
        ),
        reverse=True,
    )

    selected: list[IncidentCandidate] = []

    for incident in ranked:
        suppress = False

        for existing in selected:
            if (
                overlap_ratio(
                    incident,
                    existing,
                )
                >= cfg.overlap_suppression_ratio
            ):
                suppress = True
                break

        if suppress:
            continue

        selected.append(
            incident
        )

        if (
            len(selected)
            >= cfg.max_incidents
        ):
            break

    return sorted(
        selected,
        key=lambda x: x.peak_frame,
    )
