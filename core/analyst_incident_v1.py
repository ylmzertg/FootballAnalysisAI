from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class ErrorTimelineItem:
    event_id: str
    error_type: str
    attacking_team: str
    defending_team: str
    start_frame: int
    end_frame: int
    peak_frame: int
    severity: str
    primary_track_id: Optional[int]
    secondary_track_id: Optional[int]
    evidence: dict


@dataclass(frozen=True)
class AnalystIncident:
    incident_id: str
    attacking_team: str
    defending_team: str
    start_frame: int
    end_frame: int
    peak_frame: int

    error_event_ids: tuple[str, ...]
    error_types: tuple[str, ...]

    attack_view: tuple[str, ...]
    defense_view: tuple[str, ...]
    alternative_view: tuple[str, ...]
    outcome_view: tuple[str, ...]

    best_pass_receiver_id: Optional[int]
    best_pass_score: Optional[float]
    shot_detected: bool

    attack_merit_level: str
    defense_vulnerability_level: str


@dataclass
class AnalystIncidentConfig:
    merge_gap_frames: int = 5
    context_before_frames: int = 6
    context_after_frames: int = 22
    outcome_lookahead_frames: int = 45


def severity_score(value: str) -> int:
    return {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
    }.get(str(value).upper(), 0)


def level_from_score(score: float) -> str:
    if score >= 0.72:
        return "HIGH"
    if score >= 0.42:
        return "MEDIUM"
    return "LOW"


def merge_error_timeline(
    errors: Iterable[ErrorTimelineItem],
    config: AnalystIncidentConfig | None = None,
) -> list[list[ErrorTimelineItem]]:
    cfg = config or AnalystIncidentConfig()

    rows = sorted(
        list(errors),
        key=lambda e: (
            e.start_frame,
            e.end_frame,
        ),
    )

    if not rows:
        return []

    groups = []
    current = [rows[0]]
    current_end = rows[0].end_frame

    for event in rows[1:]:
        same_attack = (
            event.attacking_team
            == current[0].attacking_team
        )

        close = (
            event.start_frame
            <= current_end
            + cfg.merge_gap_frames
        )

        if same_attack and close:
            current.append(event)
            current_end = max(
                current_end,
                event.end_frame,
            )
        else:
            groups.append(current)
            current = [event]
            current_end = event.end_frame

    groups.append(current)

    return groups


def build_incidents(
    *,
    errors: Iterable[ErrorTimelineItem],
    pass_options_by_frame: dict[int, list[dict]],
    marking_by_frame: dict[int, list[dict]],
    shot_frames: set[int],
    config: AnalystIncidentConfig | None = None,
) -> list[AnalystIncident]:
    cfg = config or AnalystIncidentConfig()

    groups = merge_error_timeline(
        errors,
        cfg,
    )

    incidents = []

    for serial, group in enumerate(
        groups,
        start=1,
    ):
        attacking_team = group[0].attacking_team
        defending_team = group[0].defending_team

        start_frame = max(
            0,
            min(e.start_frame for e in group)
            - cfg.context_before_frames,
        )

        raw_end = max(
            e.end_frame
            for e in group
        )

        end_frame = (
            raw_end
            + cfg.context_after_frames
        )

        peak_event = max(
            group,
            key=lambda e: (
                severity_score(e.severity),
                len(e.evidence),
                e.peak_frame,
            ),
        )

        peak_frame = peak_event.peak_frame

        # Pass options: inspect a small neighborhood around tactical peak.
        pass_candidates = []

        for frame in range(
            max(0, peak_frame - 2),
            peak_frame + 4,
        ):
            for option in pass_options_by_frame.get(
                frame,
                [],
            ):
                if option.get("category") in {
                    "BEST",
                    "GOOD",
                }:
                    pass_candidates.append(
                        option
                    )

        pass_candidates.sort(
            key=lambda x: float(
                x.get("score", 0.0)
            ),
            reverse=True,
        )

        best_pass = (
            pass_candidates[0]
            if pass_candidates
            else None
        )

        # Marking: use most threatening loose/unmarked attacker around peak.
        marking_candidates = []

        for frame in range(
            max(0, peak_frame - 2),
            peak_frame + 4,
        ):
            for mark in marking_by_frame.get(
                frame,
                [],
            ):
                if (
                    mark.get("marking_state")
                    in {
                        "LOOSE",
                        "UNMARKED",
                    }
                    and mark.get("dangerous")
                ):
                    marking_candidates.append(
                        mark
                    )

        marking_candidates.sort(
            key=lambda x: float(
                x.get(
                    "threat_score",
                    0.0,
                )
            ),
            reverse=True,
        )

        top_mark = (
            marking_candidates[0]
            if marking_candidates
            else None
        )

        shot_detected = any(
            frame in shot_frames
            for frame in range(
                start_frame,
                raw_end
                + cfg.outcome_lookahead_frames
                + 1,
            )
        )

        attack_view = []
        defense_view = []
        alternative_view = []
        outcome_view = []

        error_types = tuple(
            sorted(
                {
                    e.error_type
                    for e in group
                }
            )
        )

        if top_mark is not None:
            attack_view.append(
                (
                    f"ID {top_mark.get('attacker_track_id')} "
                    f"savunma markajından ayrışarak kullanılabilir "
                    f"bir alan oluşturuyor."
                )
            )

            defense_view.append(
                (
                    f"En yakın savunmacı ID "
                    f"{top_mark.get('nearest_defender_track_id')} "
                    f"yaklaşık "
                    f"{float(top_mark.get('nearest_defender_distance_m', 0)):.1f} m "
                    f"uzakta; markaj bağlantısı zayıf."
                )
            )

            alternative_view.append(
                (
                    f"Savunma, ID {top_mark.get('attacker_track_id')} "
                    f"koşusunu daha erken devralabilir veya alanı daraltabilirdi."
                )
            )

        if best_pass is not None:
            attack_view.append(
                (
                    f"Top sahibinin ID "
                    f"{best_pass.get('receiver_track_id')} için "
                    f"{best_pass.get('category')} pas seçeneği bulunuyor "
                    f"(skor {float(best_pass.get('score', 0)):.2f})."
                )
            )

            if (
                float(
                    best_pass.get(
                        "goal_progress_m",
                        0.0,
                    )
                    or 0.0
                )
                > 5.0
            ):
                attack_view.append(
                    "Bu seçenek topu rakip kaleye anlamlı biçimde ilerletiyor."
                )

        for error_type in error_types:
            if error_type == "LATE_PRESSURE":
                defense_view.append(
                    "Top sahibine ilk baskı zamanında kurulamıyor."
                )
                alternative_view.append(
                    "İlk savunmacı top sahibine daha erken yaklaşırken ikinci oyuncu pas hattını kapatabilirdi."
                )

            elif error_type == "UNMARKED_RUNNER":
                defense_view.append(
                    "İleri koşu yapan oyuncu yeterince erken takip edilmiyor."
                )

            elif error_type == "FREE_PASSING_LANE":
                defense_view.append(
                    "Savunma blokları arasında ilerleyici bir pas koridoru açık kalıyor."
                )
                alternative_view.append(
                    "Pas hattı kapatılırken alıcıya yakın savunmacı daha dar pozisyon alabilirdi."
                )

        if shot_detected:
            outcome_view.append(
                "Sekans şutla sonuçlanıyor; hücum avantajı somut bir gol tehdidine dönüşüyor."
            )
        else:
            outcome_view.append(
                "Sekansın devamı ayrıca possession/shot katmanından değerlendirilmelidir."
            )

        # Do NOT produce fake causal percentages. Two independent qualitative
        # levels are safer and more honest.
        error_strength = sum(
            severity_score(e.severity)
            for e in group
        ) / max(
            1.0,
            3.0 * len(group),
        )

        attack_signal = 0.0

        if best_pass is not None:
            attack_signal += (
                0.55
                * float(
                    best_pass.get(
                        "score",
                        0.0,
                    )
                )
            )

        if top_mark is not None:
            attack_signal += (
                0.25
                * float(
                    top_mark.get(
                        "threat_score",
                        0.0,
                    )
                )
            )

        if shot_detected:
            attack_signal += 0.30

        incidents.append(
            AnalystIncident(
                incident_id=(
                    f"INC-{serial:04d}"
                ),
                attacking_team=attacking_team,
                defending_team=defending_team,
                start_frame=start_frame,
                end_frame=end_frame,
                peak_frame=peak_frame,
                error_event_ids=tuple(
                    e.event_id
                    for e in group
                ),
                error_types=error_types,
                attack_view=tuple(attack_view),
                defense_view=tuple(defense_view),
                alternative_view=tuple(
                    alternative_view
                ),
                outcome_view=tuple(
                    outcome_view
                ),
                best_pass_receiver_id=(
                    int(
                        best_pass.get(
                            "receiver_track_id"
                        )
                    )
                    if best_pass
                    is not None
                    else None
                ),
                best_pass_score=(
                    round(
                        float(
                            best_pass.get(
                                "score",
                                0.0,
                            )
                        ),
                        5,
                    )
                    if best_pass
                    is not None
                    else None
                ),
                shot_detected=shot_detected,
                attack_merit_level=level_from_score(
                    attack_signal
                ),
                defense_vulnerability_level=level_from_score(
                    error_strength
                ),
            )
        )

    return incidents
