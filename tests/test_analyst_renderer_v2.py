from core.analyst_renderer_v2 import (
    IncidentCandidate,
    select_incidents,
)


def inc(
    iid,
    start,
    end,
    peak,
    *,
    shot=False,
    attack="MEDIUM",
    defense="MEDIUM",
    team="TEAM_A",
):
    return IncidentCandidate(
        incident_id=iid,
        attacking_team=team,
        start_frame=start,
        end_frame=end,
        peak_frame=peak,
        attack_merit_level=attack,
        defense_vulnerability_level=defense,
        shot_detected=shot,
        error_types=("LATE_PRESSURE",),
    )


def test_shot_incident_is_prioritized():
    selected = select_incidents(
        [
            inc("A", 0, 50, 20),
            inc(
                "B",
                60,
                100,
                80,
                shot=True,
                attack="HIGH",
                defense="HIGH",
            ),
        ]
    )

    assert any(
        x.incident_id == "B"
        for x in selected
    )


def test_heavily_overlapping_incident_is_suppressed():
    selected = select_incidents(
        [
            inc(
                "A",
                100,
                150,
                120,
                attack="HIGH",
                defense="HIGH",
            ),
            inc(
                "B",
                110,
                148,
                130,
                attack="MEDIUM",
                defense="MEDIUM",
            ),
        ]
    )

    assert len(selected) == 1
    assert selected[0].incident_id == "A"


def test_non_overlapping_incidents_are_kept():
    selected = select_incidents(
        [
            inc("A", 0, 30, 15),
            inc("B", 80, 110, 95),
        ]
    )

    assert len(selected) == 2
