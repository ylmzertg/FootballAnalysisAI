from core.analyst_incident_v1 import (
    ErrorTimelineItem,
    build_incidents,
)


def err(
    event_id,
    kind,
    start,
    end,
    peak,
):
    return ErrorTimelineItem(
        event_id=event_id,
        error_type=kind,
        attacking_team="TEAM_B",
        defending_team="TEAM_A",
        start_frame=start,
        end_frame=end,
        peak_frame=peak,
        severity="HIGH",
        primary_track_id=7,
        secondary_track_id=4,
        evidence={},
    )


def test_overlapping_errors_form_one_incident():
    incidents = build_incidents(
        errors=[
            err(
                "ERR-1",
                "UNMARKED_RUNNER",
                90,
                93,
                92,
            ),
            err(
                "ERR-2",
                "FREE_PASSING_LANE",
                90,
                93,
                91,
            ),
        ],
        pass_options_by_frame={
            92: [
                {
                    "receiver_track_id": 7,
                    "category": "BEST",
                    "score": 0.82,
                    "goal_progress_m": 12.0,
                }
            ]
        },
        marking_by_frame={
            92: [
                {
                    "attacker_track_id": 7,
                    "nearest_defender_track_id": 3,
                    "nearest_defender_distance_m": 6.5,
                    "marking_state": "UNMARKED",
                    "dangerous": True,
                    "threat_score": 0.8,
                }
            ]
        },
        shot_frames={127, 128},
    )

    assert len(incidents) == 1
    assert (
        "UNMARKED_RUNNER"
        in incidents[0].error_types
    )
    assert incidents[0].best_pass_receiver_id == 7
    assert incidents[0].shot_detected is True


def test_attack_and_defense_views_both_exist():
    incidents = build_incidents(
        errors=[
            err(
                "ERR-1",
                "LATE_PRESSURE",
                10,
                15,
                13,
            )
        ],
        pass_options_by_frame={},
        marking_by_frame={},
        shot_frames=set(),
    )

    inc = incidents[0]

    assert len(inc.defense_view) > 0
    assert len(inc.alternative_view) > 0
