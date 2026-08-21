from core.error_detection_v1 import ErrorEvent
from core.error_event_sequence_v11 import sequence_error_events


def e(frame, kind, primary=1, metric=5.0):
    return ErrorEvent(
        frame_index=frame,
        error_type=kind,
        attacking_team="TEAM_B",
        defending_team="TEAM_A",
        severity="HIGH",
        primary_track_id=primary,
        secondary_track_id=9,
        metric_value=metric,
        explanation="test",
        evidence={"frame": frame},
    )


def test_repeated_runner_frames_become_one_event():
    frame_events = {
        10: [e(10, "UNMARKED_RUNNER", 7, 5.0)],
        11: [e(11, "UNMARKED_RUNNER", 7, 6.0)],
        12: [e(12, "UNMARKED_RUNNER", 7, 7.0)],
        13: [e(13, "UNMARKED_RUNNER", 7, 6.5)],
    }

    events = sequence_error_events(frame_events)

    assert len(events) == 1
    assert events[0].start_frame == 10
    assert events[0].end_frame == 13
    assert events[0].peak_frame == 12


def test_single_frame_runner_noise_is_removed():
    events = sequence_error_events(
        {
            20: [
                e(
                    20,
                    "UNMARKED_RUNNER",
                    4,
                )
            ]
        }
    )

    assert events == []


def test_distinct_tracks_remain_distinct_events():
    frame_events = {
        30: [
            e(30, "FREE_PASSING_LANE", 5),
            e(30, "FREE_PASSING_LANE", 6),
        ],
        31: [
            e(31, "FREE_PASSING_LANE", 5),
            e(31, "FREE_PASSING_LANE", 6),
        ],
        32: [
            e(32, "FREE_PASSING_LANE", 5),
            e(32, "FREE_PASSING_LANE", 6),
        ],
    }

    events = sequence_error_events(frame_events)

    assert len(events) == 2
