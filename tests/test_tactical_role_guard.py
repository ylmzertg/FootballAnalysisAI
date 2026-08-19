from core.tactical_engine import TEAM_A


def test_referee_source_class_is_hard_veto():
    track = {
        "track_id": 99,
        "class_name": "referee",
        "pitch_xy": [20.0, 20.0],
        "team_v24": {"team": TEAM_A, "role": "PLAYER"},
    }

    source_class = str(track.get("class_name", "")).strip().lower()
    role = str(track["team_v24"].get("role", "PLAYER")).upper()

    eligible = not (
        role in {"REFEREE", "OUTSIDE_PITCH"}
        or source_class == "referee"
    )

    assert eligible is False
