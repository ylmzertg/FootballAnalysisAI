def choose_team_assignment(track):
    return (
        track.get("team_v29")
        or track.get("team_v28")
        or track.get("team_v27")
        or track.get("team_v26")
        or track.get("team_v25")
        or track.get("team_v24")
        or {}
    )


def test_v29_takes_precedence():
    track = {
        "team_v24": {"team": "TEAM_A"},
        "team_v25": {"team": "TEAM_A"},
        "team_v29": {"team": "TEAM_B"},
    }

    assert choose_team_assignment(track)["team"] == "TEAM_B"


def test_fallback_to_v25():
    track = {
        "team_v25": {"team": "TEAM_B"},
        "team_v24": {"team": "TEAM_A"},
    }

    assert choose_team_assignment(track)["team"] == "TEAM_B"


def test_fallback_to_v24():
    track = {
        "team_v24": {"team": "TEAM_A"},
    }

    assert choose_team_assignment(track)["team"] == "TEAM_A"
