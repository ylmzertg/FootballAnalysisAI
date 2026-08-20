CONTESTED_FLIGHT = "CONTESTED_FLIGHT"
PASS_FLIGHT = "PASS_FLIGHT"
RAW_LOOSE = "RAW_LOOSE"


def candidate_phases():
    return {
        PASS_FLIGHT,
        "TEAM_FLIGHT",
        CONTESTED_FLIGHT,
    }


def test_contested_flight_is_shot_candidate():
    assert CONTESTED_FLIGHT in candidate_phases()


def test_raw_loose_is_not_shot_candidate():
    assert RAW_LOOSE not in candidate_phases()


def test_contested_source_team_is_authoritative():
    row = {
        "phase": CONTESTED_FLIGHT,
        "source_team": "TEAM_B",
        "target_team": "TEAM_A",
        "team_state": "LOOSE",
    }

    source = row.get("source_team")
    assert source == "TEAM_B"
