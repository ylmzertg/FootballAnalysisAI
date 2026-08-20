from core.goalkeeper_team_resolver import (
    TEAM_A,
    TEAM_B,
    ResolverPlayer,
    GoalkeeperTeamResolver,
    GoalkeeperTeamResolverConfig,
)


def p(tid, team, role, x, y):
    return ResolverPlayer(
        track_id=tid,
        team=team,
        role=role,
        pitch_xy=(x, y),
    )


def test_goalkeeper_team_comes_from_nearby_defenders_not_own_label():
    r = GoalkeeperTeamResolver(
        GoalkeeperTeamResolverConfig(
            min_neighbors=2,
            min_weight_margin_ratio=0.10,
        )
    )

    # GK's own team label is intentionally wrong.
    gk = p(29, TEAM_A, "GOALKEEPER", 100, 34)

    players = [
        gk,
        p(1, TEAM_B, "PLAYER", 95, 30),
        p(2, TEAM_B, "PLAYER", 94, 38),
        p(3, TEAM_A, "PLAYER", 75, 20),
    ]

    e = r.frame_evidence(10, gk, players)

    assert e is not None
    assert e.assigned_team == TEAM_B


def test_temporal_consensus_resolves_goalkeeper_team():
    r = GoalkeeperTeamResolver(
        GoalkeeperTeamResolverConfig(
            min_evidence_frames=3,
            min_consensus_ratio=0.65,
        )
    )

    evidence = []

    for frame in range(4):
        gk = p(29, TEAM_A, "GOALKEEPER", 100, 34)
        players = [
            gk,
            p(1, TEAM_B, "PLAYER", 95, 30),
            p(2, TEAM_B, "PLAYER", 94, 38),
        ]
        e = r.frame_evidence(frame, gk, players)
        assert e is not None
        evidence.append(e)

    c = r.consensus(evidence)[29]

    assert c.resolved_team == TEAM_B
    assert c.median_goalkeeper_x > 52.5
