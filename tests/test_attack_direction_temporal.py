from core.attack_direction_temporal import (
    TEAM_A,
    TEAM_B,
    PLUS_X,
    MINUS_X,
    UNKNOWN,
    DirectionEvidence,
    TemporalAttackDirectionResolver,
    TemporalDirectionConfig,
)


def e(team, direction, frame, confidence=0.9):
    return DirectionEvidence(
        team=team,
        direction=direction,
        confidence=confidence,
        source="trusted_goalkeeper",
        frame_index=frame,
    )


def test_temporal_consensus_locks_direction_from_sparse_frames():
    evidence = [
        e(TEAM_A, PLUS_X, i)
        for i in range(25)
    ]

    r = TemporalAttackDirectionResolver(
        TemporalDirectionConfig(
            min_evidence_frames=8,
            min_consensus_ratio=0.80,
        )
    ).resolve_pair(evidence)

    assert r[TEAM_A].direction == PLUS_X
    assert r[TEAM_B].direction == MINUS_X
    assert r[TEAM_B].source == "opponent_direction_inference"


def test_insufficient_evidence_stays_unknown():
    evidence = [
        e(TEAM_A, PLUS_X, i)
        for i in range(3)
    ]

    r = TemporalAttackDirectionResolver(
        TemporalDirectionConfig(
            min_evidence_frames=8,
        )
    ).resolve_pair(evidence)

    assert r[TEAM_A].direction == UNKNOWN
    assert r[TEAM_B].direction == UNKNOWN


def test_conflicting_pair_is_rejected():
    evidence = (
        [e(TEAM_A, PLUS_X, i) for i in range(10)]
        + [e(TEAM_B, PLUS_X, 100+i) for i in range(10)]
    )

    r = TemporalAttackDirectionResolver().resolve_pair(evidence)

    assert r[TEAM_A].direction == UNKNOWN
    assert r[TEAM_B].direction == UNKNOWN
