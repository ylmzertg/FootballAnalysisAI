import numpy as np

from core.track_identity import (
    CanonicalIdentityConfig,
    build_canonical_alias_map,
    collapse_frame_records,
)
from core.team_classifier_v24 import (
    DetectionObservation,
    GOALKEEPER,
    PLAYER,
    REFEREE,
    TeamClassifierV2,
    TeamClassifierV2Config,
)


def _rec(tid, bbox, pitch, cls="player", conf=0.9):
    return {
        "track_id": tid,
        "bbox_xyxy": bbox,
        "pitch_xy": pitch,
        "class_name": cls,
        "confidence": conf,
    }


def test_duplicate_overlap_tracks_merge_but_nearby_distinct_do_not():
    frames = {}
    for fi in range(7):
        frames[fi] = [
            _rec(11, (100, 100, 130, 180), (21.0 + fi * .02, 37.0)),
            _rec(28, (100.5, 100.2, 130.5, 180.1), (21.03 + fi * .02, 37.02), "referee", .85),
            _rec(18, (150, 100, 180, 180), (23.4, 37.0), "referee", .8),
        ]

    aliases, pairs = build_canonical_alias_map(
        frames,
        CanonicalIdentityConfig(
            min_overlap_frames=2,
            max_median_pitch_distance_m=.75,
            min_median_bbox_iou=.80,
        ),
    )

    assert aliases[28] == 11
    assert aliases[11] == 11
    assert aliases[18] == 18
    assert any({p.track_a, p.track_b} == {11, 28} for p in pairs)

    collapsed = collapse_frame_records(frames[0], aliases)
    assert sorted(r["track_id"] for r in collapsed) == [11, 18]
    merged = next(r for r in collapsed if r["track_id"] == 11)
    assert merged["raw_track_ids"] == [11, 28]


def _base_scene():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (45, 115, 45)
    obs = []
    tid = 1

    for team_idx in range(2):
        for row in range(5):
            x = 60 + team_idx * 320 + (row % 2) * 35
            y = 45 + row * 55
            colour = (0, 0, 220) if team_idx == 0 else (220, 40, 20)
            frame[y:y + 44, x:x + 22] = colour
            obs.append(
                DetectionObservation(
                    tid,
                    (x, y, x + 22, y + 44),
                    confidence=.95,
                    pitch_xy=(8 + team_idx * 82, 10 + row * 10),
                    role_hint=PLAYER,
                )
            )
            tid += 1
    return frame, obs


def _classifier():
    return TeamClassifierV2(
        TeamClassifierV2Config(
            use_deep_embedding=False,
            bootstrap_min_samples=18,
            bootstrap_samples_per_track=3,
            embedding_stride=1,
            goalkeeper_hint_min_frames=4,
            goalkeeper_hint_min_total_frames=8,
            goalkeeper_hint_min_ratio=.65,
            referee_on_pitch_hint_min_frames=3,
            referee_on_pitch_min_total_frames=10,
            referee_on_pitch_min_novelty=.10,
            referee_hint_min_novelty=.10,
            role_outlier_colour_similarity=.75,
        )
    )


def _bootstrap(clf):
    for fi in range(4):
        frame, obs = _base_scene()
        clf.classify_frame(frame, obs, fi)
    assert clf.is_ready


def _append_role(frame, obs, tid, role, pitch, colour=(0, 230, 230)):
    frame[170:220, 290:318] = colour
    obs.append(
        DetectionObservation(
            tid,
            (290, 170, 318, 220),
            confidence=.98,
            pitch_xy=pitch,
            role_hint=role,
        )
    )


def test_noisy_goalkeeper_hints_do_not_hard_label_defender():
    clf = _classifier()
    _bootstrap(clf)
    roles = []

    for offset, fi in enumerate(range(4, 14)):
        frame, obs = _base_scene()
        hint = GOALKEEPER if offset in {1, 7} else PLAYER
        _append_role(frame, obs, 100, hint, (2.0, 34.0))
        assignments = clf.classify_frame(frame, obs, fi)
        roles.append(next(a for a in assignments if a.track_id == 100).role)

    state = clf.track_states[100]
    assert state.goalkeeper_hint_frames == 2
    assert not state.goalkeeper_trusted
    assert GOALKEEPER not in roles


def test_sustained_goalkeeper_consensus_becomes_trusted():
    clf = _classifier()
    _bootstrap(clf)
    roles = []

    for offset, fi in enumerate(range(4, 14)):
        frame, obs = _base_scene()
        hint = GOALKEEPER if offset < 8 else PLAYER
        _append_role(frame, obs, 101, hint, (2.0, 34.0))
        assignments = clf.classify_frame(frame, obs, fi)
        roles.append(next(a for a in assignments if a.track_id == 101).role)

    state = clf.track_states[101]
    assert state.goalkeeper_hint_frames == 8
    assert state.goalkeeper_trusted
    assert roles[-1] == GOALKEEPER


def test_off_pitch_referee_hints_never_become_match_referee():
    clf = _classifier()
    _bootstrap(clf)
    roles = []

    for fi in range(4, 18):
        frame, obs = _base_scene()
        _append_role(frame, obs, 200, REFEREE, (109.0, 34.0))
        assignments = clf.classify_frame(frame, obs, fi)
        roles.append(next(a for a in assignments if a.track_id == 200).role)

    assert REFEREE not in roles
    assert not clf.track_states[200].referee_trusted


def test_intermittent_on_pitch_referee_hints_can_become_trusted():
    clf = _classifier()
    _bootstrap(clf)
    roles = []
    ref_frames = {4, 8, 13}

    for fi in range(4, 14):
        frame, obs = _base_scene()
        hint = REFEREE if fi in ref_frames else PLAYER
        _append_role(frame, obs, 201, hint, (52.5, 34.0))
        assignments = clf.classify_frame(frame, obs, fi)
        roles.append(next(a for a in assignments if a.track_id == 201).role)

    state = clf.track_states[201]
    assert state.referee_on_pitch_hint_frames == 3
    assert state.referee_trusted
    assert roles[-1] == REFEREE
