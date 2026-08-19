from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_CORE = ROOT / "core" / "team_classifier_v24.py"
DST_CORE = ROOT / "core" / "team_classifier_v25.py"
SRC_HARNESS = ROOT / "scripts" / "classify_teams_v24_pnl_exact.py"
DST_HARNESS = ROOT / "scripts" / "classify_teams_v25_pnl_exact.py"

TEMPORAL_BLOCK = '    def _temporal_team(\n        self,\n        state: _TrackState,\n        raw_team: str,\n        raw_confidence: float,\n        frame_index: int,\n    ) -> Tuple[str, float]:\n        if not state.votes:\n            return raw_team, raw_confidence\n\n        scores = {TEAM_A: 0.0, TEAM_B: 0.0}\n        total = 0.0\n\n        for vote in reversed(state.votes):\n            age = max(0, frame_index - vote.frame_index)\n            weight = (\n                self.config.temporal_decay ** age\n            ) * max(0.05, vote.confidence)\n            scores[vote.team] += weight\n            total += weight\n\n        if total <= 0:\n            return raw_team, raw_confidence\n\n        if state.stable_team in (TEAM_A, TEAM_B):\n            locked_score = float(\n                scores.get(state.stable_team, 0.0) / total\n            )\n            confidence = max(\n                state.stable_team_confidence,\n                locked_score,\n            )\n            state.stable_team_confidence = min(0.999, confidence)\n            return state.stable_team, state.stable_team_confidence\n\n        best_team = max(scores, key=scores.get)\n        best_score = scores[best_team]\n        other_team = TEAM_B if best_team == TEAM_A else TEAM_A\n        second_score = scores[other_team]\n\n        best_ratio = float(best_score / total)\n        second_ratio = float(second_score / total)\n        ratio_margin = best_ratio - second_ratio\n\n        if (\n            len(state.votes) >= self.config.team_lock_min_votes\n            and best_ratio >= self.config.team_lock_min_ratio\n            and ratio_margin >= self.config.team_lock_min_margin\n        ):\n            state.stable_team = best_team\n            state.stable_team_confidence = best_ratio\n            state.stable_team_locked_frame = frame_index\n            return best_team, best_ratio\n\n        if (\n            len(state.votes) < self.config.min_votes_before_stable\n            and raw_team != UNKNOWN\n        ):\n            return raw_team, max(\n                raw_confidence,\n                best_ratio * 0.8,\n            )\n\n        return best_team, best_ratio\n'
SWITCH_BLOCK = '    def _handle_possible_switch(\n        self,\n        state: _TrackState,\n        strength: float,\n    ) -> None:\n        if strength >= 1.0:\n            state.stable_team = None\n            state.stable_team_confidence = 0.0\n            state.stable_team_locked_frame = -1\n\n        if strength >= 1.15:\n            state.votes.clear()\n            state.embedding_proto = None\n            state.colour_proto = None\n            state.feature_proto = None\n            state.referee_votes = 0\n            state.referee_hint_frames = 0\n            state.non_referee_hint_frames = 0\n            state.referee_hint_streak = 0\n            state.referee_trusted = False\n            state.goalkeeper_votes = 0\n            state.goalkeeper_hint_frames = 0\n            state.non_goalkeeper_hint_frames = 0\n            state.goalkeeper_trusted = False\n\n        elif len(state.votes) > self.config.soft_reset_keep_votes:\n            newest = list(\n                state.votes\n            )[-self.config.soft_reset_keep_votes :]\n            state.votes.clear()\n            state.votes.extend(newest)\n'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def replace_function(text: str, start_marker: str, end_marker: str, new_block: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"start marker not found: {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"end marker not found: {end_marker!r}")
    return text[:start] + new_block.rstrip() + "\n\n" + text[end:]


def build_core():
    text = SRC_CORE.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "    min_votes_before_stable: int = 3\n",
        (
            "    min_votes_before_stable: int = 3\n"
            "\n"
            "    # V2.5 stable-team lock. Once a track accumulates strong team\n"
            "    # consensus, its team cannot flicker frame-to-frame.\n"
            "    team_lock_min_votes: int = 5\n"
            "    team_lock_min_ratio: float = 0.72\n"
            "    team_lock_min_margin: float = 0.18\n"
        ),
        "config team lock",
    )

    text = replace_once(
        text,
        "    goalkeeper_trusted: bool = False\n",
        (
            "    goalkeeper_trusted: bool = False\n"
            "    stable_team: Optional[str] = None\n"
            "    stable_team_confidence: float = 0.0\n"
            "    stable_team_locked_frame: int = -1\n"
        ),
        "track stable team state",
    )

    text = replace_function(
        text,
        "    def _temporal_team(\n",
        "    def _detect_id_switch(\n",
        TEMPORAL_BLOCK,
    )

    text = replace_function(
        text,
        "    def _handle_possible_switch(",
        "    def _update_track_appearance(\n",
        SWITCH_BLOCK,
    )

    text = replace_once(
        text,
        '            "has_referee_prototype": self.referee_prototype is not None,\n',
        (
            '            "has_referee_prototype": self.referee_prototype is not None,\n'
            '            "stable_team_locks": sum(\n'
            '                1 for s in self.track_states.values()\n'
            '                if s.stable_team in (TEAM_A, TEAM_B)\n'
            '            ),\n'
        ),
        "debug stable locks",
    )

    compile(text, str(DST_CORE), "exec")
    DST_CORE.write_text(text, encoding="utf-8")


def build_harness():
    text = SRC_HARNESS.read_text(encoding="utf-8")

    text = text.replace(
        "from core.team_classifier_v24 import (",
        "from core.team_classifier_v25 import (",
        1,
    )
    text = text.replace("Team Classifier V2.4", "Team Classifier V2.5")
    text = text.replace("V2.4", "V2.5")
    text = text.replace("v24", "v25")
    text = text.replace("V24", "V25")

    marker = '                    enriched["team_v25"] = assignment_to_dict(a)\n'
    if marker in text:
        text = text.replace(
            marker,
            (
                '                    enriched["team_v25"] = assignment_to_dict(a)\n'
                '                    enriched["team_v24"] = enriched["team_v25"]\n'
            ),
            1,
        )

    compile(text, str(DST_HARNESS), "exec")
    DST_HARNESS.write_text(text, encoding="utf-8")


def main():
    if not SRC_CORE.exists():
        raise FileNotFoundError(SRC_CORE)
    if not SRC_HARNESS.exists():
        raise FileNotFoundError(SRC_HARNESS)

    build_core()
    build_harness()

    print("=" * 76)
    print("Team Classifier V2.5 stable-team lock generated")
    print(f"Core    : {DST_CORE}")
    print(f"Harness : {DST_HARNESS}")
    print("V2.4 files preserved unchanged.")
    print("=" * 76)


if __name__ == "__main__":
    main()
