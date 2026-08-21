from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "tactical_engine_v1.py"

OLD = '        assignment = tr.get("team_v24") or {}\n        team = str(assignment.get("team", "UNKNOWN"))\n        role = str(assignment.get("role", "PLAYER"))\n\n        if team not in {TEAM_A, TEAM_B}:\n            continue\n\n        if role in {"REFEREE", "OUTSIDE_PITCH"}:\n            continue\n'
NEW = '        assignment = tr.get("team_v24") or {}\n        team = str(assignment.get("team", "UNKNOWN"))\n        role = str(assignment.get("role", "PLAYER")).upper()\n\n        # Hard role guard: detector/source role has veto power over a noisy\n        # temporal team-classification assignment. A detector-labelled referee\n        # must never become a passing option or team-shape player.\n        source_class = str(tr.get("class_name", "")).strip().lower()\n        source_role = str(tr.get("role_hint", "")).strip().upper()\n\n        if team not in {TEAM_A, TEAM_B}:\n            continue\n\n        if (\n            role in {"REFEREE", "OUTSIDE_PITCH"}\n            or source_class == "referee"\n            or source_role == "REFEREE"\n        ):\n            continue\n'

def main():
    text = TARGET.read_text(encoding="utf-8")
    if OLD not in text:
        raise RuntimeError("Expected parse_players block not found; target may have changed.")
    text = text.replace(OLD, NEW, 1)
    compile(text, str(TARGET), "exec")
    TARGET.write_text(text, encoding="utf-8")
    print("Tactical Engine v1.1 role guard applied.")
    print("Referee veto sources: team_v24.role, class_name, role_hint")

if __name__ == "__main__":
    main()
