from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "tactical_engine_v1.py"
TARGET = ROOT / "scripts" / "tactical_engine_v11_identity.py"


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}"
        )
    return text.replace(old, new, 1)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    text = SOURCE.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '        assignment = tr.get("team_v24") or {}\n',
        (
            '        assignment = (\n'
            '            tr.get("team_v29")\n'
            '            or tr.get("team_v28")\n'
            '            or tr.get("team_v27")\n'
            '            or tr.get("team_v26")\n'
            '            or tr.get("team_v25")\n'
            '            or tr.get("team_v24")\n'
            '            or {}\n'
            '        )\n'
        ),
        "team assignment",
    )

    text = text.replace(
        "Tactical Engine v1",
        "Tactical Engine v1.1 IDENTITY-AWARE",
    )

    compile(text, str(TARGET), "exec")
    TARGET.write_text(text, encoding="utf-8")

    print(f"Generated: {TARGET}")


if __name__ == "__main__":
    main()
