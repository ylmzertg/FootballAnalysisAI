
# Attack Direction + Defensive Line v1

Auto direction evidence:
1. trusted goalkeeper near a goal line;
2. otherwise UNKNOWN.

Manual overrides:
- `--team-a-attacks left|right`
- `--team-b-attacks left|right`

The module deliberately refuses to invent a defensive line when direction is unknown.

Current gsGol1 clip has no trusted goalkeeper in the first short validation window,
so `auto` may remain UNKNOWN. Use an override only when the match direction is
visually known.

Smoke test:
```powershell
python -m pytest tests\test_attack_direction.py -v

python -m scripts.attack_direction_defline_v1 `
  --source "C:\Users\73645\Downloads\gsGol1.mp4" `
  --team-jsonl "output\gsGol1_team_v24.jsonl" `
  --output "output\gsGol1_defline_v1.mp4" `
  --jsonl "output\gsGol1_defline_v1.jsonl"
```
