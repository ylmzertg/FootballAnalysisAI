
# Tactical Engine v1 — Pressure + Passing Lanes

Inputs:
- Team Classifier V2.4 JSONL
- Possession v1 JSONL

Outputs:
- Per-frame pressure metrics
- Open/blocked pass corridors
- Ranked open receivers
- Tactical overlay video
- Tactical JSONL

Smoke test:

```powershell
python -m pytest tests\test_tactical_engine.py -v

python -m scripts.tactical_engine_v1 `
  --source "C:\Users\73645\Downloads\gsGol1.mp4" `
  --team-jsonl "output\gsGol1_team_v24.jsonl" `
  --possession-jsonl "output\gsGol1_possession_v1.jsonl" `
  --output "output\gsGol1_tactical_v1.mp4" `
  --jsonl "output\gsGol1_tactical_v1.jsonl"
```

V1 deliberately does not classify a pass as "forward" or "progressive", because
the attacking direction has not yet been resolved reliably.
