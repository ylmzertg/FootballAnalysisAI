
# Possession v1

Inputs:

- Ball Tracking v1 JSONL
- Team Classifier V2.4 JSONL
- V2.4 PnL calibration JSON

Pipeline:

```text
ball image_xy
  -> PnL homography
  -> ball pitch_xy
  -> nearest eligible player in pitch coordinates
  -> acquire/release distance thresholds
  -> temporal confirmation + hysteresis
  -> TEAM_A / TEAM_B / LOOSE / UNKNOWN
```

Smoke test with the current `gsGol1` outputs:

```powershell
python -m pytest tests\test_possession.py -v

python -m scripts.estimate_possession_v1 `
  --source "C:\Users\73645\Downloads\gsGol1.mp4" `
  --ball-jsonl "output\gsGol1_ball_v1.jsonl" `
  --team-jsonl "output\gsGol1_team_v24.jsonl" `
  --calibration-json "output\gsGol1_team_v24_calibration.json" `
  --output "output\gsGol1_possession_v1.mp4" `
  --jsonl "output\gsGol1_possession_v1.jsonl"
```

The current ball smoke test contains 30 frames, so the common frame range will
normally be 30 frames even if the team file contains more.
