# Team Classifier V2.5 — Stable Team Lock

The 250-frame `gsGol1` validation exposed an upstream invariant violation:
the same track ID changed `TEAM_A <-> TEAM_B`.

Observed in the rendered possession video:
- owner ID 20 was TEAM_A around frame 10;
- the same owner ID 20 was TEAM_B around frames 20–40.

V2.5 preserves V2.4 and generates new files with stable team locking.

## Build

```powershell
python scripts\build_team_classifier_v25.py
```

## Test

```powershell
python -m pytest tests\test_team_classifier_v25_lock.py -v
```

## Diagnose old output

```powershell
python scripts\diagnose_team_flips.py output\gsGol1_goal_team_v24.jsonl
```

## Re-run 250-frame classifier

```powershell
python -m scripts.classify_teams_v25_pnl_exact `
  --source "input\gsGol1_goal_window.mp4" `
  --tracking "output\gsGol1_goal_tracking.jsonl" `
  --device auto `
  --max-frames 250 `
  --calibration-stride 15 `
  --max-calibration-gap 8 `
  --output "output\gsGol1_goal_team_v25.mp4" `
  --jsonl "output\gsGol1_goal_team_v25.jsonl" `
  --calibration-json "output\gsGol1_goal_team_v25_calibration.json" `
  --calibration-frames-dir "output\gsGol1_goal_v25_pnl_frames"
```

Then re-run the transition diagnostic.
