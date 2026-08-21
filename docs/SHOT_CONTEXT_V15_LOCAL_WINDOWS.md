# Shot Context v1.5 — Local Windows

The 250-frame diagnostic found the strongest local goal-closing burst at:

`frames 201..205`, W=5, closing=1.584 goal-widths, approach=0.50.

Longer windows dilute this short event. v1.5 therefore scans 5/8/12/15-sample
windows and relabels only the best local interval.

Expected current validation:
`TEAM_A | SHOT_FLIGHT | frames=201..205`.

Run:

```powershell
python -m pytest tests\test_shot_window.py -v

python -m scripts.estimate_possession_v15_local_shot `
  --source "input\gsGol1_goal_window.mp4" `
  --possession-jsonl "output\gsGol1_goal_possession_v12_longflight.jsonl" `
  --direction-jsonl "output\gsGol1_goal_defline_v11.jsonl" `
  --team-jsonl "output\gsGol1_goal_team_v25.jsonl" `
  --calibration-json "output\gsGol1_goal_team_v25_calibration.json" `
  --output "output\gsGol1_goal_possession_v15.mp4" `
  --jsonl "output\gsGol1_goal_possession_v15.jsonl"
```
