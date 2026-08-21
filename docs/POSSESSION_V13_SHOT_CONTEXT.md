# Possession v1.3 — Shot Context

Requires:
- Possession v1.2 event JSONL
- Attack Direction v1.1 JSONL

Adds conservative context labels:
- `ATTACKING_FLIGHT`
- `SHOT_FLIGHT`
- `GOAL_ATTEMPT`

It never declares a goal.

A candidate flight must have:
- resolved attacking direction;
- at least 3 valid ball pitch samples;
- sufficient forward progress toward the opponent goal.

Because airborne ball homography is approximate, the classifier uses multiple
samples and only labels strong directional runs.

## Test

```powershell
python -m pytest tests\test_shot_context.py -v
```

## Current 250-frame validation

```powershell
python -m scripts.estimate_possession_v13_shot `
  --source "input\gsGol1_goal_window.mp4" `
  --possession-jsonl "output\gsGol1_goal_possession_v12_longflight.jsonl" `
  --direction-jsonl "output\gsGol1_goal_defline_v11.jsonl" `
  --output "output\gsGol1_goal_possession_v13.mp4" `
  --jsonl "output\gsGol1_goal_possession_v13.jsonl"
```
