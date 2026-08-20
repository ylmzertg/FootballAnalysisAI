# Shot Context v1.6 — Contested Flight Guard

## Root cause

The real goal flight was:

```text
last control: frame 93, TEAM_B / ID16
frames 127..132: CONTESTED_FLIGHT
    source TEAM_B / ID16
    target TEAM_A / ID34
first control after: frame 137, TEAM_A / ID34
```

Independent image-goal geometry showed the strongest real movement toward the
MINUS_X goal:

```text
MINUS_X W=5 frames=127..132
closing=5.868 goal-widths
closest=2.014
```

V1.5 missed it because `CONTESTED_FLIGHT` was not a shot-candidate phase.
Instead it found a false positive in a later `RAW_LOOSE` run around 201..205.

## v1.6 rules

Shot candidates:
- `PASS_FLIGHT`
- `TEAM_FLIGHT`
- `CONTESTED_FLIGHT`

Not searched by default:
- `RAW_LOOSE`

For `CONTESTED_FLIGHT`, `source_team` controls shot direction. `target_team`
represents the next controller and must not overwrite the attacking team during
the flight.

## Run

```powershell
python -m pytest tests\test_shot_context_v16_guard.py tests\test_shot_window.py -v

python -m scripts.estimate_possession_v16_contested_shot `
  --source "input\gsGol1_goal_window.mp4" `
  --possession-jsonl "output\gsGol1_goal_possession_v12_longflight.jsonl" `
  --direction-jsonl "output\gsGol1_goal_defline_v11.jsonl" `
  --team-jsonl "output\gsGol1_goal_team_v25.jsonl" `
  --calibration-json "output\gsGol1_goal_team_v25_calibration.json" `
  --output "output\gsGol1_goal_possession_v16.mp4" `
  --jsonl "output\gsGol1_goal_possession_v16.jsonl"
```

Expected critical candidate:
- TEAM_B
- source phase `CONTESTED_FLIGHT`
- frames around 127..132
- MINUS_X
- strong local goal closing

The later RAW_LOOSE false positive around 201..205 should disappear.
