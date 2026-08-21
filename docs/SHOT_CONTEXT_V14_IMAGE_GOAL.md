# Shot Context v1.4 — Image Goal Geometry

## Why v1.3 was wrong

The diagnostic showed that the airborne ball's ground-plane `pitch_xy` stayed
around x=10–18 m while the visual play was at the attacking goal.

That is expected: a homography maps **ground-plane points**. An airborne ball is
not on that plane, so forcing its image point through image->pitch homography can
create a physically false field coordinate.

Reversing attack direction produced `SHOT_FLIGHT=6` only because those false
ground coordinates happened to move toward x=0.

## v1.4

Never use airborne-ball ground `pitch_xy` for shot direction.

For every frame:

```text
opponent goal center / posts in pitch coordinates
        ↓ inverse homography
goal mouth projected into current video frame
        +
ball image_xy
        ↓
distance to moving goal target
        ÷
projected goal-mouth width
        ↓
camera-scale-normalized goal approach
```

This keeps the ball in image space and uses calibration only to locate the goal.

## Test

```powershell
python -m pytest tests\test_shot_context_image_goal.py -v
```

## Current 250-frame validation

```powershell
python -m scripts.estimate_possession_v14_image_goal `
  --source "input\gsGol1_goal_window.mp4" `
  --possession-jsonl "output\gsGol1_goal_possession_v12_longflight.jsonl" `
  --direction-jsonl "output\gsGol1_goal_defline_v11.jsonl" `
  --team-jsonl "output\gsGol1_goal_team_v25.jsonl" `
  --calibration-json "output\gsGol1_goal_team_v25_calibration.json" `
  --output "output\gsGol1_goal_possession_v14.mp4" `
  --jsonl "output\gsGol1_goal_possession_v14.jsonl"
```

`GOAL_ATTEMPT` is not a goal declaration. Goal confirmation remains a separate
future event layer.
