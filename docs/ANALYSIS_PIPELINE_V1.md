# Analysis Pipeline v1

End-to-end orchestrator for the validated FootballAnalysisAI pipeline.

## Step order

```text
source video
  -> Player Tracking
  -> Team Classifier V2.5 + PnL
  -> Ball Tracking v1
  -> Possession v1.1 Hybrid
  -> Possession v1.2 Event State Machine
  -> Attack Direction v1.1
  -> Tactical Engine v1.1
  -> Team Shape + Space v1
  -> Shot Context v1.6
  -> merged analysis_v1.jsonl
  -> combined analysis_v1.mp4
```

## Resume behavior

Existing non-empty step outputs are skipped automatically. This matters on CPU,
where sliced ball inference is expensive.

Use `--force` only when you deliberately want to rebuild every selected step.

## Current gsGol1 validation

If the individual 250-frame outputs already exist under older filenames, the
first pipeline run will normally recompute them into the run-scoped directory.
For future clips, use the pipeline directly from the start.

```powershell
python -m pytest tests\test_analysis_pipeline.py -v

python -m scripts.run_analysis_pipeline_v1 `
  --source "input\gsGol1_goal_window.mp4" `
  --run-name "gsGol1_goal" `
  --device auto `
  --max-frames 250 `
  --calibration-stride 15 `
  --max-calibration-gap 8 `
  --max-bridge-gap 75 `
  --max-unresolved-flight 50
```

Outputs:

```text
output/pipeline_v1/gsGol1_goal/
  01_tracking.*
  02_team_v25.*
  03_ball_v1.*
  04_possession_v11.*
  05_possession_v12.*
  06_direction_v11.*
  07_tactical_v11.*
  08_shape_space_v1.*
  09_shot_v16.*
  analysis_v1.jsonl
  analysis_v1.mp4
  manifest.json
```

## Useful controls

Dry-run:

```powershell
python -m scripts.run_analysis_pipeline_v1 `
  --source "input\gsGol1_goal_window.mp4" `
  --run-name "gsGol1_goal" `
  --max-frames 250 `
  --dry-run
```

Resume from a known stage:

```powershell
python -m scripts.run_analysis_pipeline_v1 `
  --source "input\gsGol1_goal_window.mp4" `
  --run-name "gsGol1_goal" `
  --max-frames 250 `
  --start-at possession_v11
```

Stop after calibration/team classification:

```powershell
python -m scripts.run_analysis_pipeline_v1 `
  --source "input\gsGol1_goal_window.mp4" `
  --run-name "gsGol1_goal" `
  --max-frames 250 `
  --stop-after team_v25
```
