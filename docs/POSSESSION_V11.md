# Possession v1.1 — Hybrid pitch + image control

Why v1.1 exists:

The 250-frame `gsGol1` validation produced:

- TEAM_A: 38
- TEAM_B: 0
- LOOSE: 182
- UNKNOWN: 30
- `no_player_in_control_radius`: 171

The main limitation was relying only on ground-plane homography distance. An
airborne ball is not on the pitch plane, so its projected `pitch_xy` can move
away from the visually controlling player or even outside the pitch.

v1.1 combines:
- ball/player pitch distance in meters;
- ball-to-foot image distance normalized by player bbox height.

Smoke test:

```powershell
python -m pytest tests\test_possession.py tests\test_possession_v11.py -v

python -m scripts.estimate_possession_v11 `
  --source "input\gsGol1_goal_window.mp4" `
  --ball-jsonl "output\gsGol1_goal_ball_v1.jsonl" `
  --team-jsonl "output\gsGol1_goal_team_v24.jsonl" `
  --calibration-json "output\gsGol1_goal_team_v24_calibration.json" `
  --output "output\gsGol1_goal_possession_v11.mp4" `
  --jsonl "output\gsGol1_goal_possession_v11.jsonl"
```

Do not tune thresholds only to maximize possession frames. Visual correctness is
the quality gate.
