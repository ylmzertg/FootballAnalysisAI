# Attack Direction + Defensive Line v1.1 — Temporal Consensus

Problem found on the 250-frame `gsGol1` clip:

- TEAM_A: 25 frames `PLUS_X`, 225 frames `UNKNOWN`
- TEAM_B: 250 frames `UNKNOWN`

The old resolver was frame-local: whenever the goalkeeper left the camera view,
direction immediately returned to UNKNOWN.

v1.1:
1. collects direct goalkeeper evidence over the whole sequence;
2. locks a team direction when enough evidence agrees;
3. infers the opponent's opposite attacking direction when only one side has
   strong direct evidence;
4. rejects conflicting same-direction pair results.

Default evidence threshold: 8 frames at >=80% consensus.

Run:

```powershell
python -m pytest tests\test_attack_direction_temporal.py -v

python -m scripts.attack_direction_defline_v11 `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v25.jsonl" `
  --output "output\gsGol1_goal_defline_v11.mp4" `
  --jsonl "output\gsGol1_goal_defline_v11.jsonl" `
  --max-frames 250
```

Expected for current validation clip:
- TEAM_A -> PLUS_X from temporal goalkeeper consensus
- TEAM_B -> MINUS_X from opponent-direction inference
