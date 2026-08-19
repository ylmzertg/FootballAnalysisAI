# Possession v1.2 — Event State Machine

v1.1 intentionally outputs many `LOOSE` frames because a football in transit
is not under immediate player control.

v1.2 reconstructs the *event phase* between confirmed control segments:

- `CONTROL`
- `CONTROL_GAP`
- `PASS_FLIGHT`
- `TEAM_FLIGHT`
- `CONTESTED_FLIGHT`
- `RAW_LOOSE`
- `RAW_UNKNOWN`

Rules:
- same owner before/after a short gap -> `CONTROL_GAP`;
- same team, different owner, with ball motion -> `PASS_FLIGHT`;
- different teams before/after -> `CONTESTED_FLIGHT`;
- short unresolved movement after confirmed owner -> `TEAM_FLIGHT`.

The model does **not** call a shot yet. Reliable shot classification requires
attack direction / goal context.

## Test

```powershell
python -m pytest tests\test_possession_v12_events.py -v
```

## gsGol1 250-frame run

```powershell
python -m scripts.estimate_possession_v12_events `
  --source "input\gsGol1_goal_window.mp4" `
  --possession-jsonl "output\gsGol1_goal_possession_v11_v25.jsonl" `
  --output "output\gsGol1_goal_possession_v12.mp4" `
  --jsonl "output\gsGol1_goal_possession_v12.jsonl"
```

Visual correctness matters more than minimizing `LOOSE`.
