# Tactical / Shape v1.1 — Identity-Aware Hotfix

## Root cause

The original tactical and shape harnesses still read:

```python
tr.get("team_v24")
```

even when a newer Team Identity JSONL is supplied.

Therefore a command such as:

```powershell
--team-jsonl output\gsGol1_goal_team_v29.jsonl
```

did **not** actually make Passing Lanes / Team Shape use V2.9 identities.

This is especially important before `Pass Options Ranking`, because an open lane
to a player assigned to the wrong team corrupts the recommendation.

## New priority order

```text
team_v29
team_v28
team_v27
team_v26
team_v25
team_v24
```

## Build

```powershell
python scripts\build_tactical_v11_identity.py
python scripts\build_shape_space_v11_identity.py
```

## Test

```powershell
python -m pytest tests\test_identity_aware_assignment.py -v
```

## Re-run Tactical Engine with V2.9

```powershell
python -m scripts.tactical_engine_v11_identity `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --possession-jsonl "output\gsGol1_goal_possession_v11_v25.jsonl" `
  --output "output\gsGol1_goal_tactical_v29_v11.mp4" `
  --jsonl "output\gsGol1_goal_tactical_v29_v11.jsonl"
```

## Re-run Shape/Space with V2.9

```powershell
python -m scripts.team_shape_space_v11_identity `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --possession-jsonl "output\gsGol1_goal_possession_v11_v25.jsonl" `
  --output "output\gsGol1_goal_shape_v29_v11.mp4" `
  --jsonl "output\gsGol1_goal_shape_v29_v11.jsonl"
```

Then Error Detection v1.1 must be rerun with the corrected tactical JSONL before
Pass Options Ranking is validated.
