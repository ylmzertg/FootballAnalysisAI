# Goalkeeper Team Resolver v1

Why:

The first attack-direction pass trusted the goalkeeper's own TEAM_A/TEAM_B
classification. On `gsGol1`, reversing the directions manually produced six
`SHOT_FLIGHT` frames, strongly indicating that the goalkeeper's team label was
wrong even though its role was trusted.

Resolver v1 ignores the goalkeeper's own team label when determining team
membership. It uses calibrated nearby outfield players as weighted evidence.

Pipeline:

```text
trusted GK role + pitch_xy
        +
nearby outfield TEAM_A/B labels
        ↓
weighted local team consensus
        ↓
temporal goalkeeper-team consensus
        ↓
GK team + own-goal side
        ↓
attack direction
```

Run:

```powershell
python -m pytest tests\test_goalkeeper_team_resolver.py -v

python -m scripts.resolve_goalkeeper_team_v1 `
  --team-jsonl "output\gsGol1_goal_team_v25.jsonl" `
  --output-json "output\gsGol1_goal_gk_team_v1.json"
```

For a goalkeeper near x=105:
- if resolved team is TEAM_B, TEAM_B attacks MINUS_X;
- the opponent TEAM_A attacks PLUS_X.

For a goalkeeper near x=0:
- its resolved team attacks PLUS_X.
