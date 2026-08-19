# Team Shape + Space Detection v1

Metrics per team:
- centroid
- width
- depth
- compactness
- bounding-box area

Space candidates:
- opponent clearance
- teammate support distance
- possessor reach distance
- neutral geometric score

Smoke test:

```powershell
python -m pytest tests\test_team_shape.py -v

python -m scripts.team_shape_space_v1 `
  --source "C:\Users\73645\Downloads\gsGol1.mp4" `
  --team-jsonl "output\gsGol1_team_v24.jsonl" `
  --possession-jsonl "output\gsGol1_possession_v1.jsonl" `
  --output "output\gsGol1_shape_space_v1.mp4" `
  --jsonl "output\gsGol1_shape_space_v1.jsonl"
```

Defensive line is deferred until attacking direction is resolved reliably.
