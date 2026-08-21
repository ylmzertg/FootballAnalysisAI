# Team Identity v2.9 — Deep Kit Embedding + Balanced Clustering

## Why

V2.8 diagnostics confirmed true cluster collapse:

```text
Segments per cluster : {0: 38, 1: 3}
Resolved teams       : TEAM_A=38, TEAM_B=3
```

This is not a mapping problem. The hand-crafted colour feature space itself is
wrong for this match.

Hand-crafted colour/KMeans experiments stop here.

## V2.9

Appearance representation now uses MobileNetV3 deep embeddings.

For each physical track segment:

```text
player crop
  ├─ full kit view
  ├─ upper-shirt view
  └─ lower-kit view
        ↓
MobileNetV3 features
        ↓
temporal segment aggregation
        ↓
PCA / whitening
        ↓
balanced two-team clustering
```

No TEAM_A/B label is used while learning the deep appearance clusters.

V2.5 is used only *after* clustering to name cluster 0/1 as TEAM_A/TEAM_B.

## Why balanced clustering

The V2.8 result 38/3 is implausible for two football teams and was visibly wrong.

V2.9 uses a broad 25%-75% cluster-capacity prior. It does **not** force 50/50,
but prevents pathological collapse while still allowing tracking fragmentation.

## Model dependency

Uses the same `MobileNet_V3_Small_Weights.DEFAULT` family already used by
Team Classifier V2.5.

On a new computer, the portable setup must ensure these torchvision weights are
available/cached.

## Test

```powershell
python -m pytest tests\test_team_deep_embedding_v29.py -v
```

## gsGol1

```powershell
python -m scripts.reconcile_teams_v29_deep_embedding `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v25.jsonl" `
  --device auto `
  --output "output\gsGol1_goal_team_v29.mp4" `
  --jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --diagnostics-json "output\gsGol1_goal_team_v29_diagnostics.json"
```

Quality gate:
- no cluster collapse;
- same-kit players visually consistent;
- red/yellow and black/red form distinct groups;
- real ID-switch segments remain separate;
- referee/goalkeeper roles remain preserved.

If deep embeddings still do not separate this match, the next step is not more
clustering thresholds. It is football-specific metric learning / ReID fine-tuning.
