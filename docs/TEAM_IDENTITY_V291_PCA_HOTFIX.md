# Team Identity v2.9.1 — PCA Hotfix

Fixes the failing regression test in v2.9.

Root cause:
`PCA(whiten=True)` amplified low-variance noise dimensions in the synthetic and
deep-appearance embedding space. Members of the same obvious group could then
be split across both balanced clusters.

Changes:
- PCA whitening disabled.
- PCA dimensionality capped conservatively for small segment sets.
- farthest-pair initialization explicitly L2-normalized.
- clustering initialization uses `labels=-1` to avoid accidental first-iteration
  equality edge cases.

After extracting over the repository:

```powershell
python -m pytest tests\test_team_deep_embedding_v29.py -v
```

If all tests pass, run the existing
`scripts.reconcile_teams_v29_deep_embedding` command unchanged.
