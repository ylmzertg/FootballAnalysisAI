
# Ball Tracking v1

Pipeline:
football-ball-detection.pt -> sliced detection -> NMS -> temporal selection ->
short-gap prediction -> JSONL.

Smoke test:
```powershell
python -m pytest tests\test_ball_tracker.py -v

python -m scripts.track_ball `
  --source "C:\Users\73645\Downloads\gsGol1.mp4" `
  --device auto `
  --max-frames 30 `
  --output "output\gsGol1_ball_v1.mp4" `
  --jsonl "output\gsGol1_ball_v1.jsonl"
```
