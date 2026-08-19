# Validation Status — 2026-08-18

## Portable Windows CPU baseline

- Main: Python 3.10.11 / torch 1.13.1+cpu / OpenCV 4.11.0 / NumPy 1.26.4
- Ultralytics: OK
- Supervision ByteTrack 0.25.0: OK
- PnLCalib adapter/runtime: OK
- TVCalib runtime/checkpoint/submodule: OK
- Portable runtime + Team Classifier V2.4 tests: 7 passed

## gsGol1 Tracking

- Frames: 150
- Tracked detections: 2461
- Unique IDs: 35
- Average FPS: 1.15
- Average tracked detections/frame: ~16.4

## gsGol1 Team Classifier V2.4 + PnL

- Frames: 60
- TEAM_A: 559
- TEAM_B: 563
- UNKNOWN: 61
- Referee: 45
  - ID 18: 26
  - ID 5: 19
- Goalkeeper: 0
- PnL exact: 5
- PnL nearest: 55
- Missing geometry: 0
- Classifier ready: True
- Canonical aliases: none in this 60-frame window
- Outside-pitch tracks: none in this 60-frame window

## Architecture

- PnLCalib = PRIMARY
- TVCalib = FALLBACK
- YOLO detection
- ByteTrack
- Team Classifier V2.4
- Canonical Identity
- Referee correction
- Goalkeeper temporal consensus

## Next

1. Visual validation of gsGol1 Team A/B and referee IDs 18/5.
2. Make remaining V2.4 harness paths portable.
3. Calibration fusion / Radar v4 when resumed.
4. Ball detection/tracking.
5. Possession and Tactical Engine.
