# FootballAnalysisAI

Modular football video analysis platform.

## Goal

Convert broadcast football video into a normalized **MatchState** representation and use it for:

- pitch/camera calibration
- player detection and tracking
- team / role / jersey identification
- ball tracking
- 2D tactical map
- tactical event analysis
- telestration
- YouTube-ready analysis video

## Architecture

External research/open-source engines are treated as **replaceable adapters**.

Examples:

- SoccerNet / TrackLab
- NBJW calibration
- PnLCalib
- TVCalib
- YOLO / RF-DETR
- BoT-SORT / StrongSORT / ByteTrack / OC-SORT

Every engine must normalize its output to the project `MatchState` schema.

## Current milestone

1. Define common MatchState schema
2. Define adapter interfaces
3. Build benchmark framework
4. Add SoccerNet adapter
5. Add calibration benchmark
