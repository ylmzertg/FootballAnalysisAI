# Ball Tracking v2 Experiments

| Version | Main idea | Result |
|---|---|---|
| v1 | Full sliced detection + temporal prediction | Stable baseline, ~0.11 FPS CPU |
| v2.0 | ROI-first search + confidence states | Faster, but empty-grass false positives |
| v2.1 | Hard visual verifier | Too aggressive: LOST=79, FPS=0.23 |
| v2.2/2.2.1 | Balanced temporal + visual gate | Better recall but LOST ~75, FPS ~0.23 |
| v2.3 | Secondary crop re-detection | Current experiment |

Known validation regions on `gsGol1_goal_window.mp4`:
- real shot / ball quality: ~frames 120–132
- known empty-grass false positives: ~149, ~174, ~200, ~225

Production baseline remains Ball Tracking v1 until a v2.x experiment passes both
critical real-ball and false-positive gates.
