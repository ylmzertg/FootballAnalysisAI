# Error Detection v1.1 — Sequenced Analyst Events

V1 proved the concept but had three problems:

1. the same tactical issue was repeated frame-by-frame;
2. generic defensive logic could mark a goalkeeper as `LATE_PRESSURE`;
3. possession/tactical inputs still carried older V2.5 team identity.

V1.1 fixes the event layer.

## Required preparation

Re-run Tactical Engine using Team Identity v2.9:

```powershell
python -m scripts.tactical_engine_v1 `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --possession-jsonl "output\gsGol1_goal_possession_v11_v25.jsonl" `
  --output "output\gsGol1_goal_tactical_v29.mp4" `
  --jsonl "output\gsGol1_goal_tactical_v29.jsonl"
```

## V1.1 changes

- attacking team is resolved from the **current V2.9 possessor track**, not the
  stale V2.5 possession team label;
- goalkeeper/referee/outside-pitch tracks are excluded from generic error logic;
- weak possession confidence is ignored;
- runner/lane thresholds are tighter;
- repeated frame candidates become one temporal analyst event.

Example:

```text
frames 88..97
UNMARKED_RUNNER
        ↓
ERR-0003
start=88
peak=92
end=97
```

The final Analyst Renderer will use the peak and event window, rather than
printing the same error ten times.

## Test

```powershell
python -m pytest `
  tests\test_error_detection_v1.py `
  tests\test_error_event_sequence_v11.py `
  -v
```

## Run

```powershell
python -m scripts.detect_errors_v11 `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --possession-jsonl "output\gsGol1_goal_possession_v11_v25.jsonl" `
  --direction-jsonl "output\gsGol1_goal_defline_v11.jsonl" `
  --tactical-jsonl "output\gsGol1_goal_tactical_v29.jsonl" `
  --output "output\gsGol1_goal_errors_v11.mp4" `
  --jsonl "output\gsGol1_goal_errors_v11.jsonl" `
  --timeline-json "output\gsGol1_goal_errors_v11_timeline.json"
```

The timeline JSON is the first direct input for the future:

```text
ERROR
  -> THREAT
  -> PASS OPTIONS
  -> CONTINUATION
  -> RESULT
```

Analyst Event Sequencer.
