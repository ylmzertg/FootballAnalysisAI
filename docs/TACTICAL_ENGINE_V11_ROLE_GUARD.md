# Tactical Engine v1.1 — Role Guard

Visual validation of `gsGol1_tactical_v1.mp4` showed a pass lane targeting the referee.

v1.1 adds a hard referee veto using:
- `team_v24.role`
- original detector `class_name`
- optional `role_hint`

A detector-labelled referee can no longer become a pass receiver or tactical player.

Apply:

```powershell
python scripts\apply_tactical_v11_role_guard.py
python -m pytest tests\test_tactical_engine.py tests\test_tactical_role_guard.py -v
```

Then regenerate the 30-frame tactical smoke video.
