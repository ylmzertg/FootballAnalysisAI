from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from core.runtime_paths import resolve_project_path
from core.shot_context_image_goal import (
    UNKNOWN, RAW_LOOSE, TEAM_FLIGHT, PASS_FLIGHT, ATTACKING_FLIGHT,
    goal_image_geometry, normalized_ball_goal_distance,
)
from core.shot_window import (
    SHOT_FLIGHT, GOAL_ATTEMPT,
    LocalGoalSample, LocalShotWindowDetector,
)

CANDIDATE_PHASES = {RAW_LOOSE, TEAM_FLIGHT, PASS_FLIGHT, ATTACKING_FLIGHT}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=r"input\input.mp4")
    p.add_argument("--possession-jsonl", required=True)
    p.add_argument("--direction-jsonl", required=True)
    p.add_argument("--team-jsonl", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--jsonl", required=True)
    return p.parse_args()

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows

def read_jsonl_map(path):
    return {int(r["frame_index"]): r for r in read_jsonl(path)}

def read_calibration(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(r["frame_index"]): np.asarray(r["homography_image_to_pitch"], dtype=np.float64)
        for r in rows
        if r.get("status") == "ok" and r.get("homography_image_to_pitch") is not None
    }

def directions(rows):
    for r in rows:
        ad = r.get("attack_direction") or {}
        a = (ad.get("TEAM_A") or {}).get("direction", UNKNOWN)
        b = (ad.get("TEAM_B") or {}).get("direction", UNKNOWN)
        if a != UNKNOWN or b != UNKNOWN:
            return {"TEAM_A": a, "TEAM_B": b}
    return {"TEAM_A": UNKNOWN, "TEAM_B": UNKNOWN}

def phase(r):
    return str(r.get("phase_v14", r.get("phase_v13", r.get("phase", RAW_LOOSE))))

def team_state(r):
    return r.get("team_state_v14", r.get("team_state_v13", r.get("team_state", "UNKNOWN")))

def source_team(r):
    t = r.get("source_team")
    if t in {"TEAM_A", "TEAM_B"}:
        return t
    t = team_state(r)
    return t if t in {"TEAM_A", "TEAM_B"} else None

def build_runs(rows):
    runs = []
    i = 0
    while i < len(rows):
        if phase(rows[i]) not in CANDIDATE_PHASES:
            i += 1
            continue
        start = i
        team = source_team(rows[i])
        i += 1
        while i < len(rows) and phase(rows[i]) in CANDIDATE_PHASES:
            cur = source_team(rows[i])
            if team is not None and cur is not None and cur != team:
                break
            if team is None and cur is not None:
                team = cur
            i += 1
        runs.append((start, i-1, team))
    return runs

def ball_xy(r):
    xy = (r.get("ball") or {}).get("image_xy")
    if xy is None:
        return None
    return float(xy[0]), float(xy[1])

def main():
    args = parse_args()
    source = resolve_project_path(args.source)
    pos_path = resolve_project_path(args.possession_jsonl)
    dir_path = resolve_project_path(args.direction_jsonl)
    team_path = resolve_project_path(args.team_jsonl)
    cal_path = resolve_project_path(args.calibration_json)
    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)

    rows = read_jsonl(pos_path)
    team_rows = read_jsonl_map(team_path)
    cal = read_calibration(cal_path)
    dirs = directions(read_jsonl(dir_path))
    detector = LocalShotWindowDetector()
    enhanced = [dict(r) for r in rows]
    detected = []

    for start, end, team in build_runs(rows):
        if team not in {"TEAM_A", "TEAM_B"}:
            continue
        direction = dirs.get(team, UNKNOWN)
        if direction == UNKNOWN:
            continue
        samples = []
        for r in rows[start:end+1]:
            fi = int(r["frame_index"])
            bxy = ball_xy(r)
            tr = team_rows.get(fi)
            if bxy is None or tr is None:
                continue
            cf = tr.get("pnl_calibration_frame")
            if cf is None or int(cf) not in cal:
                continue
            geom = goal_image_geometry(cal[int(cf)], direction)
            if geom is None:
                continue
            d = normalized_ball_goal_distance(bxy, geom, 6.0)
            if d is not None:
                samples.append(LocalGoalSample(fi, float(d)))
        best = detector.best_window(samples)
        if best is None:
            continue
        detected.append((team, best))
        for r in enhanced:
            fi = int(r["frame_index"])
            if best.start_frame <= fi <= best.end_frame:
                r["phase_v15"] = best.classification
                r["team_state_v15"] = team
                r["local_shot_window"] = best.__dict__

    for r in enhanced:
        r.setdefault("phase_v15", phase(r))
        r.setdefault("team_state_v15", team_state(r))
        r.setdefault("local_shot_window", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    by_frame = {int(r["frame_index"]): r for r in enhanced}
    stats = Counter()
    idx = 0
    last = max(by_frame)
    colors = {"TEAM_A":(255,90,40),"TEAM_B":(40,70,255),"LOOSE":(0,215,255),"UNKNOWN":(150,150,150)}

    with jsonl_out.open("w", encoding="utf-8") as out:
        try:
            while idx <= last:
                ok, frame = cap.read()
                if not ok:
                    break
                r = by_frame.get(idx)
                if r is None:
                    idx += 1
                    continue
                team = r.get("team_state_v15", "UNKNOWN")
                ph = r.get("phase_v15", RAW_LOOSE)
                stats[f"team:{team}"] += 1
                stats[f"phase:{ph}"] += 1
                cv2.rectangle(frame,(0,0),(w,120),(18,18,18),-1)
                cv2.putText(frame,f"Shot Context v1.5 | {team} | {ph}",(18,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.66,colors.get(team,(200,200,200)),2,cv2.LINE_AA)
                win = r.get("local_shot_window") or {}
                if win:
                    cv2.putText(frame,
                        f"Local {win['start_frame']}..{win['end_frame']} W={win['window_size']} closing={win['closing']:.2f}",
                        (18,60),cv2.FONT_HERSHEY_SIMPLEX,0.47,(235,235,235),1,cv2.LINE_AA)
                    cv2.putText(frame,
                        f"closest={win['closest_distance']:.2f} goal-widths approach={win['approach_fraction']:.2f}",
                        (18,88),cv2.FONT_HERSHEY_SIMPLEX,0.43,(180,180,180),1,cv2.LINE_AA)
                cv2.putText(frame,f"Directions: A={dirs['TEAM_A']} B={dirs['TEAM_B']}",
                            (18,112),cv2.FONT_HERSHEY_SIMPLEX,0.40,(150,150,150),1,cv2.LINE_AA)
                writer.write(frame)
                out.write(json.dumps(r, ensure_ascii=False)+"\n")
                idx += 1
        finally:
            cap.release()
            writer.release()

    print("="*88)
    print("DONE - Shot Context v1.5 LOCAL WINDOWS")
    print(f"Directions: A={dirs['TEAM_A']} B={dirs['TEAM_B']}")
    print("Detected local windows:")
    if not detected:
        print("  NONE")
    else:
        for team, win in detected:
            print(
                f"  {team} | {win.classification} | frames={win.start_frame}..{win.end_frame} "
                f"| W={win.window_size} | closing={win.closing:.3f} "
                f"| approach={win.approach_fraction:.2f} | closest={win.closest_distance:.3f}"
            )
    print(f"Video output: {output}")
    print(f"JSONL output: {jsonl_out}")
    print("="*88)

if __name__ == "__main__":
    main()
