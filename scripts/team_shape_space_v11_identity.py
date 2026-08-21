from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

from core.runtime_paths import resolve_project_path
from core.team_shape import TEAM_A, TEAM_B, ShapePlayer, SpaceConfig, SpaceDetector, TeamShapeAnalyzer

def parse_args():
    p = argparse.ArgumentParser(description="FootballAnalysisAI - Team Shape + Space Detection v1.1 IDENTITY-AWARE")
    p.add_argument("--source", default=r"input\\input.mp4")
    p.add_argument("--team-jsonl", default=r"output\\team_classification_v24_pnl_exact.jsonl")
    p.add_argument("--possession-jsonl", default=r"output\\possession_v1.jsonl")
    p.add_argument("--output", default=r"output\\team_shape_space_v1.mp4")
    p.add_argument("--jsonl", default=r"output\\team_shape_space_v1.jsonl")
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--grid-step-m", type=float, default=4.0)
    p.add_argument("--max-spaces", type=int, default=6)
    return p.parse_args()

def read_jsonl(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rows[int(item["frame_index"])] = item
    return rows

def parse_players(team_row):
    out = []
    for tr in team_row.get("tracks", []):
        pitch = tr.get("pitch_xy")
        if not pitch or len(pitch) < 2:
            continue
        a = (
            tr.get("team_v29")
            or tr.get("team_v28")
            or tr.get("team_v27")
            or tr.get("team_v26")
            or tr.get("team_v25")
            or tr.get("team_v24")
            or {}
        )
        team = str(a.get("team","UNKNOWN"))
        role = str(a.get("role","PLAYER")).upper()
        source_class = str(tr.get("class_name","")).strip().lower()
        if team not in {TEAM_A,TEAM_B}:
            continue
        if role in {"REFEREE","OUTSIDE_PITCH"} or source_class == "referee":
            continue
        tid = int(tr.get("track_id",-1))
        if tid < 0:
            continue
        out.append(ShapePlayer(tid, team, (float(pitch[0]),float(pitch[1]))))
    return out

def shape_dict(s):
    return {
        "team": s.team,
        "player_count": s.player_count,
        "centroid_xy": list(s.centroid_xy) if s.centroid_xy else None,
        "width_m": round(s.width_m,4),
        "depth_m": round(s.depth_m,4),
        "compactness_m": round(s.compactness_m,4),
        "bbox_area_m2": round(s.bbox_area_m2,4),
        "min_x": s.min_x, "max_x": s.max_x,
        "min_y": s.min_y, "max_y": s.max_y,
    }

def draw_panel(frame, players, a, b, spaces, possessor_xy):
    panel_w = min(440, max(320, frame.shape[1]//3))
    panel_h = int(round(panel_w*68.0/105.0))
    margin = 14
    panel = np.zeros((panel_h,panel_w,3),dtype=np.uint8)
    panel[:] = (38,104,45)
    def p(x,y):
        return (
            margin + int(round((x/105.0)*(panel_w-2*margin))),
            margin + int(round((y/68.0)*(panel_h-2*margin))),
        )
    white = (230,230,230)
    cv2.rectangle(panel,p(0,0),p(105,68),white,1,cv2.LINE_AA)
    cv2.line(panel,p(52.5,0),p(52.5,68),white,1,cv2.LINE_AA)
    for s,color in ((a,(255,90,40)),(b,(40,70,255))):
        if s.player_count >= 2:
            cv2.rectangle(panel,p(s.min_x,s.min_y),p(s.max_x,s.max_y),color,1,cv2.LINE_AA)
        if s.centroid_xy:
            cv2.drawMarker(panel,p(*s.centroid_xy),color,cv2.MARKER_CROSS,10,2,cv2.LINE_AA)
    for pl in players:
        color = (255,90,40) if pl.team == TEAM_A else (40,70,255)
        cv2.circle(panel,p(*pl.pitch_xy),5,color,-1,cv2.LINE_AA)
    for i,s in enumerate(spaces):
        cv2.circle(panel,p(*s.xy),max(6,12-i),(0,255,255),2,cv2.LINE_AA)
    if possessor_xy:
        cv2.circle(panel,p(*possessor_xy),8,(255,255,255),2,cv2.LINE_AA)
    x0 = frame.shape[1]-panel_w-8
    y0 = frame.shape[0]-panel_h-8
    if x0 >= 0 and y0 >= 0:
        roi = frame[y0:y0+panel_h,x0:x0+panel_w]
        cv2.addWeighted(panel,0.92,roi,0.08,0,roi)

def main():
    args = parse_args()
    source = resolve_project_path(args.source)
    team_path = resolve_project_path(args.team_jsonl)
    poss_path = resolve_project_path(args.possession_jsonl)
    output = resolve_project_path(args.output)
    jsonl_out = resolve_project_path(args.jsonl)
    for pth in (source,team_path,poss_path):
        if not pth.exists():
            raise FileNotFoundError(pth)
    team_rows = read_jsonl(team_path)
    poss_rows = read_jsonl(poss_path)
    common = sorted(set(team_rows)&set(poss_rows))
    if args.max_frames >= 0:
        common = common[:args.max_frames]
    analyzer = TeamShapeAnalyzer()
    detector = SpaceDetector(SpaceConfig(grid_step_m=max(2.0,args.grid_step_m),max_spaces=max(1,args.max_spaces)))
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    frame_set = set(common)
    last_frame = max(common)
    stats = Counter()
    frame_index = 0
    with jsonl_out.open("w",encoding="utf-8") as out:
        try:
            while frame_index <= last_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index not in frame_set:
                    frame_index += 1
                    continue
                team_row = team_rows[frame_index]
                poss_row = poss_rows[frame_index]
                players = parse_players(team_row)
                shape_a = analyzer.shape(TEAM_A,players)
                shape_b = analyzer.shape(TEAM_B,players)
                poss = poss_row.get("possession") or {}
                poss_team = str(poss.get("state","UNKNOWN"))
                poss_id = poss.get("possessor_track_id")
                pmap = {p.track_id:p for p in players}
                possessor = pmap.get(int(poss_id)) if poss_id is not None else None
                spaces = []
                if possessor and poss_team in {TEAM_A,TEAM_B}:
                    teammates = [p for p in players if p.team == poss_team]
                    opponents = [p for p in players if p.team != poss_team]
                    spaces = detector.detect(possessor.pitch_xy,teammates,opponents)
                    stats["space_frames"] += 1
                    stats["spaces"] += len(spaces)
                else:
                    stats["no_possessor"] += 1
                draw_panel(frame,players,shape_a,shape_b,spaces,possessor.pitch_xy if possessor else None)
                cv2.rectangle(frame,(0,0),(width,110),(18,18,18),-1)
                cv2.putText(frame,f"Shape+Space v1 | Possession={poss_team} | spaces={len(spaces)}",(18,28),cv2.FONT_HERSHEY_SIMPLEX,0.60,(235,235,235),2,cv2.LINE_AA)
                cv2.putText(frame,f"A width={shape_a.width_m:.1f}m depth={shape_a.depth_m:.1f}m compact={shape_a.compactness_m:.1f}m",(18,58),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,140,90),1,cv2.LINE_AA)
                cv2.putText(frame,f"B width={shape_b.width_m:.1f}m depth={shape_b.depth_m:.1f}m compact={shape_b.compactness_m:.1f}m",(18,86),cv2.FONT_HERSHEY_SIMPLEX,0.48,(100,130,255),1,cv2.LINE_AA)
                writer.write(frame)
                out.write(json.dumps({
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index/fps,5),
                    "possession_team": poss_team,
                    "possessor_track_id": poss_id,
                    "team_shapes": {TEAM_A:shape_dict(shape_a),TEAM_B:shape_dict(shape_b)},
                    "spaces": [{
                        "xy":[round(s.xy[0],4),round(s.xy[1],4)],
                        "score":round(s.score,5),
                        "opponent_clearance_m":round(s.opponent_clearance_m,4),
                        "nearest_teammate_m":round(s.nearest_teammate_m,4),
                        "possessor_distance_m":round(s.possessor_distance_m,4),
                    } for s in spaces]
                },ensure_ascii=False)+"\n")
                frame_index += 1
        finally:
            cap.release()
            writer.release()
    print("="*80)
    print("DONE - Team Shape + Space Detection v1.1 IDENTITY-AWARE")
    print(f"Frames with spaces : {stats['space_frames']}")
    print(f"No possessor frames: {stats['no_possessor']}")
    print(f"Total spaces       : {stats['spaces']}")
    print(f"Video output       : {output}")
    print(f"JSONL output       : {jsonl_out}")
    print("="*80)

if __name__ == "__main__":
    main()
