
from __future__ import annotations

import argparse, json
from pathlib import Path
import cv2
import numpy as np

from core.attack_direction import (
    TEAM_A, TEAM_B, UNKNOWN,
    DirectionPlayer, AttackDirectionResolver, DefensiveLineEstimator,
)
from core.runtime_paths import resolve_project_path


def parse_args():
    p=argparse.ArgumentParser(
        description="FootballAnalysisAI - Attack Direction + Defensive Line v1"
    )
    p.add_argument("--source",default=r"input\input.mp4")
    p.add_argument("--team-jsonl",default=r"output\team_classification_v24_pnl_exact.jsonl")
    p.add_argument("--output",default=r"output\attack_direction_defline_v1.mp4")
    p.add_argument("--jsonl",default=r"output\attack_direction_defline_v1.jsonl")
    p.add_argument("--team-a-attacks",choices=["auto","left","right"],default="auto")
    p.add_argument("--team-b-attacks",choices=["auto","left","right"],default="auto")
    p.add_argument("--max-frames",type=int,default=-1)
    return p.parse_args()


def read_jsonl(path):
    out={}
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line); out[int(x["frame_index"])]=x
    return out


def parse_players(row):
    out=[]
    for tr in row.get("tracks",[]):
        xy=tr.get("pitch_xy")
        if not xy or len(xy)<2: continue
        a=tr.get("team_v24") or {}
        team=str(a.get("team","UNKNOWN"))
        role=str(a.get("role","PLAYER")).upper()
        source_class=str(tr.get("class_name","")).lower().strip()
        if team not in {TEAM_A,TEAM_B}: continue
        if source_class=="referee" or role in {"REFEREE","OUTSIDE_PITCH"}: continue
        tid=int(tr.get("track_id",-1))
        if tid<0: continue
        out.append(DirectionPlayer(tid,team,role,(float(xy[0]),float(xy[1]))))
    return out


def main():
    args=parse_args()
    source=resolve_project_path(args.source)
    team_path=resolve_project_path(args.team_jsonl)
    output=resolve_project_path(args.output)
    jsonl_out=resolve_project_path(args.jsonl)
    rows=read_jsonl(team_path)
    frames=sorted(rows)
    if args.max_frames>=0: frames=frames[:args.max_frames]
    if not frames: raise RuntimeError("No frames")

    resolver=AttackDirectionResolver()
    estimator=DefensiveLineEstimator()

    cap=cv2.VideoCapture(str(source))
    fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True,exist_ok=True); jsonl_out.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

    frame_set=set(frames); last=max(frames); idx=0
    with jsonl_out.open("w",encoding="utf-8") as out:
        try:
            while idx<=last:
                ok,frame=cap.read()
                if not ok: break
                if idx not in frame_set:
                    idx+=1; continue
                players=parse_players(rows[idx])
                ova=None if args.team_a_attacks=="auto" else args.team_a_attacks
                ovb=None if args.team_b_attacks=="auto" else args.team_b_attacks
                da=resolver.resolve(TEAM_A,players,ova)
                db=resolver.resolve(TEAM_B,players,ovb)
                la=estimator.estimate(TEAM_A,players,da.direction)
                lb=estimator.estimate(TEAM_B,players,db.direction)

                cv2.rectangle(frame,(0,0),(w,105),(18,18,18),-1)
                cv2.putText(frame,f"Attack Direction v1 | A={da.direction} ({da.source}) | B={db.direction} ({db.source})",
                            (18,30),cv2.FONT_HERSHEY_SIMPLEX,0.58,(235,235,235),2,cv2.LINE_AA)
                cv2.putText(frame,f"A defensive line x={la.line_x:.1f}m" if la.line_x is not None else "A defensive line: UNKNOWN",
                            (18,62),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,140,90),1,cv2.LINE_AA)
                cv2.putText(frame,f"B defensive line x={lb.line_x:.1f}m" if lb.line_x is not None else "B defensive line: UNKNOWN",
                            (18,89),cv2.FONT_HERSHEY_SIMPLEX,0.50,(100,130,255),1,cv2.LINE_AA)
                writer.write(frame)

                out.write(json.dumps({
                    "frame_index":idx,
                    "attack_direction":{
                        TEAM_A:da.__dict__,
                        TEAM_B:db.__dict__,
                    },
                    "defensive_line":{
                        TEAM_A:la.__dict__,
                        TEAM_B:lb.__dict__,
                    },
                },ensure_ascii=False)+"\n")
                idx+=1
        finally:
            cap.release(); writer.release()

    print("DONE - Attack Direction + Defensive Line v1")
    print(f"Video output: {output}")
    print(f"JSONL output: {jsonl_out}")


if __name__=="__main__":
    main()
