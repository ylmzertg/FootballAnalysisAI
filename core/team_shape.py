from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"

@dataclass(frozen=True)
class ShapePlayer:
    track_id: int
    team: str
    pitch_xy: tuple[float, float]

@dataclass(frozen=True)
class TeamShapeResult:
    team: str
    player_count: int
    centroid_xy: Optional[tuple[float, float]]
    width_m: float
    depth_m: float
    compactness_m: float
    bbox_area_m2: float
    min_x: Optional[float]
    max_x: Optional[float]
    min_y: Optional[float]
    max_y: Optional[float]

@dataclass(frozen=True)
class SpaceCandidate:
    xy: tuple[float, float]
    score: float
    opponent_clearance_m: float
    nearest_teammate_m: float
    possessor_distance_m: float

@dataclass
class SpaceConfig:
    grid_step_m: float = 4.0
    pitch_margin_m: float = 3.0
    min_opponent_clearance_m: float = 4.0
    max_teammate_support_m: float = 18.0
    max_possessor_distance_m: float = 32.0
    min_space_separation_m: float = 7.0
    max_spaces: int = 6

def dist(a, b):
    return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

class TeamShapeAnalyzer:
    def shape(self, team: str, players: Iterable[ShapePlayer]) -> TeamShapeResult:
        pts = [p.pitch_xy for p in players if p.team == team]
        if not pts:
            return TeamShapeResult(team,0,None,0.0,0.0,0.0,0.0,None,None,None,None)
        xs = [float(x) for x,_ in pts]
        ys = [float(y) for _,y in pts]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        min_x,max_x,min_y,max_y = min(xs),max(xs),min(ys),max(ys)
        depth = max_x-min_x
        width = max_y-min_y
        compactness = sum(hypot(x-cx,y-cy) for x,y in pts)/len(pts)
        return TeamShapeResult(
            team,len(pts),(cx,cy),width,depth,compactness,width*depth,
            min_x,max_x,min_y,max_y
        )

class SpaceDetector:
    def __init__(self, config: SpaceConfig | None = None):
        self.config = config or SpaceConfig()

    def detect(self, possessor_xy, teammates, opponents):
        teammates = list(teammates)
        opponents = list(opponents)
        if not teammates or not opponents:
            return []
        cfg = self.config
        raw = []
        step = max(1.0, cfg.grid_step_m)
        x = cfg.pitch_margin_m
        while x <= 105.0 - cfg.pitch_margin_m + 1e-6:
            y = cfg.pitch_margin_m
            while y <= 68.0 - cfg.pitch_margin_m + 1e-6:
                xy = (x,y)
                pd = dist(xy, possessor_xy)
                if pd <= cfg.max_possessor_distance_m:
                    oc = min(dist(xy,p.pitch_xy) for p in opponents)
                    td = min(dist(xy,p.pitch_xy) for p in teammates)
                    if oc >= cfg.min_opponent_clearance_m and td <= cfg.max_teammate_support_m:
                        clearance = min(1.0, oc/12.0)
                        support = max(0.0, 1.0-td/max(cfg.max_teammate_support_m,1e-6))
                        reach = max(0.0, 1.0-pd/max(cfg.max_possessor_distance_m,1e-6))
                        score = 0.60*clearance + 0.22*support + 0.18*reach
                        raw.append(SpaceCandidate(xy,score,oc,td,pd))
                y += step
            x += step
        raw.sort(key=lambda s: -s.score)
        selected = []
        for c in raw:
            if any(dist(c.xy,e.xy) < cfg.min_space_separation_m for e in selected):
                continue
            selected.append(c)
            if len(selected) >= max(1,cfg.max_spaces):
                break
        return selected
