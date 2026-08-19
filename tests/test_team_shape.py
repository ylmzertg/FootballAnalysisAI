from core.team_shape import TEAM_A,TEAM_B,ShapePlayer,SpaceConfig,SpaceDetector,TeamShapeAnalyzer

def p(tid,team,x,y):
    return ShapePlayer(tid,team,(x,y))

def test_team_shape_metrics():
    a = TeamShapeAnalyzer()
    s = a.shape(TEAM_A,[p(1,TEAM_A,10,10),p(2,TEAM_A,20,20),p(3,TEAM_A,30,30)])
    assert s.player_count == 3
    assert s.centroid_xy == (20.0,20.0)
    assert abs(s.width_m-20.0) < 1e-6
    assert abs(s.depth_m-20.0) < 1e-6

def test_space_detector_clearance():
    d = SpaceDetector(SpaceConfig(grid_step_m=5.0,min_opponent_clearance_m=4.0,max_spaces=4))
    teammates = [p(1,TEAM_A,20,20),p(2,TEAM_A,30,20),p(3,TEAM_A,20,30)]
    opponents = [p(10,TEAM_B,25,25),p(11,TEAM_B,40,20),p(12,TEAM_B,20,40)]
    spaces = d.detect((20,20),teammates,opponents)
    assert spaces
    assert all(s.opponent_clearance_m >= 4.0 for s in spaces)

def test_space_detector_separation():
    d = SpaceDetector(SpaceConfig(grid_step_m=2.0,min_space_separation_m=8.0,max_spaces=5))
    teammates = [p(1,TEAM_A,50,34),p(2,TEAM_A,55,34)]
    opponents = [p(10,TEAM_B,80,60)]
    spaces = d.detect((50,34),teammates,opponents)
    for i in range(len(spaces)):
        for j in range(i+1,len(spaces)):
            dx=spaces[i].xy[0]-spaces[j].xy[0]
            dy=spaces[i].xy[1]-spaces[j].xy[1]
            assert (dx*dx+dy*dy)**0.5 >= 8.0-1e-6
