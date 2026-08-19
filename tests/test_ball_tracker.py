
from core.ball_tracker import BallCandidate, BallTemporalTracker, BallTrackerConfig


def c(x, y, conf=0.8):
    return BallCandidate((x-2,y-2,x+2,y+2),(x,y),conf)


def test_temporal_candidate_selection():
    tracker = BallTemporalTracker(BallTrackerConfig(max_jump_px=100, confidence_weight=50))
    assert tracker.update([c(100,100,0.9)],0).detected
    assert tracker.update([c(110,100,0.8)],1).detected
    r = tracker.update([c(120,100,0.7), c(400,300,0.95)],2)
    assert r.detected
    assert abs(r.center_xy[0]-120) < 1e-6


def test_short_gap_prediction():
    tracker = BallTemporalTracker(BallTrackerConfig(max_gap_frames=3))
    tracker.update([c(50,50,0.9)],0)
    tracker.update([c(60,50,0.9)],1)
    r = tracker.update([],2)
    assert r.predicted
    assert abs(r.center_xy[0]-70) < 1e-6


def test_prediction_stops_after_gap_limit():
    tracker = BallTemporalTracker(BallTrackerConfig(max_gap_frames=1))
    tracker.update([c(10,10)],0)
    assert tracker.update([],1).predicted
    r = tracker.update([],2)
    assert not r.predicted
    assert r.center_xy is None
