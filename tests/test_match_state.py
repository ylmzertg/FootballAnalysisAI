from schemas.match_state import (
    BoundingBox,
    MatchFrame,
    PlayerState,
)


def test_match_frame_serialization() -> None:
    frame = MatchFrame(
        frame_index=10,
        timestamp_seconds=0.4,
        players=[
            PlayerState(
                track_id=7,
                bbox=BoundingBox(
                    x=100,
                    y=200,
                    width=40,
                    height=100,
                    confidence=0.95,
                ),
                team_id="home",
            )
        ],
    )

    data = frame.to_dict()

    assert data["frame_index"] == 10
    assert data["players"][0]["track_id"] == 7
    assert data["players"][0]["team_id"] == "home"
