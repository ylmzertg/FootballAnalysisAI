from benchmark.runner import BenchmarkMetrics
from core.engine import FootballAnalysisEngine
from schemas.match_state import BoundingBox, MatchFrame, PlayerState


def main() -> None:
    engine = FootballAnalysisEngine()

    frame = MatchFrame(
        frame_index=1,
        timestamp_seconds=0.04,
        players=[
            PlayerState(
                track_id=10,
                bbox=BoundingBox(
                    x=100,
                    y=150,
                    width=50,
                    height=120,
                    confidence=0.97,
                ),
                team_id="home",
            )
        ],
    )

    score = BenchmarkMetrics(
        quality=90,
        robustness=90,
        speed=80,
        integration=95,
        license_score=100,
    ).total_score()

    print("FootballAnalysisAI core check: OK")
    print("Registered engines:", engine.list_engines())
    print("Sample MatchState frame:", frame.to_dict())
    print("Sample benchmark score:", round(score, 2))


if __name__ == "__main__":
    main()
