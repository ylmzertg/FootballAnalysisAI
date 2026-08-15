from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adapters.base import GameStateAdapter
from schemas.match_state import MatchFrame


@dataclass(slots=True)
class EngineResult:
    engine_name: str
    frames: list[MatchFrame]


class FootballAnalysisEngine:
    """
    Main orchestrator.

    First milestone:
    - register one or more end-to-end Game State engines
    - run the selected engine
    - normalize all outputs to MatchFrame / MatchState

    Later:
    - mix calibration/detection/tracking adapters independently
    - run tactical analysis
    - render telestration / YouTube output
    """

    def __init__(self) -> None:
        self._game_state_engines: dict[str, GameStateAdapter] = {}

    def register_game_state_engine(
        self,
        name: str,
        adapter: GameStateAdapter,
    ) -> None:
        key = name.strip().lower()

        if not key:
            raise ValueError("Engine name cannot be empty.")

        self._game_state_engines[key] = adapter

    def list_engines(self) -> list[str]:
        return sorted(self._game_state_engines.keys())

    def health_report(self) -> dict[str, dict[str, object]]:
        report: dict[str, dict[str, object]] = {}

        for name, adapter in self._game_state_engines.items():
            ok, message = adapter.health_check()

            report[name] = {
                "ready": ok,
                "message": message,
                "adapter": adapter.info.name,
                "version": adapter.info.version,
                "license": adapter.info.license,
            }

        return report

    def run(
        self,
        video_path: str | Path,
        engine_name: str,
    ) -> EngineResult:
        path = Path(video_path)

        if not path.exists():
            raise FileNotFoundError(path)

        key = engine_name.strip().lower()

        if key not in self._game_state_engines:
            available = ", ".join(self.list_engines()) or "(none)"
            raise KeyError(
                f"Unknown engine '{engine_name}'. Available: {available}"
            )

        adapter = self._game_state_engines[key]
        ready, message = adapter.health_check()

        if not ready:
            raise RuntimeError(
                f"Engine '{engine_name}' is not ready: {message}"
            )

        frames = adapter.reconstruct(path)

        return EngineResult(
            engine_name=key,
            frames=frames,
        )
