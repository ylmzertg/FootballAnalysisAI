from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class BenchmarkMetrics:
    quality: float
    robustness: float
    speed: float
    integration: float
    license_score: float

    def total_score(
        self,
        quality_weight: float = 0.50,
        robustness_weight: float = 0.20,
        speed_weight: float = 0.15,
        integration_weight: float = 0.10,
        license_weight: float = 0.05,
    ) -> float:
        return (
            self.quality * quality_weight
            + self.robustness * robustness_weight
            + self.speed * speed_weight
            + self.integration * integration_weight
            + self.license_score * license_weight
        )


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    metrics: BenchmarkMetrics
    runtime_seconds: float
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.metrics.total_score()


class BenchmarkRunner(Generic[T]):
    """
    Generic benchmark harness.

    Actual football-specific evaluators will plug into this:
    - calibration reprojection / line alignment
    - player detection precision/recall
    - tracking ID switches / HOTA
    - ball coverage
    - team classification accuracy
    """

    def __init__(self) -> None:
        self.results: list[BenchmarkResult] = []

    def timed_run(
        self,
        name: str,
        fn: Callable[[], T],
    ) -> tuple[T, float]:
        start = perf_counter()
        value = fn()
        elapsed = perf_counter() - start
        return value, elapsed

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def ranking(self) -> list[BenchmarkResult]:
        return sorted(
            self.results,
            key=lambda x: x.score,
            reverse=True,
        )

    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return mean(r.score for r in self.results)
