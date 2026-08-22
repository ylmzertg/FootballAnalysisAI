from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REGRESSION_TESTS = [
    ("Ball Tracker V1", "tests/test_ball_tracker.py"),
    ("Team Identity V2.9", "tests/test_team_deep_embedding_v29.py"),
    ("Shot Context V1.6", "tests/test_shot_context_v16_guard.py"),
    ("Video i18n", "tests/test_video_i18n.py"),
    ("Analysis Pipeline", "tests/test_analysis_pipeline.py"),
    ("Analyst Incident V1", "tests/test_analyst_incident_v1.py"),
    ("Decision Comparison V1", "tests/test_decision_comparison_v1.py"),
    ("Analyst Renderer V2", "tests/test_analyst_renderer_v2.py"),
]


def run(label: str, command: list[str]) -> bool:
    print()
    print("=" * 78)
    print(f"[RUN] {label}")
    print("=" * 78)

    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if proc.returncode == 0:
        print(f"[PASS] {label}")
        return True

    print(f"[FAIL] {label} | exit={proc.returncode}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FootballAnalysisAI Portable Acceptance V1"
    )
    parser.add_argument(
        "--engines-root",
        default="",
        help="Optional CalibrationEngines root.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
    )
    args = parser.parse_args()

    engines_root = (
        Path(args.engines_root).resolve()
        if args.engines_root
        else PROJECT_ROOT.parent / "CalibrationEngines"
    )

    print("=" * 78)
    print("FootballAnalysisAI - Portable Acceptance V1")
    print(f"Project : {PROJECT_ROOT}")
    print(f"Python  : {sys.executable}")
    print(f"Engines : {engines_root}")
    print("=" * 78)

    all_ok = True

    if not args.skip_health_check:
        health_script = PROJECT_ROOT / "scripts" / "health_check.py"

        all_ok &= run(
            "Windows Health Check V2",
            [
                sys.executable,
                str(health_script),
                "--engines-root",
                str(engines_root),
            ],
        )

    for label, relative_path in REGRESSION_TESTS:
        test_path = PROJECT_ROOT / relative_path

        if not test_path.exists():
            print(f"[FAIL] {label} | missing test: {test_path}")
            all_ok = False
            continue

        all_ok &= run(
            label,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(test_path),
            ],
        )

    print()
    print("=" * 78)

    if all_ok:
        print("PORTABLE ACCEPTANCE: PASS")
        print("READY FOR DEVELOPMENT")
        print("=" * 78)
        return 0

    print("PORTABLE ACCEPTANCE: FAIL")
    print("NOT READY - see failed checks above.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
