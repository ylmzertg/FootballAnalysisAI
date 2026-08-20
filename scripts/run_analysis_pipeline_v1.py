
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from core.analysis_pipeline import (
    build_paths,
    build_steps,
    outputs_ready,
)
from core.runtime_paths import PROJECT_ROOT, resolve_project_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI - end-to-end Analysis Pipeline v1"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--run-name", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--imgsz", type=int, default=640)

    p.add_argument("--calibration-stride", type=int, default=15)
    p.add_argument("--max-calibration-gap", type=int, default=8)

    p.add_argument("--max-bridge-gap", type=int, default=75)
    p.add_argument("--max-unresolved-flight", type=int, default=50)

    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even when their outputs already exist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    p.add_argument(
        "--start-at",
        default="",
        help="Start from this step name; earlier steps must already exist.",
    )
    p.add_argument(
        "--stop-after",
        default="",
        help="Stop after this step name.",
    )
    return p.parse_args()


def run_command(module: str, args: tuple[str, ...], dry_run: bool):
    command = [
        sys.executable,
        "-m",
        module,
        *args,
    ]

    print()
    print("$", " ".join(f'"{x}"' if " " in x else x for x in command))

    if dry_run:
        return 0

    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Step failed ({proc.returncode}): {module}"
        )

    return proc.returncode


def main():
    args = parse_args()

    source = resolve_project_path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    run_name = (
        args.run_name.strip()
        or source.stem
    )

    paths = build_paths(
        PROJECT_ROOT,
        run_name,
    )
    paths.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = build_steps(
        project_root=PROJECT_ROOT,
        source=source,
        paths=paths,
        device=args.device,
        max_frames=args.max_frames,
        imgsz=args.imgsz,
        calibration_stride=args.calibration_stride,
        max_calibration_gap=args.max_calibration_gap,
        max_bridge_gap=args.max_bridge_gap,
        max_unresolved_flight=args.max_unresolved_flight,
    )

    names = [step.name for step in steps]

    if args.start_at and args.start_at not in names:
        raise ValueError(
            f"Unknown --start-at: {args.start_at}. "
            f"Choices: {', '.join(names)}"
        )

    if args.stop_after and args.stop_after not in names:
        raise ValueError(
            f"Unknown --stop-after: {args.stop_after}. "
            f"Choices: {', '.join(names)}"
        )

    start_index = (
        names.index(args.start_at)
        if args.start_at
        else 0
    )
    stop_index = (
        names.index(args.stop_after)
        if args.stop_after
        else len(steps) - 1
    )

    if stop_index < start_index:
        raise ValueError(
            "--stop-after comes before --start-at"
        )

    manifest = {
        "pipeline": "FootballAnalysisAI Analysis Pipeline v1",
        "source": str(source),
        "run_name": run_name,
        "device": args.device,
        "max_frames": args.max_frames,
        "steps": [],
    }

    started = time.perf_counter()

    print("=" * 92)
    print("FootballAnalysisAI - Analysis Pipeline v1")
    print(f"Source       : {source}")
    print(f"Run          : {run_name}")
    print(f"Run dir      : {paths.run_dir}")
    print(f"Device       : {args.device}")
    print(f"Max frames   : {args.max_frames}")
    print(f"Force        : {args.force}")
    print("=" * 92)

    for index, step in enumerate(steps):
        if index < start_index or index > stop_index:
            continue

        ready = outputs_ready(step.outputs)

        if ready and not args.force:
            status = "SKIPPED_READY"
            print()
            print(
                f"[SKIP] {step.name}: outputs already exist "
                "(use --force to rebuild)"
            )
        else:
            print()
            print(f"[RUN ] {step.name}")
            step_started = time.perf_counter()

            run_command(
                step.module,
                step.args,
                args.dry_run,
            )

            elapsed = time.perf_counter() - step_started
            status = "DRY_RUN" if args.dry_run else "DONE"

            if not args.dry_run and not outputs_ready(step.outputs):
                raise RuntimeError(
                    f"{step.name} finished but required outputs are missing."
                )

            print(
                f"[{status}] {step.name} "
                f"({elapsed:.1f}s)"
            )

        manifest["steps"].append(
            {
                "name": step.name,
                "status": status,
                "module": step.module,
                "outputs": [
                    str(p)
                    for p in step.outputs
                ],
            }
        )

    # Merge and final renderer are always safe/cheap when all dependencies exist.
    required_for_merge = (
        paths.team_jsonl,
        paths.ball_jsonl,
        paths.possession_v11_jsonl,
        paths.possession_v12_jsonl,
        paths.direction_jsonl,
        paths.tactical_jsonl,
        paths.shape_jsonl,
        paths.shot_jsonl,
    )

    if all(p.exists() for p in required_for_merge):
        merge_args = (
            "--team-jsonl", str(paths.team_jsonl),
            "--ball-jsonl", str(paths.ball_jsonl),
            "--possession-control-jsonl", str(paths.possession_v11_jsonl),
            "--possession-event-jsonl", str(paths.possession_v12_jsonl),
            "--direction-jsonl", str(paths.direction_jsonl),
            "--tactical-jsonl", str(paths.tactical_jsonl),
            "--shape-jsonl", str(paths.shape_jsonl),
            "--shot-jsonl", str(paths.shot_jsonl),
            "--output", str(paths.analysis_jsonl),
        )
        run_command(
            "scripts.merge_analysis_v1",
            merge_args,
            args.dry_run,
        )

        if not args.dry_run:
            render_args = (
                "--source", str(source),
                "--analysis-jsonl", str(paths.analysis_jsonl),
                "--output", str(paths.analysis_video),
            )
            run_command(
                "scripts.render_analysis_v1",
                render_args,
                False,
            )

    elapsed_total = time.perf_counter() - started
    manifest["elapsed_seconds"] = round(elapsed_total, 3)
    manifest["analysis_jsonl"] = str(paths.analysis_jsonl)
    manifest["analysis_video"] = str(paths.analysis_video)

    if not args.dry_run:
        paths.manifest_json.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print()
    print("=" * 92)
    print("PIPELINE FINISHED")
    print(f"Elapsed       : {elapsed_total:.1f}s")
    print(f"Analysis JSONL: {paths.analysis_jsonl}")
    print(f"Analysis video: {paths.analysis_video}")
    print(f"Manifest      : {paths.manifest_json}")
    print("=" * 92)


if __name__ == "__main__":
    main()
