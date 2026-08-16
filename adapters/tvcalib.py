from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from adapters.base import AdapterInfo, CalibrationAdapter
from schemas.match_state import CameraState


class TVCalibAdapter(CalibrationAdapter):
    """
    Process-isolated TVCalib fallback adapter.

    TVCalib stays inside its own virtual environment. FootballAnalysisAI
    invokes scripts/tvcalib_worker.py with that environment's Python.
    """

    def __init__(
        self,
        tv_root: str | Path,
        *,
        checkpoint: str | Path | None = None,
        python_exe: str | Path | None = None,
        worker_script: str | Path | None = None,
        device: str = "cuda",
        optim_steps: int = 800,
        tau: float = 0.017,
    ):
        super().__init__(
            AdapterInfo(
                name="TVCalib",
                source="https://github.com/MM4SPA/tvcalib",
                metadata={
                    "pitch_length_m": 105.0,
                    "pitch_width_m": 68.0,
                    "coordinate_system": "top_left_origin_x_0_105_y_0_68_m",
                    "role": "fallback",
                },
            )
        )

        self.tv_root = Path(tv_root).resolve()
        self.checkpoint = Path(
            checkpoint or self.tv_root / "data" / "segment_localization" / "train_59.pt"
        ).resolve()

        default_python = (
            self.tv_root / ".venv" / "Scripts" / "python.exe"
            if os.name == "nt"
            else self.tv_root / ".venv" / "bin" / "python"
        )
        self.python_exe = Path(python_exe or default_python).resolve()

        project_root = Path(__file__).resolve().parents[1]
        self.worker_script = Path(
            worker_script or project_root / "scripts" / "tvcalib_worker.py"
        ).resolve()

        self.device = device
        self.optim_steps = int(optim_steps)
        self.tau = float(tau)
        self.last_metrics: dict | None = None

    def health_check(self) -> tuple[bool, str]:
        checks = {
            "TVCalib root": self.tv_root,
            "TVCalib Python": self.python_exe,
            "segmentation checkpoint": self.checkpoint,
            "worker": self.worker_script,
        }

        missing = [
            f"{name}: {path}"
            for name, path in checks.items()
            if not Path(path).exists()
        ]

        if missing:
            return False, "Missing TVCalib files:\n" + "\n".join(missing)

        return True, (
            "TVCalib ready | "
            f"python={self.python_exe} | "
            f"device={self.device} | "
            f"optim_steps={self.optim_steps} | tau={self.tau}"
        )

    def _run_worker(self, frame_paths: list[Path]) -> list[dict]:
        ready, message = self.health_check()
        if not ready:
            raise RuntimeError(message)

        if not frame_paths:
            return []

        for frame_path in frame_paths:
            if not frame_path.exists():
                raise FileNotFoundError(frame_path)

        fd, tmp_name = tempfile.mkstemp(
            prefix="football_tvcalib_",
            suffix=".json",
        )
        os.close(fd)
        tmp_path = Path(tmp_name)

        cmd = [
            str(self.python_exe),
            str(self.worker_script),
            "--tv-root",
            str(self.tv_root),
            "--checkpoint",
            str(self.checkpoint),
            "--input",
            *[str(p.resolve()) for p in frame_paths],
            "--output",
            str(tmp_path),
            "--device",
            self.device,
            "--optim-steps",
            str(self.optim_steps),
            "--tau",
            str(self.tau),
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.tv_root),
                capture_output=True,
                text=True,
                timeout=7200,
                check=False,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    "TVCalib worker failed.\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}"
                )

            if not tmp_path.exists():
                raise RuntimeError(
                    "TVCalib worker finished without producing JSON output."
                )

            return json.loads(tmp_path.read_text(encoding="utf-8"))

        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _to_camera_state(item: dict) -> CameraState:
        if item.get("status") != "ok":
            raise RuntimeError(
                f"TVCalib failed for {item.get('input')}: "
                f"{item.get('error', 'unknown error')}"
            )

        return CameraState(
            homography=item["homography_image_to_pitch"],
            calibration_confidence=float(item.get("quality_score", 0.0)),
            calibration_engine="TVCalib",
        )

    def calibrate(self, frame_path: Path) -> CameraState:
        result = self._run_worker([Path(frame_path)])[0]
        self.last_metrics = result
        return self._to_camera_state(result)

    def calibrate_many_with_metrics(
        self,
        frame_paths: list[str | Path],
    ) -> list[dict]:
        paths = [Path(p).resolve() for p in frame_paths]
        return self._run_worker(paths)
