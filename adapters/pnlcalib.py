from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from adapters.base import AdapterInfo, CalibrationAdapter
from schemas.match_state import CameraState


class PnLCalibAdapter(CalibrationAdapter):
    """
    Process-isolated PnLCalib adapter.

    PnLCalib remains in its own virtual environment. FootballAnalysisAI invokes
    a worker with the PnLCalib Python interpreter, preventing dependency clashes.
    """

    def __init__(
        self,
        pnl_root: str | Path,
        *,
        weights_kp: str | Path | None = None,
        weights_line: str | Path | None = None,
        python_exe: str | Path | None = None,
        worker_script: str | Path | None = None,
        device: str = "cuda:0",
        pnl_refine: bool = True,
    ):
        super().__init__(
            AdapterInfo(
                name="PnLCalib",
                source="https://github.com/mguti97/PnLCalib",
                metadata={
                    "pitch_length_m": 105.0,
                    "pitch_width_m": 68.0,
                    "coordinate_system": "top_left_origin_x_0_105_y_0_68_m",
                },
            )
        )

        self.pnl_root = Path(pnl_root).resolve()
        self.weights_kp = Path(
            weights_kp or self.pnl_root / "weights" / "SV_kp"
        ).resolve()
        self.weights_line = Path(
            weights_line or self.pnl_root / "weights" / "SV_lines"
        ).resolve()

        default_python = (
            self.pnl_root / ".venv" / "Scripts" / "python.exe"
            if os.name == "nt"
            else self.pnl_root / ".venv" / "bin" / "python"
        )
        self.python_exe = Path(python_exe or default_python).resolve()

        project_root = Path(__file__).resolve().parents[1]
        self.worker_script = Path(
            worker_script or project_root / "scripts" / "pnlcalib_worker.py"
        ).resolve()

        self.device = device
        self.pnl_refine = pnl_refine
        self.last_metrics: dict | None = None

    def health_check(self) -> tuple[bool, str]:
        checks = {
            "PnLCalib root": self.pnl_root,
            "PnL Python": self.python_exe,
            "SV_kp": self.weights_kp,
            "SV_lines": self.weights_line,
            "worker": self.worker_script,
            "kp config": self.pnl_root / "config" / "hrnetv2_w48.yaml",
            "line config": self.pnl_root / "config" / "hrnetv2_w48_l.yaml",
        }

        missing = [
            f"{name}: {path}"
            for name, path in checks.items()
            if not Path(path).exists()
        ]

        if missing:
            return False, "Missing PnLCalib files:\n" + "\n".join(missing)

        return True, (
            "PnLCalib ready | "
            f"python={self.python_exe} | "
            f"device={self.device}"
        )

    def _run_worker(self, frame_paths: list[Path]) -> list[dict]:
        ready, message = self.health_check()
        if not ready:
            raise RuntimeError(message)

        for frame_path in frame_paths:
            if not frame_path.exists():
                raise FileNotFoundError(frame_path)

        fd, tmp_name = tempfile.mkstemp(
            prefix="football_pnl_",
            suffix=".json",
        )
        os.close(fd)
        tmp_path = Path(tmp_name)

        cmd = [
            str(self.python_exe),
            str(self.worker_script),
            "--pnl-root",
            str(self.pnl_root),
            "--weights-kp",
            str(self.weights_kp),
            "--weights-line",
            str(self.weights_line),
            "--input",
            *[str(p.resolve()) for p in frame_paths],
            "--output",
            str(tmp_path),
            "--device",
            self.device,
        ]

        if self.pnl_refine:
            cmd.append("--pnl-refine")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.pnl_root),
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    "PnLCalib worker failed.\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}"
                )

            if not tmp_path.exists():
                raise RuntimeError(
                    "PnLCalib worker finished without producing JSON output."
                )

            payload = json.loads(
                tmp_path.read_text(encoding="utf-8")
            )

            return payload

        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _to_camera_state(item: dict) -> CameraState:
        if item.get("status") != "ok":
            raise RuntimeError(
                f"PnLCalib calibration failed for {item.get('input')}: "
                f"{item.get('error', 'unknown error')}"
            )

        return CameraState(
            homography=item["homography_image_to_pitch"],
            calibration_confidence=float(
                item.get("quality_score", 0.0)
            ),
            calibration_engine="PnLCalib",
        )

    def calibrate(self, frame_path: Path) -> CameraState:
        result = self._run_worker([Path(frame_path)])[0]
        self.last_metrics = result
        return self._to_camera_state(result)

    def calibrate_many(
        self,
        frame_paths: list[str | Path],
    ) -> dict[Path, CameraState]:
        """
        Load the two PnLCalib HRNet models only once for a group of frames.
        This is the preferred path for video processing.
        """
        paths = [Path(p).resolve() for p in frame_paths]
        results = self._run_worker(paths)

        output: dict[Path, CameraState] = {}

        for path, item in zip(paths, results):
            if item.get("status") == "ok":
                output[path] = self._to_camera_state(item)

        return output

    def calibrate_many_with_metrics(
        self,
        frame_paths: list[str | Path],
    ) -> list[dict]:
        paths = [Path(p).resolve() for p in frame_paths]
        return self._run_worker(paths)
