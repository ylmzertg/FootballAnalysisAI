from __future__ import annotations
import argparse, inspect, subprocess, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_manifest import verify_models

def ok(label, detail=""):
    print(f"[OK]   {label}" + (f" | {detail}" if detail else ""))

def fail(label, detail=""):
    print(f"[FAIL] {label}" + (f" | {detail}" if detail else ""))
    return False

def run_python(exe: Path, code: str):
    return subprocess.run([str(exe), "-c", code], capture_output=True, text=True, timeout=60)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines-root", default="")
    a = p.parse_args()
    engines = Path(a.engines_root).resolve() if a.engines_root else PROJECT_ROOT.parent / "CalibrationEngines"
    pnl = engines / "PnLCalib"
    tv = engines / "tvcalib"
    all_ok = True

    print("="*78)
    print("FootballAnalysisAI - Windows Health Check v2.0")
    print(f"Project : {PROJECT_ROOT}")
    print(f"Engines : {engines}")
    print("="*78)

    try:
        import numpy, cv2, torch, torchvision, sklearn
        ok("Main Python stack", f"py={sys.version.split()[0]} torch={torch.__version__} opencv={cv2.__version__} numpy={numpy.__version__} cuda={torch.cuda.is_available()}")
    except Exception as e:
        all_ok &= fail("Main Python stack", repr(e))

    try:
        from ultralytics import YOLO  # noqa
        ok("Ultralytics")
    except Exception as e:
        all_ok &= fail("Ultralytics", repr(e))

    try:
        import supervision as sv
        sig = inspect.signature(sv.ByteTrack)
        needed = {"track_activation_threshold","lost_track_buffer","frame_rate","minimum_consecutive_frames"}
        missing = needed - set(sig.parameters)
        if missing:
            all_ok &= fail("Supervision ByteTrack API", f"version={getattr(sv,'__version__','?')} missing={sorted(missing)}")
        else:
            ok("Supervision ByteTrack API", f"version={getattr(sv,'__version__','?')} {sig}")
    except Exception as e:
        all_ok &= fail("Supervision ByteTrack API", repr(e))

    try:
        from adapters.pnlcalib import PnLCalibAdapter
        ad = PnLCalibAdapter(pnl, device="cpu")
        ready, msg = ad.health_check()
        ok("PnLCalib adapter", msg) if ready else fail("PnLCalib adapter", msg)
        all_ok &= ready
    except Exception as e:
        all_ok &= fail("PnLCalib adapter", repr(e))

    pnlpy = pnl/".venv"/"Scripts"/"python.exe"
    tvpy = tv/".venv"/"Scripts"/"python.exe"

    if pnlpy.exists():
        r = run_python(pnlpy, "import torch,torchvision,cv2,numpy,scipy,yaml; print(torch.__version__,torchvision.__version__,cv2.__version__,numpy.__version__,scipy.__version__,torch.cuda.is_available())")
        ok("PnLCalib Python", r.stdout.strip()) if r.returncode == 0 else fail("PnLCalib Python", r.stderr.strip())
        all_ok &= r.returncode == 0
    else:
        all_ok &= fail("PnLCalib Python", str(pnlpy))

    ckpt = tv/"data"/"segment_localization"/"train_59.pt"
    if ckpt.exists() and ckpt.stat().st_size:
        ok("TVCalib checkpoint", f"{ckpt.stat().st_size} bytes")
    else:
        all_ok &= fail("TVCalib checkpoint", str(ckpt))

    if (tv/"sn_segmentation").exists():
        ok("TVCalib sn_segmentation submodule")
    else:
        all_ok &= fail("TVCalib sn_segmentation submodule")

    if tvpy.exists():
        r = run_python(tvpy, "import torch,torchvision,kornia,cv2,numpy,pandas,yaml,pytorch_lightning,SoccerNet; print(torch.__version__,torchvision.__version__,kornia.__version__,cv2.__version__,numpy.__version__,pandas.__version__,torch.cuda.is_available())")
        ok("TVCalib Python", r.stdout.strip()) if r.returncode == 0 else fail("TVCalib Python", r.stderr.strip())
        all_ok &= r.returncode == 0
    else:
        all_ok &= fail("TVCalib Python", str(tvpy))

    try:
        model_checks = verify_models(PROJECT_ROOT)

        if not model_checks:
            all_ok &= fail("Football model manifest", "manifest contains no models")
        else:
            for check in model_checks:
                detail = (
                    f"role={check.role} "
                    f"bytes={check.actual_bytes} "
                    f"size_ok={check.size_ok} "
                    f"sha256_ok={check.sha256_ok}"
                )

                if check.ok:
                    ok(f"Model {check.name}", detail)
                else:
                    all_ok &= fail(f"Model {check.name}", detail)

    except Exception as e:
        all_ok &= fail("Football model manifest", repr(e))

    print("="*78)
    if all_ok:
        print("READY - portable runtime baseline is installed.")
        return 0
    print("NOT READY - see [FAIL] lines above.")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
