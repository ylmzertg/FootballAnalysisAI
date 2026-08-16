from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI external PnLCalib worker"
    )
    p.add_argument("--pnl-root", required=True)
    p.add_argument("--weights-kp", required=True)
    p.add_argument("--weights-line", required=True)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kp-threshold", type=float, default=0.3434)
    p.add_argument("--line-threshold", type=float, default=0.7867)
    p.add_argument("--pnl-refine", action="store_true")
    return p.parse_args()


def jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def scalar_float(value: Any, default=float("inf")) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        pass
    return float(default)


def projection_from_cam_params(final_params_dict: dict) -> np.ndarray:
    cam_params = final_params_dict["cam_params"]

    x_focal_length = float(cam_params["x_focal_length"])
    y_focal_length = float(cam_params["y_focal_length"])
    principal_point = np.asarray(
        cam_params["principal_point"], dtype=np.float64
    )
    position_meters = np.asarray(
        cam_params["position_meters"], dtype=np.float64
    )
    rotation = np.asarray(
        cam_params["rotation_matrix"], dtype=np.float64
    )

    extrinsic = np.eye(4, dtype=np.float64)[:3, :]
    extrinsic[:, -1] = -position_meters

    K = np.array(
        [
            [x_focal_length, 0.0, principal_point[0]],
            [0.0, y_focal_length, principal_point[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return K @ (rotation @ extrinsic)


def image_to_pitch_homography(final_params_dict: dict) -> np.ndarray:
    """
    Reproduce the exact camera solution used by PnLCalib's inference.py.

    PnLCalib world coordinates are centered on the pitch:
      X in [-52.5, +52.5]
      Y in [-34.0, +34.0]

    FootballAnalysisAI standardizes this to:
      x in [0, 105] metres
      y in [0, 68] metres
    """
    P = projection_from_cam_params(final_params_dict)

    # For pitch plane Z=0:
    # [u,v,w]^T = P[:, [0,1,3]] [X,Y,1]^T
    H_world_to_image = P[:, [0, 1, 3]]

    if abs(np.linalg.det(H_world_to_image)) < 1e-12:
        raise RuntimeError("PnLCalib produced a singular pitch homography.")

    H_image_to_centered_world = np.linalg.inv(H_world_to_image)

    centered_to_pitch = np.array(
        [
            [1.0, 0.0, PITCH_LENGTH_M / 2.0],
            [0.0, 1.0, PITCH_WIDTH_M / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    H = centered_to_pitch @ H_image_to_centered_world

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


def quality_from_rep_error(rep_err: float) -> float:
    """
    Heuristic [0,1] score for engine selection.
    This is NOT a calibrated probability.
    """
    if not math.isfinite(rep_err):
        return 0.0
    return float(np.clip(np.exp(-max(rep_err, 0.0) / 5.0), 0.0, 1.0))


def main() -> None:
    args = parse_args()

    pnl_root = Path(args.pnl_root).resolve()
    weights_kp = Path(args.weights_kp).resolve()
    weights_line = Path(args.weights_line).resolve()
    output_path = Path(args.output).resolve()

    if not pnl_root.exists():
        raise FileNotFoundError(f"PnLCalib root not found: {pnl_root}")
    if not weights_kp.exists():
        raise FileNotFoundError(f"PnL keypoint weights not found: {weights_kp}")
    if not weights_line.exists():
        raise FileNotFoundError(f"PnL line weights not found: {weights_line}")

    sys.path.insert(0, str(pnl_root))
    os.chdir(pnl_root)

    # Imports from the external repository happen only after sys.path is set.
    import torchvision.transforms as T
    import torchvision.transforms.functional as F
    from PIL import Image

    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (
        get_keypoints_from_heatmap_batch_maxpool,
        get_keypoints_from_heatmap_batch_maxpool_l,
        complete_keypoints,
        coords_to_dict,
    )

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = requested_device

    print(f"[PnL worker] device={device}")
    if device.startswith("cuda"):
        print(f"[PnL worker] GPU={torch.cuda.get_device_name(0)}")

    cfg = yaml.safe_load(
        (pnl_root / "config" / "hrnetv2_w48.yaml").read_text(encoding="utf-8")
    )
    cfg_l = yaml.safe_load(
        (pnl_root / "config" / "hrnetv2_w48_l.yaml").read_text(encoding="utf-8")
    )

    print("[PnL worker] loading keypoint model...")
    state_kp = torch.load(str(weights_kp), map_location=device)
    model_kp = get_cls_net(cfg)
    model_kp.load_state_dict(state_kp)
    model_kp.to(device)
    model_kp.eval()

    print("[PnL worker] loading line model...")
    state_line = torch.load(str(weights_line), map_location=device)
    model_line = get_cls_net_l(cfg_l)
    model_line.load_state_dict(state_line)
    model_line.to(device)
    model_line.eval()

    transform = T.Resize((540, 960))
    results: list[dict[str, Any]] = []

    for item in args.input:
        image_path = Path(item).resolve()

        result: dict[str, Any] = {
            "input": str(image_path),
            "status": "error",
            "engine": "PnLCalib",
            "pitch_length_m": PITCH_LENGTH_M,
            "pitch_width_m": PITCH_WIDTH_M,
            "coordinate_system": "top_left_origin_x_0_105_y_0_68_m",
        }

        try:
            frame_bgr = cv2.imread(str(image_path))
            if frame_bgr is None:
                raise RuntimeError(f"Could not read image: {image_path}")

            h_original, w_original = frame_bgr.shape[:2]

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = F.to_tensor(pil).float().unsqueeze(0)

            if tensor.size(-1) != 960:
                tensor = transform(tensor)

            tensor = tensor.to(device)

            with torch.no_grad():
                heatmaps_kp = model_kp(tensor)
                heatmaps_line = model_line(tensor)

            kp_coords = get_keypoints_from_heatmap_batch_maxpool(
                heatmaps_kp[:, :-1, :, :]
            )
            line_coords = get_keypoints_from_heatmap_batch_maxpool_l(
                heatmaps_line[:, :-1, :, :]
            )

            kp_dict = coords_to_dict(
                kp_coords, threshold=args.kp_threshold
            )
            line_dict = coords_to_dict(
                line_coords, threshold=args.line_threshold
            )

            _, _, h_net, w_net = tensor.size()

            kp_complete, lines_complete = complete_keypoints(
                kp_dict[0],
                line_dict[0],
                w=w_net,
                h=h_net,
                normalize=True,
            )

            cam = FramebyFrameCalib(
                iwidth=w_original,
                iheight=h_original,
                denormalize=True,
            )
            cam.update(kp_complete, lines_complete)

            # This is the same camera-selection path used by PnLCalib inference.py.
            final_params = cam.heuristic_voting(
                refine_lines=args.pnl_refine
            )

            if final_params is None:
                raise RuntimeError("PnLCalib could not estimate camera parameters.")

            H_img_to_pitch = image_to_pitch_homography(final_params)
            rep_err = scalar_float(final_params.get("rep_err"))

            result.update(
                {
                    "status": "ok",
                    "image_width": w_original,
                    "image_height": h_original,
                    "rep_err": rep_err,
                    "quality_score": quality_from_rep_error(rep_err),
                    "mode": final_params.get("mode"),
                    "use_ransac": jsonable(final_params.get("use_ransac")),
                    "homography_image_to_pitch": H_img_to_pitch.tolist(),
                    "cam_params": jsonable(final_params.get("cam_params")),
                }
            )

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

        results.append(result)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ok_count = sum(r["status"] == "ok" for r in results)
    print(
        f"[PnL worker] finished: {ok_count}/{len(results)} successful -> "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
