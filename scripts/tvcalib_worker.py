from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def parse_args():
    p = argparse.ArgumentParser(
        description="FootballAnalysisAI external TVCalib worker"
    )
    p.add_argument("--tv-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--optim-steps", type=int, default=800)
    p.add_argument(
        "--tau",
        type=float,
        default=0.017,
        help="TVCalib self-verification threshold on loss_ndc_total.",
    )
    return p.parse_args()


def jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
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


def centered_homography_to_pitch(homography) -> np.ndarray:
    """
    TVCalib's published pixel2world example stores image->pitch homography
    in pitch-centred coordinates. Translate the pitch centre to FootballAnalysisAI's
    upper-left 0..105 x 0..68 metre coordinate system.
    """
    H = np.asarray(homography, dtype=np.float64).reshape(3, 3)

    T = np.array(
        [
            [1.0, 0.0, PITCH_LENGTH_M / 2.0],
            [0.0, 1.0, PITCH_WIDTH_M / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    out = T @ H
    if abs(out[2, 2]) > 1e-12:
        out = out / out[2, 2]
    return out


def quality_from_loss(loss: float, tau: float) -> float:
    if not math.isfinite(loss):
        return 0.0
    scale = max(float(tau), 1e-9)
    return float(np.clip(math.exp(-max(loss, 0.0) / scale), 0.0, 1.0))


def main():
    args = parse_args()

    tv_root = Path(args.tv_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    inputs = [Path(x).resolve() for x in args.input]

    if not tv_root.exists():
        raise FileNotFoundError(f"TVCalib root not found: {tv_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"TVCalib checkpoint not found: {checkpoint}")
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)

    sys.path.insert(0, str(tv_root))
    os.chdir(tv_root)

    import pandas as pd

    from tvcalib.module import TVCalibModule
    from tvcalib.cam_distr.tv_main_center import get_cam_distr
    from tvcalib.utils.objects_3d import (
        SoccerPitchLineCircleSegments,
        SoccerPitchSNCircleCentralSplit,
    )
    import tvcalib.inference as tv_inference
    from tvcalib.inference import (
        InferenceDatasetSegmentation,
        InferenceDatasetCalibration,
        InferenceSegmentationModel,
    )

    # Avoid an external ImageNet download. The trained segmentation checkpoint
    # already carries the required network weights.
    from torchvision.models.segmentation import deeplabv3_resnet101 as _deeplabv3_resnet101

    def _deeplab_no_external_weights(*a, **kw):
        kw["weights"] = None
        kw["weights_backbone"] = None
        return _deeplabv3_resnet101(*a, **kw)

    tv_inference.deeplabv3_resnet101 = _deeplab_no_external_weights

    from tvcalib.sncalib_dataset import custom_list_collate
    from tvcalib.utils.io import detach_dict, tensor2list

    from sn_segmentation.src.custom_extremities import (
        generate_class_synthesis,
        get_line_extremities,
    )

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = requested_device

    print(f"[TVCalib worker] device={device}")
    if str(device).startswith("cuda"):
        print(f"[TVCalib worker] GPU={torch.cuda.get_device_name(0)}")

    first = cv2.imread(str(inputs[0]))
    if first is None:
        raise RuntimeError(f"Could not read {inputs[0]}")
    height, width = first.shape[:2]

    object3d = SoccerPitchLineCircleSegments(
        device=device,
        base_field=SoccerPitchSNCircleCentralSplit(),
    )

    results_by_original: dict[str, dict] = {
        str(p): {
            "input": str(p),
            "status": "error",
            "engine": "TVCalib",
            "pitch_length_m": PITCH_LENGTH_M,
            "pitch_width_m": PITCH_WIDTH_M,
            "coordinate_system": "top_left_origin_x_0_105_y_0_68_m",
        }
        for p in inputs
    }

    with tempfile.TemporaryDirectory(prefix="football_tvcalib_inputs_") as tmp:
        tmp_dir = Path(tmp)
        image_id_to_original: dict[str, Path] = {}

        for index, source in enumerate(inputs):
            target = tmp_dir / f"input_{index:06d}{source.suffix.lower() or '.jpg'}"
            shutil.copy2(source, target)
            image_id_to_original[target.name] = source

        print("[TVCalib worker] loading segmentation model...")
        model_seg = InferenceSegmentationModel(
            checkpoint,
            device,
        )

        dataset_seg = InferenceDatasetSegmentation(
            tmp_dir,
            width,
            height,
        )

        loader_seg = torch.utils.data.DataLoader(
            dataset_seg,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=custom_list_collate,
        )

        image_ids: list[str] = []
        keypoints_raw: list[dict] = []

        print("[TVCalib worker] segmenting pitch geometry...")
        for batch_dict in loader_seg:
            with torch.no_grad():
                sem_lines = model_seg.inference(
                    batch_dict["image"].to(device)
                )

            sem_lines = sem_lines.cpu().numpy().astype(np.uint8)

            for image_id, sem_line in zip(
                batch_dict["image_id"],
                sem_lines,
            ):
                skeleton = generate_class_synthesis(
                    sem_line,
                    radius=4,
                )
                points = get_line_extremities(
                    skeleton,
                    maxdist=30,
                    width=455,
                    height=256,
                    num_points_lines=4,
                    num_points_circles=8,
                )

                image_ids.append(str(image_id))
                keypoints_raw.append(points)

        if image_ids:
            model_calib = TVCalibModule(
                object3d,
                get_cam_distr(
                    1.96,
                    1,
                    1,
                ),
                None,
                (height, width),
                int(args.optim_steps),
                device,
                log_per_step=False,
                tqdm_kwqargs=None,
            )

            dataset_calib = InferenceDatasetCalibration(
                keypoints_raw,
                width,
                height,
                object3d,
            )

            loader_calib = torch.utils.data.DataLoader(
                dataset_calib,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                collate_fn=custom_list_collate,
            )

            per_sample_output = {}
            ordered_outputs = []

            for sample_index, x_dict in enumerate(loader_calib):
                image_id = image_ids[sample_index]

                try:
                    batch_sz = x_dict[
                        "lines__ndc_projected_selection_shuffled"
                    ].shape[0]

                    per_sample_loss, cam, _ = model_calib.self_optim_batch(
                        x_dict
                    )

                    output_dict = tensor2list(
                        detach_dict(
                            {
                                **cam.get_parameters(batch_sz),
                                **per_sample_loss,
                            }
                        )
                    )

                    sample = {}
                    for key, value in output_dict.items():
                        if isinstance(value, list) and len(value) == 1:
                            sample[key] = value[0]
                        else:
                            sample[key] = value

                    ordered_outputs.append((image_id, sample))

                except Exception as exc:
                    ordered_outputs.append(
                        (
                            image_id,
                            {"_error": f"{type(exc).__name__}: {exc}"},
                        )
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for image_id, sample in ordered_outputs:
                original = image_id_to_original.get(image_id)
                if original is None:
                    continue

                result = results_by_original[str(original)]

                if "_error" in sample:
                    result["error"] = sample["_error"]
                    continue

                try:
                    if "homography" not in sample:
                        raise RuntimeError(
                            "TVCalib camera output does not contain 'homography'."
                        )

                    loss = scalar_float(sample.get("loss_ndc_total"))
                    H_pitch = centered_homography_to_pitch(
                        sample["homography"]
                    )

                    self_verified = (
                        math.isfinite(loss)
                        and loss <= float(args.tau)
                    )

                    result.update(
                        {
                            "status": "ok",
                            "image_width": width,
                            "image_height": height,
                            "loss_ndc_total": loss,
                            "tau": float(args.tau),
                            "self_verified": bool(self_verified),
                            "quality_score": quality_from_loss(
                                loss,
                                float(args.tau),
                            ),
                            "homography_image_to_pitch": H_pitch.tolist(),
                        }
                    )

                    for field in (
                        "aov_radian",
                        "aov_degrees",
                        "position_meters",
                        "pan_degrees",
                        "tilt_degrees",
                        "roll_degrees",
                        "principal_point",
                        "x_focal_length",
                        "y_focal_length",
                    ):
                        if field in sample:
                            result[field] = jsonable(sample[field])

                except Exception as exc:
                    result["status"] = "error"
                    result["error"] = f"{type(exc).__name__}: {exc}"

    results = [results_by_original[str(p)] for p in inputs]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ok = sum(r.get("status") == "ok" for r in results)
    verified = sum(
        r.get("status") == "ok" and r.get("self_verified")
        for r in results
    )

    print(
        f"[TVCalib worker] finished: ok={ok}/{len(results)} | "
        f"self_verified={verified}/{len(results)} -> {output_path}"
    )


if __name__ == "__main__":
    main()
