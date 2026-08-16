from __future__ import annotations

from collections.abc import Iterable


def _accepted_pnl(record: dict) -> bool:
    if record.get("status") != "ok":
        return False

    explicit_flags = [
        record.get("accepted_for_v24"),
        record.get("accepted_for_v23"),
        record.get("accepted"),
    ]
    explicit = [x for x in explicit_flags if x is not None]

    return bool(explicit[0]) if explicit else True


def _accepted_tv(record: dict) -> bool:
    return (
        record.get("status") == "ok"
        and bool(record.get("self_verified", False))
        and record.get("homography_image_to_pitch") is not None
    )


def build_fused_geometry(
    frame_indices: Iterable[int],
    pnl_records: Iterable[dict],
    tv_records: Iterable[dict],
) -> list[dict]:
    """
    PnLCalib has strict precedence. TVCalib is used only when PnLCalib
    did not yield an accepted geometry for the same frame.
    """
    pnl_by_frame = {
        int(r["frame_index"]): r
        for r in pnl_records
        if "frame_index" in r
    }
    tv_by_frame = {
        int(r["frame_index"]): r
        for r in tv_records
        if "frame_index" in r
    }

    fused = []

    for frame_index in sorted(set(int(x) for x in frame_indices)):
        pnl = pnl_by_frame.get(frame_index)
        tv = tv_by_frame.get(frame_index)

        if pnl is not None and _accepted_pnl(pnl):
            fused.append(
                {
                    "frame_index": frame_index,
                    "status": "ok",
                    "engine": "PnLCalib",
                    "homography_image_to_pitch": pnl[
                        "homography_image_to_pitch"
                    ],
                    "quality_score": float(
                        pnl.get("quality_score", 0.0) or 0.0
                    ),
                    "source": {
                        "rep_err": pnl.get("rep_err"),
                        "quality_score": pnl.get("quality_score"),
                    },
                }
            )
            continue

        if tv is not None and _accepted_tv(tv):
            fused.append(
                {
                    "frame_index": frame_index,
                    "status": "ok",
                    "engine": "TVCalib",
                    "homography_image_to_pitch": tv[
                        "homography_image_to_pitch"
                    ],
                    "quality_score": float(
                        tv.get("quality_score", 0.0) or 0.0
                    ),
                    "source": {
                        "loss_ndc_total": tv.get("loss_ndc_total"),
                        "tau": tv.get("tau"),
                        "self_verified": tv.get("self_verified"),
                    },
                }
            )
            continue

        fused.append(
            {
                "frame_index": frame_index,
                "status": "missing",
                "engine": None,
                "homography_image_to_pitch": None,
                "quality_score": 0.0,
                "source": {
                    "pnl_status": pnl.get("status") if pnl else None,
                    "tv_status": tv.get("status") if tv else None,
                    "tv_self_verified": (
                        tv.get("self_verified") if tv else None
                    ),
                },
            }
        )

    return fused
