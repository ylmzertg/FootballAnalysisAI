from core.calibration_fusion import build_fused_geometry


H_PNL = [[1, 0, 1], [0, 1, 2], [0, 0, 1]]
H_TV = [[2, 0, 3], [0, 2, 4], [0, 0, 1]]


def test_pnl_has_priority_over_tvcalib():
    pnl = [
        {
            "frame_index": 0,
            "status": "ok",
            "accepted_for_v23": True,
            "homography_image_to_pitch": H_PNL,
            "quality_score": 0.7,
        }
    ]
    tv = [
        {
            "frame_index": 0,
            "status": "ok",
            "self_verified": True,
            "homography_image_to_pitch": H_TV,
            "quality_score": 0.8,
        }
    ]

    fused = build_fused_geometry([0], pnl, tv)

    assert fused[0]["engine"] == "PnLCalib"
    assert fused[0]["homography_image_to_pitch"] == H_PNL


def test_tvcalib_fills_only_pnl_missing_frame():
    pnl = [
        {
            "frame_index": 0,
            "status": "error",
        }
    ]
    tv = [
        {
            "frame_index": 0,
            "status": "ok",
            "self_verified": True,
            "homography_image_to_pitch": H_TV,
            "quality_score": 0.8,
            "loss_ndc_total": 0.010,
            "tau": 0.017,
        }
    ]

    fused = build_fused_geometry([0], pnl, tv)

    assert fused[0]["engine"] == "TVCalib"
    assert fused[0]["homography_image_to_pitch"] == H_TV


def test_unverified_tvcalib_is_not_used():
    pnl = [{"frame_index": 0, "status": "error"}]
    tv = [
        {
            "frame_index": 0,
            "status": "ok",
            "self_verified": False,
            "homography_image_to_pitch": H_TV,
        }
    ]

    fused = build_fused_geometry([0], pnl, tv)

    assert fused[0]["status"] == "missing"
    assert fused[0]["engine"] is None
