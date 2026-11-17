"""Phase 3: PaddleOCR DET (PP-OCRv5_mobile_det) on plate crops.
Model tests are skipped when the models are not downloaded (setup_paddleocr_models.py)."""
import anpr_config as _cfg_gate
import pytest as _pytest_gate
pytestmark = _pytest_gate.mark.skipif(not _cfg_gate.PADDLE_OCR_ENABLED,
                                      reason="PaddleOCR second engine is switched off (hybrid plan, 2026-11-11)")

import cv2
import numpy as np
import pytest

import anpr_config as cfg
import paddle_ocr_engine as poe
import setup_paddleocr_models as setup

needs_models = pytest.mark.skipif(not setup.model_ok(cfg.PADDLE_DET_MODEL),
                                  reason="PaddleOCR DET model not downloaded")


def plate(lines, w=520, line_h=110):
    """White plate with black text lines, like a tight pump-camera crop."""
    img = np.full((line_h * len(lines) + 20, w, 3), 235, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (18, 10 + line_h * (i + 1) - 22), cv2.FONT_HERSHEY_SIMPLEX, 2.6,
                    (20, 20, 20), 7, cv2.LINE_AA)
    return img


def box(cx, cy, w=100, h=30):
    return {"center": [cx, cy], "height": h, "width": w}


# ---------------------------------------------------------------- pure functions
def test_order_lines_top_to_bottom_then_left_to_right():
    a, b, c = box(300, 40), box(80, 45), box(150, 120)     # a and b on one row, c below
    assert poe.order_lines([c, a, b]) == [b, a, c]
    assert poe.order_lines([]) == []


def test_crop_quad_rectifies_and_rotates_tall_lines():
    img = np.zeros((100, 200, 3), np.uint8)
    out = poe.crop_quad(img, [[10, 10], [110, 10], [110, 40], [10, 40]])
    assert out.shape[:2] == (30, 100)
    tall = poe.crop_quad(img, [[10, 10], [40, 10], [40, 90], [10, 90]])
    assert tall.shape[1] > tall.shape[0]                    # turned to a horizontal line


def test_resolve_device_falls_back_to_cpu():
    assert poe.resolve_device("cpu") == "cpu"
    import paddle
    if not (paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count()):
        assert poe.resolve_device("cuda") == "cpu"


def test_number_parses_yml_fractions():
    assert poe._number("1./255.") == pytest.approx(1 / 255.0)
    assert poe._number(0.5) == 0.5


def test_missing_model_is_reported(tmp_path):
    with pytest.raises(poe.PaddleOCRUnavailable):
        poe.load_model(tmp_path, "det", "cpu")


# ---------------------------------------------------------------- model
@pytest.fixture(scope="module")
def det():
    return poe.PaddleDet("cpu")


@needs_models
def test_single_row_plate_gives_one_line_covering_the_text(det):
    img = plate(["MH12RK5441"])
    lines = det.detect(img)
    assert len(lines) == 1
    xs = [p[0] for p in lines[0]["points"]]
    assert (max(xs) - min(xs)) / img.shape[1] > 0.8         # no characters cut off
    assert 0.0 < lines[0]["score"] <= 1.0


@needs_models
def test_dual_row_plate_gives_two_lines_in_reading_order(det):
    lines = det.detect(plate(["TN11", "AE6414"], w=420))
    assert len(lines) == 2
    assert lines[0]["center"][1] < lines[1]["center"][1]


@needs_models
def test_blank_and_empty_inputs_give_no_lines(det):
    assert det.detect(np.full((120, 400, 3), 200, np.uint8)) == []
    assert det.detect(np.zeros((0, 0, 3), np.uint8)) == []


@needs_models
def test_boxes_stay_inside_the_crop(det):
    img = plate(["PY01CX7009"])
    for ln in det.detect(img):
        for x, y in ln["points"]:
            assert 0 <= x < img.shape[1] and 0 <= y < img.shape[0]


@needs_models
def test_engine_detect_lines_returns_crops_and_timing():
    eng = poe.PaddleOCREngine("cpu")
    lines, crops, ms = eng.detect_lines(plate(["KA01AB1234"]))
    assert len(crops) == len(lines) >= 1 and all(c.size for c in crops)
    assert ms > 0
    assert eng.describe()["det"] == "PP-OCRv5_mobile_det" and eng.describe()["device"] == "cpu"
