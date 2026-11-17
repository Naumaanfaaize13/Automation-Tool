"""Phase 2: PaddleOCR DET / CLS / REC models are present, unchanged, and load in the installed
PaddlePaddle with the per-model oneDNN settings from anpr_config. Skipped when the models have not
been downloaded (python src/setup_paddleocr_models.py)."""
import anpr_config as _cfg_gate
import pytest as _pytest_gate
pytestmark = _pytest_gate.mark.skipif(not _cfg_gate.PADDLE_OCR_ENABLED,
                                      reason="PaddleOCR second engine is switched off (hybrid plan, 2026-11-11)")

import json

import numpy as np
import pytest

import anpr_config as cfg
import setup_paddleocr_models as setup

MODELS = {"det": (cfg.PADDLE_DET_MODEL, (1, 3, 96, 320)),
          "cls": (cfg.PADDLE_CLS_MODEL, (1, 3, 80, 160)),
          "rec": (cfg.PADDLE_REC_MODEL, (1, 3, 48, 320))}

pytestmark = pytest.mark.skipif(not all(setup.model_ok(d) for d, _ in MODELS.values()),
                                reason="PaddleOCR models not downloaded")


def test_runtime_compatible():
    info, problems = setup.check_runtime()
    assert problems == [], problems


def test_models_match_manifest():
    manifest = json.loads((cfg.PADDLEOCR_MODEL_DIR / "manifest.json").read_text())
    assert set(manifest) == set(setup.MODELS.values())
    for name, entry in manifest.items():
        for f, h in entry["sha256"].items():
            assert setup.sha256(cfg.PADDLEOCR_MODEL_DIR / name / f) == h, f"{name}/{f} changed"


@pytest.mark.parametrize("role", ["det", "cls", "rec"])
def test_model_runs(role):
    from paddle import inference
    folder, shape = MODELS[role]
    c = inference.Config(str(folder / "inference.json"), str(folder / "inference.pdiparams"))
    c.disable_gpu()
    c.set_cpu_math_library_num_threads(cfg.PADDLE_CPU_THREADS)
    c.disable_glog_info()
    c.enable_mkldnn() if cfg.PADDLE_ONEDNN[role] else c.disable_mkldnn()
    p = inference.create_predictor(c)
    h = p.get_input_handle(p.get_input_names()[0])
    h.reshape(shape)
    h.copy_from_cpu(np.random.rand(*shape).astype("float32"))
    p.run()
    out = p.get_output_handle(p.get_output_names()[0]).copy_to_cpu()
    expected = {"det": (1, 1, 96, 320), "cls": (1, 2), "rec": (1, 40, 438)}[role]
    assert out.shape == expected
