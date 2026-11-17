"""Hybrid gate (hybrid_verifier.py) and the denormal flush in awiros_ocr.py. No model needed."""
import json

import numpy as np
import pytest

import anpr_config as cfg
import hybrid_verifier as hv
import ocr_calibration as oc


def probs_for(text, p=0.99, T=40):
    """A CTC probability matrix (T x 64) that greedy-decodes to `text` with confidence p."""
    dec = oc.Decoder()
    idx = {c: i for i, c in enumerate(dec.ctc_chars)}
    m = np.full((T, len(dec.ctc_chars)), (1 - p) / (len(dec.ctc_chars) - 1), np.float32)
    m[:, 0] = 0
    rows = []
    for ch in text:
        rows += [idx[ch], 0]
    rows += [0] * (T - len(rows))
    for t, c in enumerate(rows[:T]):
        m[t, :] = (1 - p) / (m.shape[1] - 1)
        m[t, c] = p
    return m.astype(np.float16)


class FakeReader:
    def __init__(self, weights, tta_reads=None):
        self.weights = weights
        self.tta_reads = tta_reads or []
        self.calls = 0

    def read_chars(self, imgs):
        self.calls += 1
        return [self.tta_reads[i % len(self.tta_reads)] for i in range(len(imgs))] if self.tta_reads else \
            [{"raw": "", "avg": 0.0} for _ in imgs]


def plate(text, status="SUCCESS"):
    d = {"status": status, "reasons": [], "final_text": text if status == "SUCCESS" else "",
         "manual_review_status": ""}
    return {"status": status, "text": text, "final_text": d["final_text"], "decision": d}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    w = tmp_path / "w.safetensors"
    w.write_bytes(b"weights")
    cal = tmp_path / "cal.json"

    def make(sha=None, **info):
        cal.write_text(json.dumps(dict({"temperature": 1.0, "plate_w": [10.0, 0.0, 0.0], "plate_b": 5.0,
                                        "weights_sha256": sha or hv.weights_sha256(w),
                                        "auto_threshold": 0.5, "min_char_p": 0.3, "tta_gate": 0.0,
                                        "tta_variants": [], "region_check": True}, **info)))
        return cal
    monkeypatch.setattr(cfg, "AUTO_APPROVE_STATE_CODES", ("PY", "TN"))
    monkeypatch.setattr(cfg, "HYBRID_GATE_DEMOTES", True)
    monkeypatch.setattr(cfg, "HYBRID_AUTO_MIN_P_OVERRIDE", None)
    return w, make


def test_confident_in_region_reading_stays_success(setup):
    w, make = setup
    v = hv.HybridVerifier(FakeReader(w), make())
    p = v.apply(plate("PY01CM1918"), {"probs": probs_for("PY01CM1918", 0.999), "img": None})
    assert p["status"] == "SUCCESS" and p["final_text"] == "PY01CM1918"
    assert p["calibrated_probability"] > 0.9


def test_low_calibrated_probability_goes_to_review(setup):
    w, make = setup
    v = hv.HybridVerifier(FakeReader(w), make(auto_threshold=0.99))
    p = v.apply(plate("PY01CM1918"), {"probs": probs_for("PY01CM1918", 0.9), "img": None})
    assert p["status"] == "REVIEW" and p["final_text"] == ""
    assert "LOW_CALIBRATED_PROBABILITY" in p["decision"]["reasons"]
    assert p["text"] == "PY01CM1918"            # text is never changed


def test_never_promotes_review(setup):
    w, make = setup
    v = hv.HybridVerifier(FakeReader(w), make())
    p = v.apply(plate("PY01CM1918", "REVIEW"), {"probs": probs_for("PY01CM1918", 0.999), "img": None})
    assert p["status"] == "REVIEW" and p["final_text"] == ""


def test_out_of_region_state_goes_to_review(setup):
    w, make = setup
    v = hv.HybridVerifier(FakeReader(w), make())
    p = v.apply(plate("KA01CM1918"), {"probs": probs_for("KA01CM1918", 0.999), "img": None})
    assert p["status"] == "REVIEW" and "OUT_OF_REGION_STATE" in p["hybrid_reasons"]


def test_tta_disagreement_goes_to_review(setup):
    w, make = setup
    other = {"raw": "PY01CM1916", "avg": 0.95}
    r = FakeReader(w, [other])
    v = hv.HybridVerifier(r, make(tta_gate=1.01, tta_variants=["clahe", "gaussian"]))
    img = (np.random.default_rng(0).random((60, 200, 3)) * 255).astype(np.uint8)
    p = v.apply(plate("PY01CM1918"), {"probs": probs_for("PY01CM1918", 0.999), "img": img})
    assert r.calls == 1                          # one batched call for all variants
    assert p["status"] == "REVIEW" and "TTA_DISAGREEMENT" in p["hybrid_reasons"]


def test_tta_skipped_when_confident(setup):
    w, make = setup
    r = FakeReader(w, [{"raw": "PY01CM1916", "avg": 0.95}])
    v = hv.HybridVerifier(r, make(tta_gate=0.5, tta_variants=["clahe"]))
    img = np.zeros((60, 200, 3), np.uint8)
    p = v.apply(plate("PY01CM1918"), {"probs": probs_for("PY01CM1918", 0.999), "img": img})
    assert r.calls == 0 and p["status"] == "SUCCESS"


def test_calibration_for_other_weights_is_not_used(setup):
    w, make = setup
    v = hv.HybridVerifier(FakeReader(w), make(sha="0" * 64, auto_threshold=0.999999))
    assert v.cal is None
    p = v.apply(plate("PY01CM1918"), {"probs": probs_for("PY01CM1918", 0.9), "img": None})
    assert p["status"] == "SUCCESS" and p["hybrid_reasons"] == ["CALIBRATION_UNAVAILABLE"]


def test_flush_denormals_only_touches_denormals():
    from awiros_ocr import flush_denormals
    tiny = np.finfo(np.float32).tiny
    a = np.array([0.0, 1e-39, -1e-40, tiny, 0.5, -2.0], np.float32)
    b = flush_denormals(a)
    assert b.tolist() == [0.0, 0.0, 0.0, float(tiny), 0.5, -2.0]
    assert a[1] != 0                             # input not modified
    i = np.array([1, 2], np.int64)
    assert flush_denormals(i) is i


def test_no_demotion_when_disabled(setup, monkeypatch):
    w, make = setup
    monkeypatch.setattr(cfg, "HYBRID_GATE_DEMOTES", False)
    v = hv.HybridVerifier(FakeReader(w), make(auto_threshold=0.99))
    p = v.apply(plate("PY01CM1918"), {"probs": probs_for("PY01CM1918", 0.9), "img": None})
    assert p["status"] == "SUCCESS" and p["final_text"] == "PY01CM1918"
    assert "INFO_LOW_CALIBRATED_PROBABILITY" in p["hybrid_reasons"]
    assert "LOW_CALIBRATED_PROBABILITY" not in p["decision"]["reasons"]
