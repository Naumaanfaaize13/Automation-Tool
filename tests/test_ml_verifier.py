import anpr_config as cfg
"""The ML verifier may only decide eligible readings and never changes plate text.
Uses a tiny HistGradientBoostingClassifier trained on synthetic features (no OCR models)."""
import csv
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard_log as dl  # noqa: E402
import ocr_verifier as ov  # noqa: E402
import plate_decision as pdec  # noqa: E402
from tests.test_anpr_rules import attempt  # noqa: E402


@pytest.fixture(scope="module")
def verifier(tmp_path_factory):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (400, len(ov.FEATURES)))
    wi = ov.FEATURES.index("weakest")
    y = (X[:, wi] > 0.5).astype(int)          # "correct" iff weakest char > 0.5
    m = HistGradientBoostingClassifier(max_iter=50).fit(X, y)
    path = tmp_path_factory.mktemp("m") / "v.joblib"
    joblib.dump({"model": m, "features": ov.FEATURES, "threshold": 0.5, "min_weakest": 0.3}, path)
    return ov.OCRVerifier(path)


def plate_from(attempts, **kw):
    d = pdec.decide(attempts, **kw)
    return {"decision": d, "status": d["status"], "text": d["text"], "chars": d["chosen"]["chars"],
            "avg": d["avg"], "weakest": d["weakest"], "ocr_attempts": len(attempts), "retry_count": 0,
            "preprocessing": "original", "det_conf": 0.8, "quality": {"flags": []}, "box_aspect": 4.0}


def test_promotes_uncertain_but_likely_correct(verifier):
    p = plate_from([attempt("MH12DE1433", low={9: (0.70, [("8", 0.25)])})])
    assert p["decision"]["status"] == pdec.REVIEW                  # rules alone: review
    verifier.apply(p)
    d = p["decision"]
    assert d["status"] == pdec.SUCCESS and d["final_text"] == "MH12DE1433" and p["ml_probability"] > 0.5
    assert p["text"] == "MH12DE1433"                                # text unchanged


def test_holds_back_low_probability(verifier):
    p = plate_from([attempt("MH12DE1433", low={9: (0.35, [("8", 0.30)])})])
    verifier.apply(p)
    d = p["decision"]
    assert d["status"] == pdec.REVIEW and d["final_text"] == ""
    assert "ML_VERIFIER_LOW_PROBABILITY" in d["reasons"]


def test_does_not_demote_rule_success_by_default(verifier):
    # trained on PARSeq readings: it must not overrule a reading that passed every rule
    p = plate_from([attempt("MH12DE1433", p=0.99)])
    assert p["decision"]["status"] == pdec.SUCCESS
    p["weakest"] = 0.45
    verifier.apply(p)
    assert p["decision"]["status"] == pdec.SUCCESS and p["ml_probability"] is not None


def test_can_demote_rule_success_when_enabled(verifier, monkeypatch):
    monkeypatch.setattr(cfg, "ML_VERIFIER_CAN_DEMOTE", True)
    p = plate_from([attempt("MH12DE1433", p=0.99)])
    assert p["decision"]["status"] == pdec.SUCCESS
    p["weakest"] = 0.45                                            # model says: likely wrong
    verifier.apply(p)
    assert p["decision"]["status"] == pdec.REVIEW


@pytest.mark.parametrize("make", [
    lambda: plate_from([attempt("TNO1AW5679", low={2: (0.55, [("0", 0.40)])})]),   # swapped chars
    lambda: plate_from([attempt("MH12DE1433"), attempt("MH12DE1438", variant="clahe")]),  # conflict
    lambda: plate_from([attempt("VEHICLE", p=0.999)]),                             # reject
    lambda: plate_from([attempt("2DE1433", p=0.999)]),                             # partial
])
def test_never_decides_ineligible(verifier, make):
    p = make()
    before = dict(p["decision"])
    verifier.apply(p)
    assert p["ml_probability"] is None
    assert p["decision"]["status"] == before["status"] != pdec.SUCCESS
    assert p["decision"]["final_text"] == ""


def test_floor_on_weakest_character(verifier):
    verifier.threshold = 0.0                                       # even if the model says yes
    p = plate_from([attempt("MH12DE1433", low={9: (0.25, [("8", 0.2)])})])
    verifier.apply(p)
    assert p["decision"]["status"] == pdec.REVIEW
    verifier.threshold = 0.5


def test_log_gains_new_columns_without_losing_rows(tmp_path, monkeypatch):
    r = tmp_path / "results"
    for k, v in dict(RESULTS=r, CROPS=r / "crops", THUMBS=r / "thumbs", ANNOT=r / "annotated",
                     ENHANCED=r / "enhanced", LOG=r / "log.csv", AUDIT=r / "corrections.csv").items():
        monkeypatch.setattr(dl, k, v)
    r.mkdir()
    older = [f for f in dl.FIELDS if f not in ("ml_probability", "rule_status")]
    with open(dl.LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=older)
        w.writeheader()
        w.writerow({k: "" for k in older} | {"row_id": "abc", "plate_text": "TN01AB1234", "status": "SUCCESS",
                                            "ocr_conf": "0.9", "weakest": "0.9", "batch": "b", "thumb_file": "t"})
    rows = dl.load_rows()
    assert len(rows) == 1 and rows[0]["row_id"] == "abc" and "ml_probability" in rows[0]
    assert list(r.glob("log_backup_*.csv"))
