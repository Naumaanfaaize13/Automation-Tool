"""ML verifier: a HistGradientBoostingClassifier that estimates P(reading is correct) for a
plate reading, from OCR / agreement / quality / detection features.

It only decides among readings that already passed the hard rules (valid format, read directly
without character swaps, no conflicting confident reading, usable crop). It can promote such a
reading to SUCCESS or hold it back for review. It never creates or changes plate text.

Trained by train_verifier.py -> models/ocr_verifier_hgb.joblib (+ .json with metrics).
This is separate from any BPCL transaction-verification model."""
import json
import math
from pathlib import Path

import anpr_config as cfg

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ocr_verifier_hgb.joblib"

FEATURES = [
    "avg", "weakest", "second_weakest", "mean_p", "n_chars", "n_below_50", "n_below_80",
    "direct_support", "corrected_support", "agreement", "n_candidates", "runner_up_avg",
    "ocr_attempts", "retry_count", "chosen_is_original", "two_line",
    "w_extracted", "w_no_series", "w_series_io", "w_legacy",
    "is_bh", "series_len", "rto_len", "text_len",
    "det_conf", "crop_mode", "box_aspect",
    "q_mean", "q_std", "q_sharpness", "q_noise", "q_height", "q_aspect", "q_tilt", "q_n_flags",
]

# reasons the ML verifier may overrule (soft evidence); every other reason blocks SUCCESS
SOFT_REASONS = {"LOW_AVERAGE_CONFIDENCE", "UNCERTAIN_CHARACTER", "EXTRACTED_FROM_LONGER_TEXT",
                "NO_SERIES_LETTERS", "LEGACY_STATE_CODE", "LOW_DETECTION_CONFIDENCE"}


def features(plate):
    """Feature dict for one plate dict from Pipeline.read_plate()."""
    d = plate["decision"]
    ps = sorted(c["p"] for c in plate["chars"]) or [0.0]
    cands = d.get("candidates") or []
    top = cands[0] if cands else {}
    v = d.get("validation") or {}
    w = set(v.get("warnings") or [])
    q = plate["quality"]
    att = max(1, plate["ocr_attempts"])
    nan = float("nan")
    return {
        "avg": plate["avg"], "weakest": plate["weakest"],
        "second_weakest": ps[1] if len(ps) > 1 else ps[0],
        "mean_p": sum(ps) / len(ps), "n_chars": len(plate["chars"]),
        "n_below_50": sum(p < 0.5 for p in ps), "n_below_80": sum(p < 0.8 for p in ps),
        "direct_support": top.get("direct_support", 0),
        "corrected_support": top.get("corrected_support", 0),
        "agreement": top.get("direct_support", 0) / att, "n_candidates": len(cands),
        "runner_up_avg": cands[1]["best_avg"] if len(cands) > 1 else 0.0,
        "ocr_attempts": plate["ocr_attempts"], "retry_count": plate["retry_count"],
        "chosen_is_original": float(plate["preprocessing"].startswith("original")),
        "two_line": float(plate.get("two_line", False)),
        "w_extracted": float("EXTRACTED_FROM_LONGER_TEXT" in w),
        "w_no_series": float("NO_SERIES_LETTERS" in w), "w_series_io": float("SERIES_HAS_I_OR_O" in w),
        "w_legacy": float("LEGACY_STATE_CODE" in w),
        "is_bh": float(v.get("format") == "BH"), "series_len": len(v.get("series") or ""),
        "rto_len": len(v.get("rto_code") or ""), "text_len": len(plate["text"] or ""),
        "det_conf": plate["det_conf"] if plate["det_conf"] is not None else nan,
        "crop_mode": float(plate.get("crop_mode", False)),
        "box_aspect": plate.get("box_aspect") if plate.get("box_aspect") is not None else nan,
        "q_mean": q.get("mean", 0), "q_std": q.get("std", 0), "q_sharpness": q.get("sharpness", 0),
        "q_noise": q.get("noise", 0), "q_height": q.get("height", 0), "q_aspect": q.get("aspect", 0),
        "q_tilt": q.get("tilt", 0), "q_n_flags": len(q.get("flags", [])),
    }


def eligible(decision):
    """Can the verifier decide this reading? Only valid, directly-read text whose remaining
    review reasons are all soft."""
    if decision["status"] not in ("SUCCESS", "REVIEW") or not decision.get("text"):
        return False
    if decision.get("corrected") or not (decision.get("validation") or {}).get("is_valid_format"):
        return False
    return set(decision.get("reasons", [])) <= SOFT_REASONS


class OCRVerifier:
    def __init__(self, path=MODEL_PATH):
        import joblib
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = float(bundle["threshold"])
        self.min_weakest = float(bundle.get("min_weakest", cfg.ML_MIN_WEAKEST_CHAR))
        self.info = bundle.get("info", {})
        self.path = str(path)

    @classmethod
    def load_default(cls):
        return cls(MODEL_PATH) if MODEL_PATH.exists() else None

    def probability(self, feat):
        import numpy as np
        x = np.array([[feat.get(f, float("nan")) for f in self.features]], dtype=float)
        return float(self.model.predict_proba(x)[0, 1])

    def apply(self, plate):
        """Update plate['decision'] in place. Sets plate['ml_probability'] (or None)."""
        d = plate["decision"]
        plate["ml_probability"] = None
        if not eligible(d):
            return d
        p = self.probability(features(plate))
        plate["ml_probability"] = p
        soft = [r for r in d["reasons"] if r in SOFT_REASONS]
        hard = [r for r in d["reasons"] if r not in SOFT_REASONS]
        if d["status"] == "SUCCESS" and not cfg.ML_VERIFIER_CAN_DEMOTE:
            d["ml_note"] = (f"rules accepted the reading; ML verifier p={p:.3f} recorded only "
                            f"(ML_VERIFIER_CAN_DEMOTE is off)")
            return d
        if p >= self.threshold and plate["weakest"] >= self.min_weakest and not hard:
            d.update(status="SUCCESS", final_text=d["text"], reasons=[], manual_review_status="",
                     ml_note=f"accepted by ML verifier (p={p:.3f}); rule notes: {', '.join(soft) or 'none'}")
        else:
            reasons = hard + soft
            if p < self.threshold:
                reasons.append("ML_VERIFIER_LOW_PROBABILITY")
            if plate["weakest"] < self.min_weakest:
                reasons.append("UNCERTAIN_CHARACTER") if "UNCERTAIN_CHARACTER" not in reasons else None
            d.update(status="REVIEW", final_text="", reasons=reasons,
                     manual_review_status="MANUAL_REVIEW_REQUIRED")
        return d

    def describe(self):
        return {"path": self.path, "threshold": self.threshold, "min_weakest": self.min_weakest,
                **{k: self.info[k] for k in ("trained_at", "train_rows", "test_metrics") if k in self.info}}


def _finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))
