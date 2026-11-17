"""Review resolver: a second, independent check for plates the first pass sent to REVIEW.

Layers (all evidence is real model output; no text is invented or edited):
  1. YOLO + whole-image reads   - the first pass already read the box crops, wider crops,
                                  enhanced variants and (automatic crop mode) the whole image.
  2. Awiros vote                - how many of those reads give each valid registration, how
                                  confident, and how far ahead the leader is.
  3. PARSeq cross-check         - a second OCR model with a different architecture and training
                                  data reads the same crops. Two independent models agreeing on
                                  all ~10 characters is strong evidence; they rarely make the
                                  same mistake.
  4. ML (HistGradientBoosting)  - turns these signals into P(reading is correct). A REVIEW plate
                                  becomes SUCCESS only if P >= a threshold chosen on held-out
                                  labelled plates for a target precision (train_resolver.py).

Hard limits that the model can never overrule: the text must be a valid registration that was
read directly by Awiros (no character swaps), no character below ML_MIN_WEAKEST_CHAR, and the
crop must be usable. Everything else stays MANUAL_REVIEW_REQUIRED.

Model: models/review_resolver_hgb.joblib (+ .json with the held-out metrics).
"""
import json
from collections import defaultdict
from pathlib import Path

import anpr_config as cfg
import indian_plate_validator as ipv

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "review_resolver_hgb.joblib"
RESOLVED_BY = "REVIEW_RESOLVER"

# reasons that always keep a plate in review (the resolver may only weigh the others)
BLOCKING = {"CHARACTERS_SUBSTITUTED_FROM_MODEL_ALTERNATIVES", "POOR_IMAGE_QUALITY",
            "UNUSUAL_PLATE_SHAPE", "PARTIAL_OR_INVALID_PLATE_TEXT", "UNUSABLE_CROP",
            "NOT_A_REGISTRATION_NUMBER"}
REASON_FLAGS = ["LOW_AVERAGE_CONFIDENCE", "UNCERTAIN_CHARACTER", "CONFLICTING_READINGS",
                "LOW_DETECTION_CONFIDENCE", "LEGACY_STATE_CODE", "NO_SERIES_LETTERS",
                "EXTRACTED_FROM_LONGER_TEXT", "SERIES_HAS_I_OR_O", "ML_VERIFIER_LOW_PROBABILITY"]
ENHANCED_PREFIX = ("adaptive", "otsu", "gray", "clahe", "sharpen", "denoise", "low_light",
                   "upscale", "deskew", "perspective")

FEATURES = None   # set below from feature_names()


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def eligible(plate):
    """Only REVIEW plates with a valid, directly-read registration and no blocking reason."""
    d = plate["decision"]
    if plate["status"] != "REVIEW" or d.get("corrected"):
        return False
    if not (d.get("validation") or {}).get("is_valid_format"):
        return False
    if set(d.get("reasons") or []) & BLOCKING:
        return False
    return plate["weakest"] >= cfg.ML_MIN_WEAKEST_CHAR


def vote(attempts, winner):
    """Awiros vote over all first-pass reads."""
    g = defaultdict(lambda: {"n": 0, "avg": 0.0, "sum": 0.0, "warn": True, "plain": 0})
    n_valid = 0
    for a in attempts:
        v = ipv.find_in_text(a["raw"])
        if not v["is_valid_format"]:
            continue
        n_valid += 1
        e = g[v["normalized_text"]]
        e["n"] += 1
        e["avg"] = max(e["avg"], a["avg"])
        e["sum"] += a["avg"]
        e["warn"] = e["warn"] and bool(v["warnings"])
        e["plain"] += not str(a.get("variant", "")).startswith(ENHANCED_PREFIX)
    w = g.get(winner, {"n": 0, "avg": 0.0, "sum": 0.0, "warn": False, "plain": 0})
    rivals = sorted(((t, e) for t, e in g.items() if t != winner), key=lambda te: -te[1]["sum"])
    r = rivals[0][1] if rivals else {"n": 0, "avg": 0.0, "sum": 0.0, "warn": False, "plain": 0}
    strong = [e for _, e in rivals if e["avg"] >= cfg.CONFLICT_MIN_CONF]
    return {
        "v_share": w["n"] / max(n_valid, 1), "v_support": w["n"], "v_plain_support": w["plain"],
        "v_sum": w["sum"], "v_rival_support": r["n"], "v_rival_sum": r["sum"],
        "v_rival_avg": r["avg"], "v_rival_warn": float(bool(rivals) and r["warn"]),
        "v_margin": w["sum"] - r["sum"], "v_n_texts": len(g), "v_strong_rivals": len(strong),
        "v_strong_rivals_clean": sum(not e["warn"] for e in strong),
        "v_rival_dist": float(lev(winner, rivals[0][0])) if rivals else 0.0,
    }


def parseq_evidence(reads, winner):
    """PARSeq reads of the same crops vs the Awiros winner."""
    texts = []
    for r in reads:
        v = ipv.find_in_text(r.get("raw", ""))
        t = v["normalized_text"] if v["is_valid_format"] else ipv.normalize(r.get("raw", ""))
        texts.append((t, v["is_valid_format"], r.get("avg", 0.0)))
    exact = [t == winner for t, _, _ in texts]
    sims = [1.0 - lev(t, winner) / max(len(winner), 1) for t, _, _ in texts] or [0.0]
    agree_conf = [c for (t, _, c) in texts if t == winner]
    other_valid = [t for t, ok, c in texts if ok and t != winner and c >= cfg.CONFLICT_MIN_CONF]
    return {
        "p_n": len(texts), "p_exact": float(sum(exact)), "p_any_exact": float(any(exact)),
        "p_best_sim": max(sims), "p_mean_sim": sum(sims) / len(sims),
        "p_agree_conf": max(agree_conf) if agree_conf else 0.0,
        "p_other_valid_confident": float(len(other_valid)),
    }


def features(plate, attempts, parseq_reads):
    from ocr_verifier import features as base
    f = dict(base(plate))
    winner = plate["text"] or ""
    f.update(vote(attempts, winner))
    f.update(parseq_evidence(parseq_reads, winner))
    reasons = set(plate["decision"].get("reasons") or [])
    f.update({f"r_{r.lower()}": float(r in reasons) for r in REASON_FLAGS})
    return f


def feature_names():
    from ocr_verifier import FEATURES as BASE
    v = ["v_share", "v_support", "v_plain_support", "v_sum", "v_rival_support", "v_rival_sum",
         "v_rival_avg", "v_rival_warn", "v_margin", "v_n_texts", "v_strong_rivals",
         "v_strong_rivals_clean", "v_rival_dist"]
    p = ["p_n", "p_exact", "p_any_exact", "p_best_sim", "p_mean_sim", "p_agree_conf",
         "p_other_valid_confident"]
    return list(BASE) + v + p + [f"r_{r.lower()}" for r in REASON_FLAGS]


FEATURES = feature_names()


class ReviewResolver:
    def __init__(self, model=None, threshold=None, meta=None, device="cpu"):
        self.model, self.threshold, self.meta = model, threshold, meta or {}
        self.device = device
        self._parseq = None
        self.collect = None          # list -> collect features instead of deciding

    @classmethod
    def load_default(cls, device="cpu", collect_only=False):
        """collect_only: no model needed (train_resolver.py uses it to compute features)."""
        if collect_only:
            return cls(device=device)
        if not MODEL_PATH.is_file():
            return None
        import joblib
        obj = joblib.load(MODEL_PATH)
        meta = {}
        js = MODEL_PATH.with_suffix(".json")
        if js.is_file():
            meta = json.loads(js.read_text())
        return cls(obj["model"], obj["threshold"], meta, device)

    @property
    def parseq(self):
        if self._parseq is None:
            from plate_reader import PARSeqReader
            self._parseq = PARSeqReader("parseq", self.device)
        return self._parseq

    def describe(self):
        return {"model": "HistGradientBoostingClassifier", "threshold": self.threshold,
                "second_ocr": "PARSeq", "heldout": self.meta.get("heldout")}

    def evidence(self, plate, attempts, views):
        """views: list of BGR crops of the plate (un-enhanced) for the PARSeq cross-check."""
        reads = self.parseq.read_plate_chars([v for v in views if v is not None and v.size])
        return features(plate, attempts, reads)

    def apply(self, plate, attempts, views):
        """Promote an eligible REVIEW plate to SUCCESS when P(correct) >= threshold.
        Sets plate['resolver_probability'] whenever it was evaluated."""
        if not eligible(plate):
            return plate
        if self.model is None:
            if self.collect is not None:             # train_resolver.py: features only
                self.collect.append({"text": plate["text"],
                                     "features": self.evidence(plate, attempts, views)})
            return plate
        f = self.evidence(plate, attempts, views)
        import numpy as np
        x = np.array([[f.get(k, float("nan")) for k in FEATURES]], dtype=float)
        p = float(self.model.predict_proba(x)[0, 1])
        plate["resolver_probability"] = p
        d = plate["decision"]
        if p >= self.threshold:
            d["resolved_reasons"] = list(d.get("reasons") or [])
            d.update(status="SUCCESS", final_text=plate["text"], manual_review_status="",
                     resolved_by=RESOLVED_BY, reasons=[])
            plate.update(status="SUCCESS", final_text=plate["text"])
        return plate
