"""Hybrid automatic-verification gate (hybrid accuracy work, 2026-11-11).

Runs LAST on every plate, after the existing rules (plate_decision.py), the existing ML verifier
(ocr_verifier.py, unchanged, ML_VERIFIER_CAN_DEMOTE stays False) and the review resolver. It can
only keep a SUCCESS or turn it into REVIEW (MANUAL_REVIEW_REQUIRED). It never promotes a plate,
never changes the recognised text and never invents characters. So no single signal - not the ML
verifier, not a valid format - can approve a plate on its own: a SUCCESS needs the existing rules
AND this gate.

Evidence checked for a SUCCESS candidate:
  1. Calibrated P(reading exactly correct): temperature-scaled Awiros CTC probabilities + Platt
     scaling (ocr_calibration.py), fitted on the separate labelled calibration split. Only used
     when the calibration file was fitted for the weights that are actually loaded (sha256 check);
     otherwise the gate reports CALIBRATION_UNAVAILABLE and leaves the rules' decision alone.
  2. Weakest temperature-scaled character probability.
  3. Selective test-time augmentation on the SAME crop: only when (1) is below the TTA gate,
     the chosen crop is re-read with a few OpenCV variants (one batched call). A variant that reads
     a different valid registration with confidence >= CONFLICT_MIN_CONF = TTA_DISAGREEMENT.
     No voting across frames, no averaging of raw scores.
  4. Expected state codes (site setting AUTO_APPROVE_STATE_CODES, e.g. PY/TN at the IOCL pumps):
     a reading from another state is not auto-approved (OUT_OF_REGION_STATE -> manual review).
     Most confident misreads in the labelled data turn a PY/TN plate into KA/MP/MH/PB.

All thresholds are chosen on the validation + calibration splits (bench/analyze_hybrid.py) and
stored in models/calibration/awiros_calibration.json together with the weights' sha256."""
import hashlib
import json
import logging
from pathlib import Path

import anpr_config as cfg
import image_enhance as ie
import indian_plate_validator as ipv
import ocr_calibration as oc

log = logging.getLogger("hybrid_verifier")

_SHA_CACHE = {}


def weights_sha256(path):
    path = Path(path)
    key = (str(path), path.stat().st_size, path.stat().st_mtime)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


class HybridVerifier:
    def __init__(self, reader, cal_path=None):
        self.reader = reader
        self.cal_path = Path(cal_path or cfg.HYBRID_CALIBRATION_PATH)
        self.cal, self.info = None, {}
        self.decoder = None
        if self.cal_path.is_file():
            info = json.loads(self.cal_path.read_text())
            want = info.get("weights_sha256")
            have = weights_sha256(reader.weights) if getattr(reader, "weights", None) else None
            if want and want == have:
                self.cal = oc.PlateCalibrator(info["temperature"], info["plate_w"], info["plate_b"])
                self.info = info
                self.decoder = oc.Decoder()
            else:
                log.warning("Calibration %s was fitted for other weights (%s); the hybrid gate runs "
                            "without calibrated probabilities", self.cal_path, (want or "?")[:12])
        self.auto_min_p = float(cfg.HYBRID_AUTO_MIN_P_OVERRIDE if cfg.HYBRID_AUTO_MIN_P_OVERRIDE is not None
                                else self.info.get("auto_threshold", cfg.HYBRID_AUTO_MIN_P))
        self.tta_gate = float(self.info.get("tta_gate", cfg.HYBRID_TTA_GATE))
        self.min_char_p = float(self.info.get("min_char_p", cfg.HYBRID_MIN_CHAR_P))
        self.tta_variants = tuple(self.info.get("tta_variants", cfg.HYBRID_TTA_VARIANTS))
        self.region_check = bool(self.info.get("region_check", True))

    @classmethod
    def load_default(cls, reader):
        if not cfg.HYBRID_VERIFIER_ENABLED or not hasattr(reader, "read_chars"):
            return None
        return cls(reader)

    def describe(self):
        return {"calibrated": self.cal is not None, "auto_min_p": self.auto_min_p,
                "tta_gate": self.tta_gate, "tta_variants": list(self.tta_variants),
                "min_char_p": self.min_char_p, "region_check": self.region_check, "expected_states": list(cfg.AUTO_APPROVE_STATE_CODES),
                "fitted_on": self.info.get("fitted_on", "")}

    # ------------------------------------------------------------------ evidence
    def calibrated(self, read):
        """(P(correct), weakest tempered char p) for one Awiros read, or (None, None)."""
        if self.cal is None or read.get("probs") is None:
            return None, None
        d = self.decoder.decode(read["probs"].astype("float32"), self.cal.T)
        if not d["chars"]:
            return 0.0, 0.0
        return self.cal.prob(d), d["weakest"]

    def tta(self, img, text):
        """Re-read the same crop with a few OpenCV variants. -> (n_views, n_agree, disagreeing)"""
        named = [(n, ie.apply(img, n)) for n in self.tta_variants]
        named = [(n, im) for n, im in named if im is not None and not ie.is_identity(im, img)]
        if not named:
            return 0, 0, []
        reads = self.reader.read_chars([im for _, im in named])
        agree, other = 0, []
        for (n, _), r in zip(named, reads):
            v = ipv.find_in_text(r["raw"])
            t = v["normalized_text"] if v["is_valid_format"] else ipv.normalize(r["raw"])
            if t == text:
                agree += 1
            elif v["is_valid_format"] and r["avg"] >= cfg.CONFLICT_MIN_CONF:
                other.append(f"{n}:{t}:{r['avg']:.2f}")
        return len(named), agree, other

    # ------------------------------------------------------------------ gate
    def apply(self, plate, chosen):
        """plate: dict from Pipeline.read_plate (after rules / ML verifier / resolver).
        chosen: the OCR attempt the reading came from ({"probs", "img", ...})."""
        plate.setdefault("hybrid_reasons", [])
        p_cal, weak_t = self.calibrated(chosen)
        plate["calibrated_probability"] = p_cal
        plate["tta_result"] = ""
        if plate["status"] != "SUCCESS":
            return plate
        reasons = []
        text = plate["final_text"] or plate["text"]
        if p_cal is None:
            plate["hybrid_reasons"] = ["CALIBRATION_UNAVAILABLE"]   # informational only
        else:
            if p_cal < self.auto_min_p:
                reasons.append("LOW_CALIBRATED_PROBABILITY")
            if weak_t is not None and weak_t < self.min_char_p:
                reasons.append("UNCERTAIN_CHARACTER_CALIBRATED")
        states = tuple(cfg.AUTO_APPROVE_STATE_CODES or ()) if self.region_check else ()
        if states and not text.startswith(states):
            reasons.append("OUT_OF_REGION_STATE")
        if self.tta_variants and (p_cal is None or p_cal < self.tta_gate) and chosen.get("img") is not None:
            n, agree, other = self.tta(chosen["img"], text)
            plate["tta_result"] = f"{agree}/{n} agree" + (f"; other: {', '.join(other)}" if other else "")
            if other:
                reasons.append("TTA_DISAGREEMENT")
        if reasons and not cfg.HYBRID_GATE_DEMOTES:
            plate["hybrid_reasons"] = [f"INFO_{r}" for r in reasons]   # logged only, status kept
        elif reasons:
            d = plate["decision"]
            d["reasons"] = list(d.get("reasons") or []) + reasons
            d.update(status="REVIEW", final_text="", manual_review_status="MANUAL_REVIEW_REQUIRED")
            plate.update(status="REVIEW", final_text="")
            plate["hybrid_reasons"] = reasons
        return plate
