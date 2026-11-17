"""Fit + select the hybrid gate for one Awiros model and report the final comparison.

    python bench/analyze_hybrid.py --tag ft --weights models/awiros_finetuned/model.safetensors

Inputs: bench/cache/<split>.pkl (original model) and bench/cache/<tag>/<split>.pkl (the model to
deploy), both made by bench/cache_reads.py with the same production pipeline.

  * Temperature + Platt calibration: fitted on the CALIBRATION split only.
  * Gate thresholds (calibrated P, weakest char, TTA gate/variants, region check): chosen on
    calibration + validation - maximum coverage with zero false acceptances there.
  * Model selection happened on validation (finetune scripts). TEST is only read for the report.

Ground truth = the human-audited manifest label. Images whose label was visually confirmed to
belong to a different vehicle (LABEL_MISMATCH below) are excluded from threshold SELECTION only;
the test report gives both numbers (all images, and without the confirmed label mismatches).

Writes models/calibration/awiros_calibration.json (with the weights' sha256),
bench/reports/hybrid_benchmark.json and bench/reports/failures_test_final.csv."""
import argparse
import collections
import csv
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/
sys.path.insert(0, str(ROOT / "bench"))
import anpr_config as cfg  # noqa: E402
import indian_plate_validator as ipv  # noqa: E402
import ocr_calibration as oc  # noqa: E402
from analyze import lev, text_metrics, decision_metrics  # noqa: E402

CACHE = ROOT / "bench" / "cache"
REP = ROOT / "bench" / "reports"
DEC = oc.Decoder()

# Checked by eye on 2026-11-11 (bench/reports/suspected_label_errors.csv): the image clearly
# shows a different registration than the manifest label. Not corrected - reported.
LABEL_MISMATCH = {
    "380398__plate__day__bike__correct__ab7e6cd5.jpg",    # val:   label PY01CY7642, plate PY01CM1918
    "377087__plate__day__bike__correct__db2821ef.jpg",    # calib: label PY01DC4523, plate ...C4573
    "380191__plate__day__bike__correct__6226366d.jpg",    # calib: label PY01CH7182, plate PY01BF4088
    "377040__plate__day__bike__correct__fc13d44c.jpg",    # test:  label PY01DF5367, plate PY05VH0425
    "378808__plate__night__bike__correct__935b6b7a.jpg",  # test:  label PY01CR1476, plate PY05E5875
    "386563__plate__day__car__correct__90e306ac.jpg",     # test:  label PY01BS3628, plate TN09DA3638
}


def load(tag, split):
    p = (CACHE / tag if tag else CACHE) / f"{split}.pkl"
    return pickle.load(open(p, "rb")) if p.is_file() else []


def name(r):
    return Path(r["path"]).name


def whole_text(r):
    d = DEC.decode(r["views"]["whole"].astype(np.float32))
    v = ipv.find_in_text(d["raw"])
    return v["normalized_text"] if v["is_valid_format"] else ipv.normalize(d["raw"])


def evidence(r, cal):
    """Everything the gate needs for one cached image (production reading of the model)."""
    p = r["prod"]
    if p is None:
        return None
    e = {"text": p["text"], "rule_success": p["status"] == "SUCCESS", "p_cal": None, "weak_t": None}
    if p.get("chosen_probs") is not None and cal is not None:
        d = DEC.decode(p["chosen_probs"].astype(np.float32), cal.T)
        e["p_cal"] = cal.prob(d) if d["chars"] else 0.0
        e["weak_t"] = d["weakest"] if d["chars"] else 0.0
    tta = {}
    for n, x in (p.get("tta") or {}).items():
        v = ipv.find_in_text(x["raw"])
        tta[n] = (v["normalized_text"] if v["is_valid_format"] else ipv.normalize(x["raw"]),
                  bool(v["is_valid_format"]), x["avg"])
    e["tta"] = tta
    return e


def gate(e, s):
    """Simulation of HybridVerifier.apply. -> (auto, reasons)"""
    if e is None or not e["rule_success"]:
        return False, ["RULES"]
    reasons = []
    if e["p_cal"] is not None:
        if e["p_cal"] < s["auto_threshold"]:
            reasons.append("LOW_CALIBRATED_PROBABILITY")
        if e["weak_t"] < s["min_char_p"]:
            reasons.append("UNCERTAIN_CHARACTER_CALIBRATED")
    if s["region"] and not e["text"].startswith(tuple(cfg.AUTO_APPROVE_STATE_CODES)):
        reasons.append("OUT_OF_REGION_STATE")
    if s["tta_variants"] and (e["p_cal"] is None or e["p_cal"] < s["tta_gate"]):
        for n in s["tta_variants"]:
            if n in e["tta"]:
                t, valid, avg = e["tta"][n]
                if t != e["text"] and valid and avg >= cfg.CONFLICT_MIN_CONF:
                    reasons.append("TTA_DISAGREEMENT")
                    break
    return not reasons, reasons


def summarize(rs, preds, auto):
    labels = [r["label"] for r in rs]
    m = text_metrics(preds, labels)
    m.update(decision_metrics(preds, labels, auto))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", required=True)
    a = ap.parse_args()
    from hybrid_verifier import weights_sha256
    W = Path(a.weights).resolve()

    base = {s: load("", s) for s in ("val", "calib", "test")}
    new = {s: load(a.tag, s) for s in ("val", "calib", "test")}
    for s in new:
        assert [name(r) for r in new[s]] == [name(r) for r in base[s]], f"{s}: different images"
    out = {"weights": str(W), "weights_sha256": weights_sha256(W),
           "splits": {s: len(v) for s, v in new.items()}}

    # ---- calibration (calibration split only)
    cal_rs = [r for r in new["calib"] if r["prod"] is not None and r["prod"].get("chosen_probs") is not None]
    probs = [r["prod"]["chosen_probs"].astype(np.float32) for r in cal_rs]
    T = oc.fit_temperature(DEC, probs, [r["label"] for r in cal_rs])
    dec = [DEC.decode(p, T) for p in probs]
    cal = oc.PlateCalibrator.fit(dec, [r["prod"]["text"] == r["label"] for r in cal_rs], T)
    out["calibration"] = {"temperature": T, "plate_w": cal.w, "plate_b": cal.b, "n_fit": len(cal_rs)}

    ev = {s: [evidence(r, cal) for r in new[s]] for s in new}

    # calibration quality on val / test (raw = product of raw char probs, i.e. P(all chars right))
    for s in ("val", "test"):
        y, raw, calp, cp_raw, cp_t = [], [], [], [], []
        for r, e in zip(new[s], ev[s]):
            if e is None or e["p_cal"] is None:
                continue
            y.append(e["text"] == r["label"])
            raw.append(float(np.prod([c[1] for c in r["prod"]["chars"]])) if r["prod"]["chars"] else 0.0)
            calp.append(e["p_cal"])
        pr = [r["prod"]["chosen_probs"].astype(np.float32) for r in new[s] if r["prod"] and r["prod"].get("chosen_probs") is not None]
        lb = [r["label"] for r in new[s] if r["prod"] and r["prod"].get("chosen_probs") is not None]
        c_raw, c_t = oc.char_pairs(DEC, pr, lb, 1.0), oc.char_pairs(DEC, pr, lb, T)
        out["calibration"][s] = {
            "plate_uncalibrated": {"ece": round(oc.ece(raw, y), 4), "brier": round(oc.brier(raw, y), 4), "nll": round(oc.nll(raw, y), 4)},
            "plate_calibrated": {"ece": round(oc.ece(calp, y), 4), "brier": round(oc.brier(calp, y), 4), "nll": round(oc.nll(calp, y), 4)},
            "char_uncalibrated": {"ece": round(oc.ece(*c_raw), 4), "nll": round(oc.nll(*c_raw), 4), "n_chars": int(len(c_raw[0]))},
            "char_temperature_scaled": {"ece": round(oc.ece(*c_t), 4), "nll": round(oc.nll(*c_t), 4)},
            "reliability_plate_calibrated": oc.reliability(calp, y),
            "reliability_plate_uncalibrated": oc.reliability(raw, y)}

    # ---- gate selection on calib + val (confirmed label mismatches excluded from selection)
    pool = [(r, e) for s in ("calib", "val") for r, e in zip(new[s], ev[s]) if name(r) not in LABEL_MISMATCH]
    var_sets = {"none": (), "clahe": ("clahe",), "default": tuple(cfg.HYBRID_TTA_VARIANTS),
                "photometric": ("clahe", "clahe_bilateral", "gaussian", "bilateral", "sharpen", "low_light"),
                "all": ("clahe", "clahe_bilateral", "gaussian", "bilateral", "sharpen", "low_light",
                        "deskew", "perspective", "resize")}
    grid = []
    for thr, mc, tg, vs, reg in itertools.product(
            [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99],
            [0.0, 0.3, 0.5, 0.7, 0.8], [0.9, 0.95, 0.98, 1.01], list(var_sets), [False, True]):
        s = {"auto_threshold": thr, "min_char_p": mc, "tta_gate": tg, "tta_variants": var_sets[vs],
             "tta_set": vs, "region": reg}
        auto = [gate(e, s)[0] for _, e in pool]
        ok = [e is not None and e["text"] == r["label"] for r, e in pool]
        na, fa = sum(auto), sum(x and not c for x, c in zip(auto, ok))
        n_tta = sum(1 for _, e in pool if e and e["rule_success"] and (e["p_cal"] or 0) < tg) if var_sets[vs] else 0
        grid.append((fa, -na, n_tta, thr, mc, tg, vs, reg, s))
    zero = [g for g in grid if g[0] == 0 and g[1] < 0]
    # max coverage with 0 false accepts; ties -> fewer TTA reads, then the stricter settings
    best = min(zero, key=lambda g: (g[1], g[2], -g[3], -g[4])) if zero else min(grid)
    S = best[-1]
    no_gate = {"auto_threshold": 0.0, "min_char_p": 0.0, "tta_gate": 0.0, "tta_variants": (), "tta_set": "none", "region": False}
    out["gate_selection"] = {
        "rule": "max auto-approvals with zero false acceptances on calib+val (confirmed label mismatches excluded)",
        "chosen": {k: v for k, v in S.items()}, "pool_size": len(pool),
        "pool_auto": -best[1], "pool_false_accepts": best[0],
        "pool_rules_only": {"auto": sum(gate(e, no_gate)[0] for _, e in pool),
                            "false_accepts": sum(gate(e, no_gate)[0] and e["text"] != r["label"] for r, e in pool)}}

    # ---- final comparison on the untouched TEST split
    res = {}
    rs_b, rs_n = base["test"], new["test"]
    keep = [i for i, r in enumerate(rs_n) if name(r) not in LABEL_MISMATCH]

    def both(key, rs, preds, auto, extra=None):
        m = summarize(rs, preds, auto)
        m2 = summarize([rs[i] for i in keep], [preds[i] for i in keep], [auto[i] for i in keep])
        res[key] = {"all_test_images": dict(m, **(extra or {})), "without_confirmed_label_mismatches": m2}

    none = [False] * len(rs_b)
    both("1_original_awiros_plain_read", rs_b, [whole_text(r) for r in rs_b], none,
         {"note": "uploaded plate image read once by the original Awiros, no OpenCV, no rules"})
    both("2_original_production_pipeline", rs_b, [(r["prod"] or {}).get("text", "") for r in rs_b],
         [bool(r["prod"]) and r["prod"]["status"] == "SUCCESS" for r in rs_b],
         {"note": "YOLO11 + crop logic + adaptive OpenCV retries + original Awiros + rules + ML verifier",
          "mean_seconds": round(float(np.mean([r["prod_seconds"] for r in rs_b])), 3)})
    both("3_finetuned_awiros_plain_read", rs_n, [whole_text(r) for r in rs_n], none)
    both("4_finetuned_production_pipeline", rs_n, [(r["prod"] or {}).get("text", "") for r in rs_n],
         [bool(r["prod"]) and r["prod"]["status"] == "SUCCESS" for r in rs_n],
         {"mean_seconds": round(float(np.mean([r["prod_seconds"] for r in rs_n])), 3)})
    auto = [gate(e, S)[0] for e in ev["test"]]
    tta_used = sum(1 for e in ev["test"] if e and e["rule_success"] and S["tta_variants"] and (e["p_cal"] or 0) < S["tta_gate"])
    both("5_full_hybrid_finetuned", rs_n, [(r["prod"] or {}).get("text", "") for r in rs_n], auto,
         {"tta_used_on_images": tta_used,
          "note": "4 + calibrated gate + selective TTA consistency + region check (simulated from the cached evidence; bench/run_e2e.py confirms it with the real pipeline)"})
    out["test_comparison"] = res
    out["yolo"] = {"test_images_with_a_yolo_box_conf>=0.25": sum(bool(r["boxes"]) for r in rs_n), "n": len(rs_n),
                   "note": "detection precision needs box annotations, which the dataset does not have"}

    # ---- failure analysis (test, final system)
    REP.mkdir(parents=True, exist_ok=True)
    cats = collections.Counter()
    with open(REP / "failures_test_final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "awiros_prediction", "edit_distance", "char_errors", "rule_status",
                    "final_status", "gate_reasons", "p_calibrated", "avg_conf", "weakest_conf", "ml_probability",
                    "preprocessing", "quality_flags", "yolo_boxes", "two_row", "category"])
        for r, e, a in zip(rs_n, ev["test"], auto):
            p = r["prod"] or {}
            gt, pd = r["label"], p.get("text", "")
            wrong = pd != gt
            if not wrong and a:
                continue
            q = p.get("quality") or {}
            flags = list(q.get("flags") or [])
            h, wd = r["view_shapes"]["whole"]
            two = wd / max(h, 1) < cfg.TWO_LINE_ASPECT
            if name(r) in LABEL_MISMATCH:
                cat = "LABEL_MISMATCH (image shows another registration)"
            elif not wrong:
                cat = "CORRECT_BUT_SENT_TO_REVIEW"
            elif not pd:
                cat = "OCR_FAILURE_NO_TEXT"
            elif not r["boxes"] and False:
                cat = "DETECTION_FAILURE"
            elif len(pd) < len(gt):
                cat = "MISSING_CHARACTERS"
            elif len(pd) > len(gt):
                cat = "EXTRA_CHARACTERS"
            elif pd[:2] != gt[:2]:
                cat = "INCORRECT_STATE_CODE"
            else:
                cat = "INCORRECT_CHARACTER"
            if wrong and a:
                cat = "FALSE_ACCEPTANCE: " + cat
            cats[cat.split(" (")[0]] += 1
            errs = ";".join(f"{i}:{g}>{x}" for i, (g, x) in enumerate(zip(gt, pd)) if g != x) if len(gt) == len(pd) else ""
            w.writerow([name(r), gt, pd, lev(pd, gt), errs, p.get("status", "NO PLATE"),
                        "SUCCESS" if a else ("REVIEW" if p else "NO PLATE"), "|".join(gate(e, S)[1]) if e else "",
                        "" if not e or e["p_cal"] is None else round(e["p_cal"], 4), round(p.get("avg", 0), 4),
                        round(p.get("weakest", 0), 4), p.get("ml_probability"), p.get("preprocessing"),
                        "|".join(flags), len(r["boxes"]), two, cat])
    out["test_failure_categories"] = dict(cats)

    # ---- ML verifier compatibility (existing model, unchanged)
    from sklearn.metrics import roc_auc_score
    mlc = {}
    for s in ("val", "calib", "test"):
        pts = [(r["prod"]["ml_probability"], r["prod"]["text"] == r["label"], r["prod"]["avg"], e["p_cal"],
                r["prod"]["rule_status"], r["prod"]["status"])
               for r, e in zip(new[s], ev[s]) if r["prod"] and r["prod"].get("ml_probability") is not None and e]
        ent = {"rows_with_ml_probability": len(pts),
               "ml_promoted_review_to_success": sum(p[4] == "REVIEW" and p[5] == "SUCCESS" for p in pts)}
        if len({p[1] for p in pts}) == 2:
            ent["auc_ml_probability"] = round(roc_auc_score([p[1] for p in pts], [p[0] for p in pts]), 4)
            ent["auc_awiros_avg_conf"] = round(roc_auc_score([p[1] for p in pts], [p[2] for p in pts]), 4)
            if all(p[3] is not None for p in pts):
                ent["auc_calibrated_probability"] = round(roc_auc_score([p[1] for p in pts], [p[3] for p in pts]), 4)
        mlc[s] = ent
    out["ml_verifier_compatibility"] = mlc

    cal.save(oc.CAL_PATH, fitted_on="calib split (training/splits/calib.csv); thresholds on calib+val",
             weights=str(W), weights_sha256=out["weights_sha256"], auto_threshold=S["auto_threshold"],
             min_char_p=S["min_char_p"], tta_gate=S["tta_gate"], tta_variants=list(S["tta_variants"]),
             region_check=S["region"])
    (REP / "hybrid_benchmark.json").write_text(json.dumps(out, indent=2, default=str))
    for k, v in res.items():
        m = v["all_test_images"]
        print(f"{k:34s} exact {m['exact']}/{m['n']} = {m['exact_pct']}%  CER {m['cer_pct']}%  auto {m['auto_success']} "
              f"prec {m['auto_precision_pct']}  cov {m['coverage_pct']}%  FA {m['false_acceptance']}"
              f" | w/o label mismatches: {v['without_confirmed_label_mismatches']['exact_pct']}% "
              f"prec {v['without_confirmed_label_mismatches']['auto_precision_pct']}")
    print("gate:", out["gate_selection"])
    print("calibration T:", T, json.dumps({s: {k: out['calibration'][s][k] for k in ('plate_uncalibrated', 'plate_calibrated', 'char_uncalibrated', 'char_temperature_scaled')} for s in ('val', 'test')}))
    print("failures:", dict(cats))
    print("ml:", mlc)


if __name__ == "__main__":
    main()
