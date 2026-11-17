"""Offline comparison of preprocessing, selective TTA, calibration and decision policies on the
cached Awiros reads (bench/cache_reads.py). Everything is CHOSEN on val/calib and REPORTED on the
untouched test split.

    python bench/analyze.py            -> bench/reports/benchmark.json + failures_test.csv

Ground truth = human-audited plate_number. Exact match = the complete registration string."""
import collections
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/
import indian_plate_validator as ipv  # noqa: E402
import ocr_calibration as oc  # noqa: E402

CACHE = ROOT / "bench" / "cache"
REP = ROOT / "bench" / "reports"
DEC = oc.Decoder()


def load(split):
    p = CACHE / f"{split}.pkl"
    return pickle.load(open(p, "rb")) if p.is_file() else []


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def reading(probs, T=1.0):
    d = DEC.decode(probs, T)
    v = ipv.find_in_text(d["raw"])
    d["valid"] = bool(v["is_valid_format"])
    d["warnings"] = list(v.get("warnings") or [])
    d["plate"] = v["normalized_text"] if d["valid"] else ipv.normalize(d["raw"])
    return d


def text_metrics(preds, labels):
    n = len(labels)
    ok = sum(p == g for p, g in zip(preds, labels))
    edits = sum(lev(p, g) for p, g in zip(preds, labels))
    chars = sum(len(g) for g in labels)
    return {"n": n, "exact": ok, "exact_pct": round(100 * ok / max(n, 1), 2),
            "cer_pct": round(100 * edits / max(chars, 1), 2),
            "char_acc_pct": round(100 * (1 - edits / max(chars, 1)), 2)}


def decision_metrics(preds, labels, auto):
    n = len(labels)
    acc = [p == g for p, g in zip(preds, labels)]
    na = sum(auto)
    good = sum(a and c for a, c in zip(auto, acc))
    return {"auto_success": na, "auto_precision_pct": round(100 * good / na, 2) if na else None,
            "coverage_pct": round(100 * na / max(n, 1), 2),
            "false_acceptance": na - good, "false_acceptance_pct": round(100 * (na - good) / max(n, 1), 2),
            "manual_review_pct": round(100 * (n - na) / max(n, 1), 2),
            "false_rejection": sum(c and not a for a, c in zip(auto, acc)),
            "false_rejection_pct_of_correct": round(100 * sum(c and not a for a, c in zip(auto, acc)) / max(sum(acc), 1), 2)}


# ----------------------------------------------------------------------------- policies
def primary_view(r, name):
    if name in r["views"]:
        return name
    return "whole"


def tta_select(r, primary, variants, T, cal, gate_p):
    """Selective TTA on ONE image: read the primary view; only if its calibrated P(correct) is
    below gate_p, look at the variants. Choose the reading supported by most views (valid
    format first); ties -> higher calibrated probability. No averaging of raw confidences."""
    base = reading(r["views"][primary], T)
    base["p_cal"] = cal.prob(base) if cal else base["avg"]
    base["view"] = primary
    if base["valid"] and base["p_cal"] >= gate_p:
        base["tta"] = False
        base["agree"] = 1
        base["n_views"] = 1
        return base
    reads = [base]
    for v in variants:
        if v in r["views"] and v != primary:
            d = reading(r["views"][v], T)
            d["p_cal"] = cal.prob(d) if cal else d["avg"]
            d["view"] = v
            reads.append(d)
    votes = collections.Counter(d["plate"] for d in reads if d["valid"])
    def key(d):
        return (d["valid"], votes.get(d["plate"], 0), d["p_cal"])
    best = max(reads, key=key)
    best = dict(best)
    best["tta"] = True
    best["agree"] = votes.get(best["plate"], 0)
    best["n_views"] = len(reads)
    best["conflict"] = len([t for t in votes if t != best["plate"]]) > 0
    return best


def main():
    splits = {s: load(s) for s in ("train", "val", "calib", "test")}
    splits = {s: v for s, v in splits.items() if v}
    print("cached:", {s: len(v) for s, v in splits.items()})
    out = {"splits": {s: len(v) for s, v in splits.items()}}

    # ---- 1. current production system (YOLO11 + Awiros + existing rules + ML verifier)
    prod = {}
    for s, rs in splits.items():
        labels = [r["label"] for r in rs]
        preds = [(r["prod"] or {}).get("text", "") for r in rs]
        auto = [bool(r["prod"]) and r["prod"]["status"] == "SUCCESS" for r in rs]
        m = text_metrics(preds, labels); m.update(decision_metrics(preds, labels, auto))
        m["no_plate"] = sum(r["prod"] is None for r in rs)
        m["mean_seconds"] = round(float(np.mean([r["prod_seconds"] for r in rs])), 2)
        prod[s] = m
    out["1_production_current"] = prod

    # ---- detection
    det = {}
    for s, rs in splits.items():
        det[s] = {"yolo_box_found": sum(bool(r["boxes"]) for r in rs), "n": len(rs),
                  "recall_pct": round(100 * sum(bool(r["boxes"]) for r in rs) / len(rs), 2)}
    out["yolo_detection"] = det

    # ---- 2. single views / OpenCV variants (T = 1)
    names = sorted({n for rs in splits.values() for r in rs for n in r["views"]})
    views = {}
    for s in ("val", "test", "calib", "train"):
        rs = splits.get(s, [])
        if not rs:
            continue
        views[s] = {}
        for n in names:
            labels, preds = [], []
            for r in rs:
                labels.append(r["label"])
                v = n if n in r["views"] else ("whole" if n.startswith("whole") else
                                               "whole" + n[len("yolo_pad6"):] if ("whole" + n[len("yolo_pad6"):]) in r["views"] else "whole")
                preds.append(reading(r["views"][v])["plate"])
            views[s][n] = text_metrics(preds, labels)
    out["2_single_views"] = views
    sel_split = "val"
    ranking = sorted(views[sel_split].items(), key=lambda kv: -kv[1]["exact"])
    best_primary = ranking[0][0]
    out["primary_view_chosen_on_val"] = best_primary
    print("best single view on val:", ranking[:6])

    # ---- 3. calibration (fit on calib, primary view)
    cal_rs = splits.get("calib", [])
    T = oc.fit_temperature(DEC, [r["views"][primary_view(r, best_primary)] for r in cal_rs],
                           [r["label"] for r in cal_rs])
    dec_cal = [reading(r["views"][primary_view(r, best_primary)], T) for r in cal_rs]
    cal = oc.PlateCalibrator.fit(dec_cal, [d["plate"] == r["label"] for d, r in zip(dec_cal, cal_rs)], T)
    calib = {"temperature": T, "plate_w": cal.w, "plate_b": cal.b}
    for s in ("val", "test"):
        rs = splits.get(s, [])
        if not rs:
            continue
        raw_d = [reading(r["views"][primary_view(r, best_primary)], 1.0) for r in rs]
        t_d = [reading(r["views"][primary_view(r, best_primary)], T) for r in rs]
        y = [d["plate"] == r["label"] for d, r in zip(raw_d, rs)]
        raw_conf = [float(np.prod([c["p"] for c in d["chars"]])) if d["chars"] else 0.0 for d in raw_d]
        cal_conf = [cal.prob(d) for d in t_d]
        pc_raw = oc.char_pairs(DEC, [r["views"][primary_view(r, best_primary)] for r in rs], [r["label"] for r in rs], 1.0)
        pc_t = oc.char_pairs(DEC, [r["views"][primary_view(r, best_primary)] for r in rs], [r["label"] for r in rs], T)
        calib[s] = {
            "plate_raw": {"ece": round(oc.ece(raw_conf, y), 4), "brier": round(oc.brier(raw_conf, y), 4), "nll": round(oc.nll(raw_conf, y), 4)},
            "plate_calibrated": {"ece": round(oc.ece(cal_conf, y), 4), "brier": round(oc.brier(cal_conf, y), 4), "nll": round(oc.nll(cal_conf, y), 4)},
            "char_raw": {"ece": round(oc.ece(*pc_raw), 4), "nll": round(oc.nll(*pc_raw), 4), "n": int(len(pc_raw[0]))},
            "char_tempered": {"ece": round(oc.ece(*pc_t), 4), "nll": round(oc.nll(*pc_t), 4)},
            "reliability_plate_calibrated": oc.reliability(cal_conf, y),
            "reliability_plate_raw": oc.reliability(raw_conf, y)}
    out["3_calibration"] = calib

    # ---- 4. selective TTA (variant set + gate chosen on val)
    base_names = [n for n in names if n.startswith(best_primary.split("+")[0] + "+")] or names
    cand_sets = {"none": [], "clahe": ["+clahe"], "clahe_bilateral": ["+clahe_bilateral"],
                 "photometric": ["+clahe", "+clahe_bilateral", "+bilateral", "+sharpen", "+low_light"],
                 "geometric": ["+deskew", "+perspective"],
                 "crops": ["whole", "yolo_pad6", "yolo_pad25", "yolo_pad50"],
                 "crops+photometric": ["whole", "yolo_pad6", "yolo_pad25", "+clahe", "+clahe_bilateral", "+sharpen"],
                 "all": ["whole", "yolo_pad6", "yolo_pad25", "yolo_pad50", "+clahe", "+clahe_bilateral",
                         "+bilateral", "+sharpen", "+low_light", "+deskew", "+perspective", "+gaussian", "+resize", "+upscale"]}
    stem = best_primary.split("+")[0]

    def expand(lst):
        return [x if not x.startswith("+") else stem + x for x in lst]

    tta = {}
    for sname, lst in cand_sets.items():
        for gate in (0.0, 0.5, 0.8, 0.9, 0.95, 1.01):
            rs = splits["val"]
            res = [tta_select(r, primary_view(r, best_primary), expand(lst), T, cal, gate) for r in rs]
            m = text_metrics([d["plate"] for d in res], [r["label"] for r in rs])
            m["tta_used_pct"] = round(100 * sum(d["tta"] for d in res) / len(rs), 1)
            m["mean_views"] = round(float(np.mean([d["n_views"] for d in res])), 2)
            tta[f"{sname}@{gate}"] = m
    best_tta = max(tta.items(), key=lambda kv: (kv[1]["exact"], -kv[1]["mean_views"]))
    out["4_tta_val_grid"] = tta
    out["tta_chosen_on_val"] = best_tta[0]
    print("best TTA on val:", best_tta)
    tset, tgate = best_tta[0].rsplit("@", 1)
    tgate = float(tgate)

    def run_hybrid(rs):
        return [tta_select(r, primary_view(r, best_primary), expand(cand_sets[tset]), T, cal, tgate) for r in rs]

    # ---- 5. hybrid decision threshold (chosen on calib+val: zero false accepts there)
    pool = splits.get("calib", []) + splits.get("val", [])
    pr = run_hybrid(pool)

    def hybrid_auto(d, r, thr):
        q = (r.get("view_quality") or {}).get("whole", {})
        return (d["valid"] and not d["warnings"] and d["p_cal"] >= thr and not d.get("conflict")
                and "UNUSABLE" not in (q.get("flags") or []) and min([c["p"] for c in d["chars"]] or [0]) >= 0.5)

    thr_grid = np.round(np.arange(0.50, 1.0001, 0.005), 3)
    rows = []
    for thr in thr_grid:
        auto = [hybrid_auto(d, r, thr) for d, r in zip(pr, pool)]
        m = decision_metrics([d["plate"] for d in pr], [r["label"] for r in pool], auto)
        rows.append((float(thr), m))
    zero_fa = [t for t, m in rows if m["false_acceptance"] == 0 and m["auto_success"] > 0]
    thr = min(zero_fa) if zero_fa else 1.0
    out["5_hybrid_threshold"] = {"chosen": thr, "rule": "lowest threshold with zero false acceptances on calib+val",
                                 "grid": {str(t): m for t, m in rows[::5]}}

    final = {}
    fails = []
    for s in ("test", "val", "calib"):
        rs = splits.get(s, [])
        if not rs:
            continue
        res = run_hybrid(rs)
        preds = [d["plate"] for d in res]
        auto = [hybrid_auto(d, r, thr) for d, r in zip(res, rs)]
        m = text_metrics(preds, [r["label"] for r in rs])
        m.update(decision_metrics(preds, [r["label"] for r in rs], auto))
        m["tta_used_pct"] = round(100 * sum(d["tta"] for d in res) / len(rs), 1)
        final[s] = m
        if s == "test":
            for d, r, a in zip(res, rs, auto):
                if d["plate"] != r["label"] or (a and d["plate"] != r["label"]):
                    fails.append((r, d, a))
    out["6_hybrid_final"] = final

    # ---- ML verifier compatibility (existing model, existing pipeline output)
    mlc = {}
    for s, rs in splits.items():
        pts = [(r["prod"]["ml_probability"], r["prod"]["text"] == r["label"], r["prod"]["avg"],
                r["prod"]["rule_status"], r["prod"]["status"])
               for r in rs if r["prod"] and r["prod"].get("ml_probability") is not None]
        promoted = [p for p in pts if p[3] == "REVIEW" and p[4] == "SUCCESS"]
        entry = {"rows_with_ml_probability": len(pts),
                 "promoted_review_to_success": len(promoted),
                 "promoted_correct": sum(p[1] for p in promoted)}
        if len({p[1] for p in pts}) == 2:
            from sklearn.metrics import roc_auc_score
            entry["auc_ml_probability"] = round(roc_auc_score([p[1] for p in pts], [p[0] for p in pts]), 4)
            entry["auc_awiros_avg_conf"] = round(roc_auc_score([p[1] for p in pts], [p[2] for p in pts]), 4)
        mlc[s] = entry
    out["ml_verifier_compatibility"] = mlc

    calib_path = oc.CAL_PATH
    cal.save(calib_path, fitted_on="calib split (training/splits/calib.csv)", primary_view=best_primary,
             tta=best_tta[0], auto_threshold=thr)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "benchmark.json").write_text(json.dumps(out, indent=2, default=str))
    with open(REP / "failures_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "prediction", "auto_approved", "p_calibrated", "view",
                    "tta", "valid_format", "edit_distance", "char_errors", "quality_flags",
                    "yolo_boxes", "production_text", "production_status", "category"])
        for r, d, a in fails:
            q = (r.get("view_quality") or {}).get("whole", {})
            gt, pd = r["label"], d["plate"]
            if not pd:
                cat = "OCR_FAILURE_EMPTY"
            elif len(pd) < len(gt):
                cat = "MISSING_CHARACTERS"
            elif len(pd) > len(gt):
                cat = "EXTRA_CHARACTERS"
            else:
                cat = "INCORRECT_CHARACTER"
            if "DARK" in (q.get("flags") or []):
                cat += "|LOW_LIGHT"
            if "TILTED" in (q.get("flags") or []):
                cat += "|PERSPECTIVE_TILT"
            if not r["boxes"]:
                cat += "|NO_YOLO_BOX"
            errs = ";".join(f"{i}:{g}>{p}" for i, (g, p) in enumerate(zip(gt, pd)) if g != p) if len(gt) == len(pd) else ""
            w.writerow([Path(r["path"]).name, gt, pd, a, round(d["p_cal"], 4), d["view"], d["tta"],
                        d["valid"], lev(pd, gt), errs, "|".join(q.get("flags") or []), len(r["boxes"]),
                        (r["prod"] or {}).get("text", ""), (r["prod"] or {}).get("status", "NO PLATE"), cat])
    print(json.dumps({k: out[k] for k in ("1_production_current", "6_hybrid_final", "ml_verifier_compatibility",
                                          "primary_view_chosen_on_val", "tta_chosen_on_val")}, indent=1))
    print("calibration:", {k: v for k, v in calib.items() if not isinstance(v, dict) or "plate_raw" in v and False} or calib.get("test", {}).get("plate_calibrated"))


if __name__ == "__main__":
    main()
