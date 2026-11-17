"""Train the ML OCR verifier (HistGradientBoostingClassifier) on labelled images.

  python scripts/train_verifier.py --manifest ~/Desktop/plates/car/manifest.csv \
      --manifest ~/Desktop/plates/bike/manifest.csv --manifest ~/Desktop/plates/others/manifest.csv

Steps
 1. Run the pipeline (rules only, no ML) in YOLO mode and plate-crop mode on every image, plus
    synthetic degraded copies, and record the features of the reading it would report.
 2. Target = reading equals the label. Only "eligible" readings are used (valid format, read
    directly, no hard review reason): the verifier never decides anything else.
 3. Split by ORIGINAL image (70 % train / 30 % test; degraded copies stay with their original).
 4. Choose the probability threshold from 5-fold grouped cross-validation on the train part so
    that accepted readings reach --target-precision.
 5. Report rules vs ML on the untouched test part, then refit on all data and save
    models/ocr_verifier_hgb.joblib (+ .json report).
Labels are whatever the manifest says. If they are not human-checked, the model learns their
errors too, and the metrics measure agreement with them, not true accuracy.
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import sys  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/

import cv2
import numpy as np

import anpr_config as cfg
import indian_plate_validator as ipv
from evaluate import DEGRADATIONS, degrade, load_items
from ocr_verifier import FEATURES, MODEL_PATH, eligible, features

DATASET = ROOT / "models" / "verifier_dataset.csv"


def build_dataset(args):
    from pipeline import Pipeline
    items = load_items(args)
    pipe = Pipeline(args.weights, args.device, use_ml=False)
    rng = np.random.default_rng(0)
    work = [(it, "") for it in items]
    if args.degrade_limit:
        clean = [it for it in items if ipv.validate(it["gt"])["is_valid_format"]][: args.degrade_limit]
        work += [(it, k) for k in DEGRADATIONS for it in clean]
    rows, t0 = [], time.time()
    for i, (it, kind) in enumerate(work):
        img = cv2.imread(it["path"])
        if img is None:
            continue
        if kind:
            img = degrade(img, kind, rng)
        for det in ("yolo", "crop"):
            plates = pipe.analyze(img, args.det_conf, crop_mode=det == "crop")
            base = {"group": it["path"], "gt": it["gt"], "degradation": kind, "detect": det,
                    "gt_valid": int(kind == "occlude" or ipv.validate(it["gt"])["is_valid_format"])}
            if not plates:
                rows.append(dict(base, rule_status="NO PLATE", text="", eligible=0))
                continue
            p = plates[0]
            r = dict(base, rule_status=p["status"], text=p["text"] or "",
                     eligible=int(eligible(p["decision"])))
            r.update(features(p))
            rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(work)} images, {time.time() - t0:.0f}s", flush=True)
    DATASET.parent.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(DATASET, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows


def load_dataset():
    rows = list(csv.DictReader(open(DATASET, newline="")))
    for r in rows:
        for k in ("gt_valid", "eligible"):
            r[k] = int(r[k])
    return rows


def X_of(rows):
    return np.array([[float(r[f]) if r.get(f) not in (None, "", "nan") else np.nan for f in FEATURES]
                     for r in rows], dtype=float)


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=15, l2_regularization=1.0,
                                          random_state=0)


def oof_probs(rows):
    from sklearn.model_selection import GroupKFold
    X, y = X_of(rows), np.array([r["text"] == r["gt"] for r in rows], int)
    g = np.array([r["group"] for r in rows])
    out = np.zeros(len(rows))
    for tr, va in GroupKFold(n_splits=5).split(X, y, g):
        m = make_model().fit(X[tr], y[tr])
        out[va] = m.predict_proba(X[va])[:, 1]
    return out, y


def pick_threshold(probs, y, weakest, target, floor):
    """Lowest threshold whose accepted set (prob >= t and weakest >= floor) keeps precision >= target."""
    ok = weakest >= floor
    order = np.argsort(-probs)
    best_t, tp = 1.01, 0
    accepted = 0
    for i in order:
        if not ok[i]:
            continue
        accepted += 1
        tp += y[i]
        if tp / accepted >= target and accepted >= 10:
            best_t = probs[i]
    return float(best_t)


def statuses(rows, probs, t, floor):
    """ML-mode status per row; rows not eligible keep the rule status."""
    out = []
    for r, p in zip(rows, probs):
        if r["eligible"]:
            out.append("SUCCESS" if p >= t and float(r["weakest"]) >= floor else "REVIEW")
        else:
            out.append(r["rule_status"])
    return out


def report(rows, sts):
    lab = [(r, s) for r, s in zip(rows, sts) if r["gt_valid"]]
    n = len(lab)
    succ = [(r, s) for r, s in lab if s == "SUCCESS"]
    wrong = sum(r["text"] != r["gt"] for r, _ in succ)
    pct = lambda a, b: round(100 * a / b, 2) if b else None   # noqa: E731
    return {"labelled": n, "exact_text_pct": pct(sum(r["text"] == r["gt"] for r, _ in lab), n),
            "success_rate_pct": pct(len(succ), n), "success_precision_pct": pct(len(succ) - wrong, len(succ)),
            "false_acceptance_pct": pct(wrong, n), "wrong_accepted": wrong,
            "review_pct": pct(sum(s == "REVIEW" for _, s in lab), n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append")
    ap.add_argument("--folder", action="append")
    ap.add_argument("--labels", action="append")
    ap.add_argument("--degrade-limit", type=int, default=120)
    ap.add_argument("--reuse", action="store_true", help=f"reuse {DATASET.name} instead of re-running OCR")
    ap.add_argument("--target-precision", type=float, default=cfg.ML_TARGET_PRECISION)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/runs/vehicle_plate/weights/best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    rows = load_dataset() if args.reuse and DATASET.exists() else build_dataset(args)
    import joblib
    import sklearn
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.array([r["group"] for r in rows])
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
                          .split(np.zeros(len(rows)), groups=groups))
    train = [rows[i] for i in tr_idx]
    test = [rows[i] for i in te_idx]
    tr_el = [r for r in train if r["eligible"] and r["gt_valid"]]
    print(f"rows {len(rows)}; train eligible {len(tr_el)} "
          f"({sum(r['text'] == r['gt'] for r in tr_el)} correct); test rows {len(test)}")

    floor = cfg.ML_MIN_WEAKEST_CHAR
    p_oof, y_oof = oof_probs(tr_el)
    t = pick_threshold(p_oof, y_oof, np.array([float(r["weakest"]) for r in tr_el]),
                       args.target_precision, floor)
    model = make_model().fit(X_of(tr_el), np.array([r["text"] == r["gt"] for r in tr_el], int))

    te_el_idx = [i for i, r in enumerate(test) if r["eligible"]]
    probs = np.zeros(len(test))
    if te_el_idx:
        probs[te_el_idx] = model.predict_proba(X_of([test[i] for i in te_el_idx]))[:, 1]
    ml_st = statuses(test, probs, t, floor)
    rule_st = [r["rule_status"] for r in test]

    results = {}
    for det in ("yolo", "crop"):
        for deg in ["clean"] + list(DEGRADATIONS):
            sel = [i for i, r in enumerate(test) if r["detect"] == det and (r["degradation"] or "clean") == deg]
            if not sel:
                continue
            sub = [test[i] for i in sel]
            results[f"{det}/{deg}"] = {"rules": report(sub, [rule_st[i] for i in sel]),
                                       "ml": report(sub, [ml_st[i] for i in sel])}

    # which features matter (permutation importance on held-out eligible rows)
    imp = {}
    if len(te_el_idx) > 20:
        from sklearn.inspection import permutation_importance
        te_el = [test[i] for i in te_el_idx]
        pi = permutation_importance(model, X_of(te_el), np.array([r["text"] == r["gt"] for r in te_el], int),
                                    scoring="roc_auc", n_repeats=10, random_state=0)
        imp = dict(sorted(((f, round(float(m), 4)) for f, m in zip(FEATURES, pi.importances_mean)),
                          key=lambda kv: -kv[1])[:12])
        from sklearn.metrics import roc_auc_score
        y_te = np.array([r["text"] == r["gt"] for r in te_el], int)
        auc_ml = roc_auc_score(y_te, probs[te_el_idx]) if 0 < y_te.sum() < len(y_te) else None
        auc_avg = roc_auc_score(y_te, [float(r["avg"]) for r in te_el]) if auc_ml else None
        auc_weak = roc_auc_score(y_te, [float(r["weakest"]) for r in te_el]) if auc_ml else None
    else:
        auc_ml = auc_avg = auc_weak = None

    # final model on all eligible labelled rows; threshold from grouped CV on all of them
    all_el = [r for r in rows if r["eligible"] and r["gt_valid"]]
    p_all, y_all = oof_probs(all_el)
    t_final = pick_threshold(p_all, y_all, np.array([float(r["weakest"]) for r in all_el]),
                             args.target_precision, floor)
    final = make_model().fit(X_of(all_el), y_all)

    info = {"trained_at": datetime.now().isoformat(timespec="seconds"),
            "sklearn": sklearn.__version__, "algorithm": "HistGradientBoostingClassifier",
            "train_rows": len(all_el), "positives": int(y_all.sum()),
            "target_precision": args.target_precision, "threshold_split_model": t,
            "threshold": t_final, "min_weakest": floor,
            "test_auc": {"ml_verifier": auc_ml, "avg_conf_only": auc_avg, "weakest_char_only": auc_weak},
            "test_metrics": results, "permutation_importance_auc": imp,
            "note": "Labels came from the manifests and are not human-audited; metrics measure "
                    "agreement with them."}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "threshold": t_final, "min_weakest": floor,
                 "info": info}, MODEL_PATH)
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(info, indent=2))

    print(f"\nthreshold (split model) {t:.3f}; final threshold {t_final:.3f}")
    print(f"held-out AUC: ML {auc_ml}, avg-conf {auc_avg}, weakest-char {auc_weak}")
    print("\n| test set (30% held out) | rules: success / precision / wrong | ML: success / precision / wrong |")
    print("|---|---|---|")
    for k, v in results.items():
        a, b = v["rules"], v["ml"]
        print(f"| {k} (n={a['labelled']}) | {a['success_rate_pct']}% / {a['success_precision_pct']}% / "
              f"{a['wrong_accepted']} | {b['success_rate_pct']}% / {b['success_precision_pct']}% / {b['wrong_accepted']} |")
    print("\ntop features:", imp)
    print(f"\nSaved {MODEL_PATH} and {MODEL_PATH.with_suffix('.json')}")


if __name__ == "__main__":
    main()
