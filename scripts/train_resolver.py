"""Train the review resolver (review_resolver.py) on labelled plate images.

    python scripts/train_resolver.py --manifest ~/Desktop/plates/car/manifest.csv \
        --manifest ~/Desktop/plates/bike/manifest.csv --manifest ~/Desktop/plates/others/manifest.csv
    python scripts/train_resolver.py --reuse          # retrain from models/resolver_dataset.csv

Labels: training/verified_labels.csv (label_tool.py) wins over the manifest label for the same
image; images marked unreadable there are left out. Manifest labels are unaudited, so the
measured precision includes their errors (it is a lower bound if the model is right and the
label wrong, an upper bound otherwise) - verify labels to make the numbers trustworthy.

Split by vehicle (prefix before the first '__'): 70 % train, 30 % held out. The threshold is the
lowest P that reaches RESOLVER_TARGET_PRECISION in 5-fold grouped cross-validation on the train
part. Held-out results go to models/review_resolver_hgb.json and evaluation/review_resolver.md.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import sys  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/

import numpy as np

import anpr_config as cfg
import indian_plate_validator as ipv
from finetune_awiros import group_of

DATASET = ROOT / "models" / "resolver_dataset.csv"


def load_items(args):
    import evaluate as ev
    items = ev.load_items(argparse.Namespace(manifest=args.manifest, folder=None, labels=None,
                                             every=1, limit=args.limit))
    verified = {}
    vl = ROOT / "training" / "verified_labels.csv"
    if vl.is_file():
        for r in csv.DictReader(open(vl, newline="", encoding="utf-8")):
            verified[str(Path(r["path"]).resolve())] = r
    out = []
    for it in items:
        v = verified.get(str(Path(it["path"]).resolve()))
        if v is not None:
            if v["status"] != "ok":
                continue
            out.append(dict(it, gt=ipv.normalize(v["label"]), label_source="verified"))
        elif ipv.validate(it["gt"])["is_valid_format"]:
            out.append(it)
    return out


def collect(args):
    import cv2
    from pipeline import Pipeline
    from review_resolver import FEATURES, ReviewResolver
    from server import default_weights
    items = load_items(args)
    print(f"{len(items)} labelled images "
          f"({sum(i.get('label_source') == 'verified' for i in items)} verified by hand)")
    pl = Pipeline(default_weights(), args.device, use_resolver=False)
    pl.resolver = ReviewResolver(device=args.device)
    import evaluate as ev
    kinds = [None] + ([k for k in ev.DEGRADATIONS if k != "occlude"] if args.degrade else [])
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        img0 = cv2.imread(it["path"])
        if img0 is None:
            continue
        per_image = [None] + (list(rng.choice(kinds[1:], size=min(args.degrade_per_image, len(kinds) - 1),
                                              replace=False)) if args.degrade else [])
        for kind in per_image:
            img = img0 if kind is None else ev.degrade(img0, kind, rng)
            pl.resolver.collect = []
            pl._image_cache.clear()
            plates = pl.analyze(img)
            if not (plates and plates[0]["status"] == "REVIEW"):
                continue
            text = plates[0]["text"]
            match = [c for c in pl.resolver.collect if c["text"] == text]
            if match:
                f = match[-1]["features"]
                rows.append({"path": it["path"], "group": group_of(it["path"]), "gt": it["gt"],
                             "text": text, "label": int(text == it["gt"]), "variant": kind or "clean",
                             "label_source": it.get("label_source", ""),
                             **{k: f.get(k, float("nan")) for k in FEATURES}})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(items)} images, {len(rows)} eligible REVIEW plates, "
                  f"{time.time() - t0:.0f}s", flush=True)
    if not rows:
        raise SystemExit("No eligible REVIEW plates found in these images.")
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def model_():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                          l2_regularization=1.0, min_samples_leaf=10,
                                          random_state=0)


def pick_threshold(p, y, target):
    """Lowest threshold whose accepted set reaches the target precision (>= 5 accepted)."""
    best = 1.01
    for t in sorted(set(np.round(p, 4)), reverse=True):
        acc = p >= t
        if acc.sum() >= 5 and y[acc].mean() >= target:
            best = t
    return float(best)


def oof(X, y, groups, k=5):
    from sklearn.model_selection import GroupKFold
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(k, len(set(groups)))).split(X, y, groups):
        m = model_().fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--degrade", action="store_true",
                    help="also read blurred / low-res / night / noisy / tilted copies (more REVIEW cases)")
    ap.add_argument("--degrade-per-image", type=int, default=2)
    ap.add_argument("--target", type=float, default=cfg.RESOLVER_TARGET_PRECISION)
    args = ap.parse_args()
    from review_resolver import FEATURES, MODEL_PATH

    if args.reuse:
        rows = list(csv.DictReader(open(DATASET, newline="", encoding="utf-8")))
    else:
        if not args.manifest:
            raise SystemExit("Give --manifest (or --reuse).")
        rows = collect(args)
    X = np.array([[float(r[k]) if r[k] not in ("", None) else np.nan for k in FEATURES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    g = np.array([r["group"] for r in rows])
    print(f"{len(y)} eligible REVIEW plates, {y.mean():.1%} of them read correctly")

    from sklearn.metrics import roc_auc_score
    groups = sorted(set(g)); random.Random(0).shuffle(groups)
    held = set(groups[: int(len(groups) * 0.3)])
    te = np.array([x in held for x in g]); tr = ~te
    p_cv = oof(X[tr], y[tr], g[tr])
    thr = pick_threshold(p_cv, y[tr], args.target)
    m = model_().fit(X[tr], y[tr])
    p_te = m.predict_proba(X[te])[:, 1]
    acc = p_te >= thr
    held_out = {"plates": int(te.sum()), "correct_share": float(y[te].mean()),
                "auc": float(roc_auc_score(y[te], p_te)) if len(set(y[te])) > 1 else None,
                "promoted": int(acc.sum()), "promoted_wrong": int((acc & (y[te] == 0)).sum()),
                "precision": float(y[te][acc].mean()) if acc.any() else None,
                "recall_of_correct": float(acc[y[te] == 1].mean()) if (y[te] == 1).any() else None}
    # baseline: PARSeq + Awiros agree exactly and nothing else
    agree = X[te][:, FEATURES.index("p_any_exact")] >= 1
    held_out["baseline_engines_agree"] = {
        "promoted": int(agree.sum()), "wrong": int((agree & (y[te] == 0)).sum())}
    print(json.dumps({"threshold_cv": thr, "heldout": held_out}, indent=2))

    # final model on all data, threshold from cross-validation on all data
    p_all = oof(X, y, g)
    thr_all = pick_threshold(p_all, y, args.target)
    final = model_().fit(X, y)
    import joblib
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "threshold": thr_all, "features": FEATURES}, MODEL_PATH)
    meta = {"trained": time.strftime("%Y-%m-%d %H:%M"), "n": int(len(y)), "target_precision": args.target,
            "threshold": thr_all, "heldout": held_out, "heldout_threshold": thr,
            "labels": "verified + unaudited manifest" if any(r.get("label_source") == "verified" for r in rows)
            else "unaudited manifest labels"}
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    prec_s = "" if held_out["precision"] is None else f"{held_out['precision']:.1%}"
    rep = ROOT / "evaluation" / "review_resolver.md"
    rep.write_text(
        "# Review resolver (second layer for REVIEW plates)\n\n"
        f"Eligible REVIEW plates: {len(y)} ({y.mean():.1%} read correctly by the first pass). "
        f"Labels: {meta['labels']}.\n\n"
        "| held-out (by vehicle) | plates | promoted to SUCCESS | wrong among promoted | precision | AUC |\n"
        "|---|---|---|---|---|---|\n"
        f"| ML resolver (P >= {thr:.3f}) | {held_out['plates']} | {held_out['promoted']} | "
        f"{held_out['promoted_wrong']} | "
        f"{prec_s} | "
        f"{'' if held_out['auc'] is None else round(held_out['auc'], 3)} |\n"
        f"| Awiros and PARSeq agree (no ML) | {held_out['plates']} | "
        f"{held_out['baseline_engines_agree']['promoted']} | {held_out['baseline_engines_agree']['wrong']} | | |\n\n"
        f"Deployed threshold (cross-validated on all {len(y)} plates): {thr_all:.3f}.\n")
    print(f"saved {MODEL_PATH} (threshold {thr_all:.3f}); report {rep}")


if __name__ == "__main__":
    main()
