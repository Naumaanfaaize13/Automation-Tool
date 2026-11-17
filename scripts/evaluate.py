"""Measure plate-reading accuracy on labelled images for the whole pipeline (detector, OCR,
rules, ML verifier). --engines awiros,parseq runs the same images through both OCR engines.

Label sources (any combination):
  --manifest path/manifest.csv   columns 'path' (relative to the manifest folder) and 'plate_number'
                                 (optional: day_night, vehicle_type, plate_source)
  --folder DIR                   ground truth = file name up to the first '_' (e.g. TN01AB1234_x.jpg)
  --labels file.csv              columns 'file','plate' (paths relative to the CSV)
An empty label means "no readable registration": any SUCCESS there is counted as unverifiable.

  python scripts/evaluate.py --manifest ~/Desktop/plates/car/manifest.csv --modes legacy,new --detect crop
  python scripts/evaluate.py --folder my_labelled_images --detect yolo

--degrade adds synthetic night/blur/low-res/noise/tilt/occlusion copies (labelled as synthetic).
Results: <out>/report.json, <out>/report.md, <out>/predictions.csv. Nothing is written to results/.
"""
import argparse
import csv
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
import sys  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/

import cv2
import numpy as np

import indian_plate_validator as ipv

RANK = {"SUCCESS": 3, "REVIEW": 2, "REJECT": 1}


# ----------------------------------------------------------------------------- data
def load_items(args):
    items = []
    for m in args.manifest or []:
        m = Path(m).expanduser()
        for r in csv.DictReader(open(m, newline="", encoding="utf-8")):
            items.append({"path": str(m.parent / r["path"]), "gt": ipv.normalize(r.get("plate_number", "")),
                          "day_night": r.get("day_night", ""), "vehicle_type": r.get("vehicle_type", ""),
                          "label_source": r.get("plate_source", "") or "none", "synthetic": ""})
    for d in args.folder or []:
        for p in sorted(Path(d).expanduser().iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                m = re.match(r"^([A-Za-z0-9]+)_", p.name)
                items.append({"path": str(p), "gt": ipv.normalize(m.group(1)) if m else "",
                              "day_night": "", "vehicle_type": "", "label_source": "filename",
                              "synthetic": ""})
    for f in args.labels or []:
        f = Path(f).expanduser()
        for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
            items.append({"path": str(f.parent / r["file"]), "gt": ipv.normalize(r.get("plate", "")),
                          "day_night": "", "vehicle_type": "", "label_source": "labels", "synthetic": ""})
    if args.every > 1:
        items = items[:: args.every]
    if args.limit:
        items = items[: args.limit]
    return items


def degrade(img, kind, rng):
    h, w = img.shape[:2]
    if kind == "night":
        out = (255 * (img / 255.0) ** 2.2 * 0.6).astype(np.float32)
        return np.clip(out + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), 2.2)
    if kind == "lowres":
        s = cv2.resize(img, (max(8, w // 5), max(4, h // 5)), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "noise":
        return np.clip(img + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8)
    if kind == "tilt":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 9, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "occlude":   # hide the middle ~25% of the plate: the true number is NOT readable
        out = img.copy()
        cv2.rectangle(out, (int(w * 0.42), 0), (int(w * 0.67), h), (40, 40, 40), -1)
        return out
    raise ValueError(kind)


DEGRADATIONS = ("night", "blur", "lowres", "noise", "tilt", "occlude")


# ----------------------------------------------------------------------------- metrics
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(preds):
    """preds: list of prediction dicts for one (mode, detect) setting."""
    lab = [p for p in preds if p["gt_valid"]]
    unl = [p for p in preds if not p["gt_valid"]]
    n = len(lab)
    succ = [p for p in lab if p["status"] == "SUCCESS"]
    correct_text = [p for p in lab if p["text"] == p["gt"]]
    retried = [p for p in lab if p["retry_count"]]
    rescued = [p for p in retried if p["base_status"] != "SUCCESS" and p["status"] == "SUCCESS"]
    t = [p["seconds"] for p in preds]

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else None

    out = {
        "images": len(preds), "labelled_images": n,
        "exact_text_accuracy_pct": pct(len(correct_text), n),
        "character_accuracy_pct": round(100 * statistics.mean(
            [max(0.0, 1 - lev(p["text"], p["gt"]) / len(p["gt"])) for p in lab]), 2) if n else None,
        "cer_pct": round(100 * sum(lev(p["text"], p["gt"]) for p in lab)
                         / sum(len(p["gt"]) for p in lab), 2) if n else None,
        "success_rate_pct": pct(len(succ), n),
        "success_precision_pct": pct(sum(p["text"] == p["gt"] for p in succ), len(succ)),
        "false_acceptance_rate_pct": pct(sum(p["text"] != p["gt"] for p in succ), n),
        "false_rejection_rate_pct": pct(sum(p["status"] != "SUCCESS" for p in correct_text), len(correct_text)),
        "manual_review_rate_pct": pct(sum(p["status"] == "REVIEW" for p in lab), n),
        "reject_or_no_plate_pct": pct(sum(p["status"] in ("REJECT", "NO PLATE") for p in lab), n),
        "retried_images": len(retried),
        "retry_success_rate_pct": pct(len(rescued), len(retried)),
        "retry_rescued_correct": sum(p["text"] == p["gt"] for p in rescued),
        "retry_rescued_wrong": sum(p["text"] != p["gt"] for p in rescued),
        "unlabelled_images": len(unl),
        "unlabelled_success_unverifiable": sum(p["status"] == "SUCCESS" for p in unl),
        "seconds_per_image_mean": round(statistics.mean(t), 3) if t else None,
        "seconds_per_image_p50": round(statistics.median(t), 3) if t else None,
        "seconds_per_image_p95": round(sorted(t)[int(0.95 * (len(t) - 1))], 3) if t else None,
    }
    return out


# ----------------------------------------------------------------------------- run
def predict(pipe, legacy, img, mode, crop_mode, args):
    t0 = time.time()
    if mode == "legacy":
        plates = legacy.analyze(pipe, img, args.det_conf, 0.85, crop_mode)
        plates.sort(key=lambda p: (-RANK.get(p["status"], 0), -p["avg"]))
    else:
        plates = pipe.analyze(img, args.det_conf, crop_mode=crop_mode)
    dt = time.time() - t0
    if not plates:
        return {"status": "NO PLATE", "text": "", "avg": 0.0, "weakest": 0.0, "retry_count": 0,
                "base_status": "", "reasons": "", "preprocessing": "", "seconds": dt,
                "ocr_status": "DETECTION_FAILED", "ocr_ms": 0.0}
    p = plates[0]
    return {"status": p["status"], "text": p["text"] or "", "avg": p["avg"], "weakest": p["weakest"],
            "ocr_status": p.get("ocr_status", ""), "ocr_ms": round(p.get("ocr_ms", 0.0), 2),
            "retry_count": p.get("retry_count", 0), "base_status": p.get("base_status") or "",
            "reasons": "|".join(p["decision"]["reasons"]) if "decision" in p else "",
            "preprocessing": p.get("preprocessing", "original"), "seconds": dt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append")
    ap.add_argument("--folder", action="append")
    ap.add_argument("--labels", action="append")
    ap.add_argument("--modes", default="new", help="new and/or legacy (legacy needs PARSeq)")
    ap.add_argument("--engines", default="awiros",
                    help="OCR engines to compare with the same images: awiros, parseq or awiros,parseq")
    ap.add_argument("--detect", default="yolo", choices=["yolo", "crop", "both"],
                    help="yolo = run the detector; crop = images already are plate crops")
    ap.add_argument("--degrade", action="store_true", help="add synthetic degraded copies")
    ap.add_argument("--degrade-limit", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/runs/vehicle_plate/weights/best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "evaluation"))
    args = ap.parse_args()

    items = load_items(args)
    if not items:
        raise SystemExit("No images. Give --manifest, --folder or --labels.")
    from pipeline import Pipeline
    import legacy_pipeline as legacy
    engines = args.engines.split(",")
    pipes = {e: Pipeline(args.weights, args.device, ocr_engine=e) for e in engines}
    modes = args.modes.split(",")
    if "legacy" in modes and "parseq" not in pipes:
        raise SystemExit("--modes legacy needs --engines to include parseq")
    detects = ["yolo", "crop"] if args.detect == "both" else [args.detect]
    rng = np.random.default_rng(0)

    work = [(it, None) for it in items]
    if args.degrade:
        clean = [it for it in items if ipv.validate(it["gt"])["is_valid_format"]][: args.degrade_limit]
        work += [(it, k) for k in DEGRADATIONS for it in clean]

    preds, out = [], Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for i, (it, kind) in enumerate(work):
        img = cv2.imread(it["path"])
        if img is None:
            continue
        if kind:
            img = degrade(img, kind, rng)
        for engine, mode in [(e, m) for e in engines for m in modes
                             if m == "new" or e == "parseq"]:
            pipe = pipes[engine]
            for det in detects:
                pr = predict(pipe, legacy, img, mode, det == "crop", args)
                gt = it["gt"]
                if kind == "occlude":
                    gt_valid = True       # truth is known, but it is not visible
                else:
                    gt_valid = ipv.validate(gt)["is_valid_format"]
                preds.append(dict(pr, file=it["path"], gt=gt, gt_valid=gt_valid, engine=engine,
                                  mode=mode, detect=det,
                                  degradation=kind or "", day_night=it["day_night"],
                                  vehicle_type=it["vehicle_type"], label_source=it["label_source"]))
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(work)} images  {time.time() - t_start:.0f}s", flush=True)

    report = {"settings": {k: v for k, v in vars(args).items()}, "results": {}}
    groups = defaultdict(list)
    for p in preds:
        groups[(p["engine"], p["mode"], p["detect"], p["degradation"] or "clean")].append(p)
    for (engine, mode, det, deg), ps in sorted(groups.items()):
        key = f"{engine}/{mode}/{det}/{deg}"
        report["results"][key] = {"all": metrics(ps)}
        if deg == "clean":
            nb = [p for p in ps if p["label_source"] != "parseq"]
            if nb:
                report["results"][key]["label_source=NOT_parseq"] = metrics(nb)
            for field in ("day_night", "vehicle_type", "label_source"):
                for val in sorted({p[field] for p in ps if p[field]}):
                    report["results"][key][f"{field}={val}"] = metrics([p for p in ps if p[field] == val])
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        w.writeheader()
        w.writerows(preds)

    cols = ["labelled_images", "exact_text_accuracy_pct", "character_accuracy_pct", "cer_pct",
            "success_rate_pct",
            "success_precision_pct", "false_acceptance_rate_pct", "false_rejection_rate_pct",
            "manual_review_rate_pct", "retry_success_rate_pct", "unlabelled_success_unverifiable",
            "seconds_per_image_mean"]
    lines = ["| setting | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for key, res in report["results"].items():
        lines.append(f"| {key} | " + " | ".join(str(res["all"][c]) for c in cols) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}/report.json, report.md, predictions.csv")


if __name__ == "__main__":
    main()
