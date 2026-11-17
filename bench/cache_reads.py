"""Run the CURRENT production pipeline on every labelled image and cache, per image, the Awiros
CTC probability matrices of a fixed set of crop views / OpenCV variants, so that preprocessing,
TTA, calibration and decision policies can be compared offline on exactly the same reads.

    python bench/cache_reads.py --splits train val calib test [--limit N]
    python bench/cache_reads.py --weights models/awiros_finetuned_head/candidate_head.safetensors --tag ft

Writes bench/cache/<split>.pkl (or bench/cache/<tag>/<split>.pkl). The production pipeline runs
WITHOUT the hybrid gate here (the gate is simulated from the cached evidence by
bench/analyze_hybrid.py); per image it also stores the chosen reading's CTC matrix and the reads
of the same crop's TTA variants. Nothing in the production code or any weights is changed."""
import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/
import anpr_config as cfg  # noqa: E402
import plate_preprocess as pp  # noqa: E402
import pipeline as pl  # noqa: E402
import image_enhance as ie  # noqa: E402

YOLO = ROOT / "runs" / "detect" / "runs" / "vehicle_plate" / "weights" / "best.pt"


def views_for(frame, boxes):
    """name -> BGR image. 'whole' = the uploaded plate image; 'yolo*' = the best YOLO box."""
    v = {"whole": frame}
    if boxes:
        b = max(boxes, key=lambda b: b[4])
        for px in cfg.BASE_H_PADS:
            v[f"yolo_pad{int(px * 100)}"] = pl._crop(frame, b, px)
    for base in ("whole", "yolo_pad6"):
        if base not in v:
            continue
        for name, img in ie.all_variants(v[base]).items():
            v[f"{base}+{name}"] = img
    return {k: im for k, im in v.items() if im is not None and im.size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "calib", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", help="Awiros weights to benchmark (default: anpr_config)")
    ap.add_argument("--tag", default="", help="cache sub-folder for these weights")
    a = ap.parse_args()
    if a.weights:
        cfg.AWIROS_WEIGHTS = Path(a.weights).resolve()
    P = pl.Pipeline(YOLO, "cpu", use_resolver=False, use_hybrid=False)
    print("weights:", P.reader.weights, flush=True)
    tta_names = ["clahe", "clahe_bilateral", "gaussian", "bilateral", "sharpen", "low_light", "deskew",
                 "perspective", "resize"]
    R = P.reader
    dec = R._decode
    R._decode = lambda probs: dict(dec(probs), probs=probs.astype(np.float16))
    out_dir = ROOT / "bench" / "cache" / a.tag if a.tag else ROOT / "bench" / "cache"; out_dir.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        rows = list(csv.DictReader(open(ROOT / "training" / "splits" / f"{split}.csv")))
        if a.limit:
            rows = rows[: a.limit]
        recs, t_start = [], time.time()
        for i, r in enumerate(rows):
            frame = cv2.imread(r["path"])
            t0 = time.perf_counter()
            plates = P.analyze(frame)
            prod_s = time.perf_counter() - t0
            res = P.detector.predict(frame, conf=0.25, iou=0.4, verbose=False, device="cpu")[0].boxes
            boxes = [list(map(float, b.xyxy[0].tolist())) + [float(b.conf[0])] for b in res]
            best = plates[0] if plates else None
            prod = None
            if best is not None:
                d = best["decision"]
                prod = {"status": best["status"], "text": best["text"], "final_text": best["final_text"],
                        "raw": best["raw"], "avg": best["avg"], "weakest": best["weakest"],
                        "ml_probability": best.get("ml_probability"), "rule_status": best.get("rule_status"),
                        "reasons": list(d.get("reasons") or []), "preprocessing": best["preprocessing"],
                        "ocr_attempts": best["ocr_attempts"], "auto_crop": best.get("auto_crop"),
                        "ocr_status": best.get("ocr_status"), "quality": best["quality"],
                        "n_plates": len(plates), "chars": [(c["char"], c["p"]) for c in best["chars"]]}
                ch = d.get("chosen") or {}
                prod["chosen_probs"] = ch.get("probs")
                img = ch.get("img")
                if img is not None and img.size:
                    named = [(n, ie.apply(img, n)) for n in tta_names]
                    named = [(n, im) for n, im in named if im is not None and not ie.is_identity(im, img)]
                    rr = R.read_chars([im for _, im in named]) if named else []
                    prod["tta"] = {n: {"raw": x["raw"], "avg": x["avg"]} for (n, _), x in zip(named, rr)}
            vs = views_for(frame, boxes)
            names = list(vs)
            reads = R.read_chars([vs[n] for n in names])
            vq = {n: pp.assess(vs[n]) for n in ("whole", "yolo_pad6") if n in vs}
            recs.append({**r, "boxes": boxes, "prod": prod, "prod_seconds": prod_s,
                         "views": {n: rd["probs"] for n, rd in zip(names, reads)},
                         "view_quality": vq, "view_shapes": {n: vs[n].shape[:2] for n in names}})
            if (i + 1) % 10 == 0:
                el = time.time() - t_start
                print(f"{split} {i + 1}/{len(rows)}  {el / (i + 1):.2f}s/img", flush=True)
        with open(out_dir / f"{split}.pkl", "wb") as f:
            pickle.dump(recs, f)
        print(f"{split}: {len(recs)} images cached in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
