"""Run the REAL production pipeline (pipeline.Pipeline.analyze, hybrid gate included) on one split
and score it against the audited labels. Confirms the simulated numbers of analyze_hybrid.py.

    python bench/run_e2e.py --split test [--weights PATH] [--no-hybrid]

Writes bench/reports/e2e_<split>_<tag>.csv and prints the metrics. Changes nothing else."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/
sys.path.insert(0, str(ROOT / "bench"))
import anpr_config as cfg  # noqa: E402

YOLO = ROOT / "runs" / "detect" / "runs" / "vehicle_plate" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights")
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--tag", default="final")
    a = ap.parse_args()
    if a.weights:
        cfg.AWIROS_WEIGHTS = Path(a.weights).resolve()
    import pipeline as pl
    from analyze import text_metrics, decision_metrics
    from analyze_hybrid import LABEL_MISMATCH
    P = pl.Pipeline(YOLO, "cpu", use_resolver=False, use_hybrid=not a.no_hybrid)
    info = {"weights": str(P.reader.weights), "inference": P.reader.inference_mode,
            "hybrid": P.hybrid.describe() if P.hybrid else None,
            "ml_verifier": P.verifier is not None}
    print(json.dumps(info), flush=True)
    rows = list(csv.DictReader(open(ROOT / "training" / "splits" / f"{a.split}.csv")))
    out, secs = [], []
    for i, r in enumerate(rows):
        frame = cv2.imread(r["path"])
        t0 = time.perf_counter()
        plates = P.analyze(frame)
        secs.append(time.perf_counter() - t0)
        b = plates[0] if plates else None
        out.append({"image": Path(r["path"]).name, "label": r["label"],
                    "text": b["text"] if b else "", "status": b["status"] if b else "NO PLATE",
                    "ocr_status": b.get("ocr_status", "") if b else "DETECTION_FAILED",
                    "avg": round(b["avg"], 4) if b else "", "ml_probability": b.get("ml_probability") if b else "",
                    "calibrated_probability": b.get("calibrated_probability") if b else "",
                    "hybrid_reasons": "|".join(b.get("hybrid_reasons") or []) if b else "",
                    "tta_result": b.get("tta_result", "") if b else "",
                    "reasons": "|".join(b["decision"].get("reasons") or []) if b else "",
                    "seconds": round(secs[-1], 3)})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)}  {np.mean(secs):.2f}s/img", flush=True)
    rep = ROOT / "bench" / "reports" / f"e2e_{a.split}_{a.tag}.csv"
    with open(rep, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    res = {}
    for k, sel in (("all_images", out), ("without_confirmed_label_mismatches",
                                         [o for o in out if o["image"] not in LABEL_MISMATCH])):
        m = text_metrics([o["text"] for o in sel], [o["label"] for o in sel])
        m.update(decision_metrics([o["text"] for o in sel], [o["label"] for o in sel],
                                  [o["status"] == "SUCCESS" for o in sel]))
        res[k] = m
    res["latency_s"] = {"mean": round(float(np.mean(secs)), 3), "median": round(float(np.median(secs)), 3),
                        "p95": round(float(np.percentile(secs, 95)), 3)}
    res["info"] = info
    (rep.with_suffix(".json")).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
