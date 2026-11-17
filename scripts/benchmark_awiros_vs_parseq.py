"""Awiros ANPR OCR: unit tests (pytest) and an OCR benchmark against PARSeq.

Unit tests (need the Awiros files; skipped when they are missing):
    python -m pytest scripts/benchmark_awiros_vs_parseq.py -q

Benchmark (same plate crops for both engines, OCR only - no detector, no decision rules):
    python scripts/benchmark_awiros_vs_parseq.py --benchmark --manifest ~/Desktop/plates/car/manifest.csv \
        --manifest ~/Desktop/plates/bike/manifest.csv --manifest ~/Desktop/plates/others/manifest.csv \
        --engines awiros,parseq --degrade --device cpu
Writes evaluation/awiros_vs_parseq/{report.md, report.json, predictions.csv}.

Categories: clean, synthetic blur / night (low light) / tilt / lowres / noise / occlude
(partially visible: the true number is NOT fully visible), dual-row vs single-row (by crop
aspect), and 'unreadable' (images labelled llm_invalid: no readable registration).
Metrics: exact full-registration accuracy, character accuracy, CER, mean inference time per
crop (batch of 1), throughput (batched), peak GPU memory (GPU only).
Accuracy is only as good as the labels: see the label_source breakdown in the report.
"""
import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
import sys  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/

import cv2
import numpy as np
import pytest

import anpr_config as cfg
import indian_plate_validator as ipv

AWIROS_SAMPLES = {   # Awiros' own showcase images and the ground truth from their README
    "10_14_68e6fcf21e55ac002f310971_awi_1.jpg": "HR12AX8522",
    "10_20_68f0b5ea88aefc002f543e60_awi_1.jpg": "HR35M2576",
    "10_20_68fc5fb360b026003091b0eb_awi_1.jpg": "HR34M4007",
    "10_20_69027403e20fe9002fceedaa_awi_1.jpg": "RJ29GB2097",
    "10_21_68ecddf9925604002f70876a_awi_1.jpg": "HR38AB2421",
    "9_13_68f45a82b5e646002f924bbd_awi_1.jpg": "DL9CBH1669",
    "9_16_68f325be3a50df002e43a509_awi_1.jpg": "HR51BV8223",
    "9_25_6901aa035e29cc002f516180_awi_1.jpg": "HR46E0227",
    "two_row-set-1_1_1420_.jpg": "UP14BQ9208",
}


# ============================================================================ unit tests
@pytest.fixture(scope="module")
def reader():
    from awiros_ocr import AwirosReader, AwirosUnavailable
    try:
        return AwirosReader("cpu")
    except AwirosUnavailable as e:
        pytest.skip(f"Awiros not available: {e}")


def test_preprocess_matches_paddleocr_resize_norm_img():
    from awiros_ocr import preprocess
    img = np.full((40, 100, 3), 255, np.uint8)               # 48*100/40 = 120 px wide
    x = preprocess(img)
    assert x.shape == (3, 48, 320) and x.dtype == np.float32
    assert np.allclose(x[:, :, :120], 1.0)                   # white -> +1
    assert np.allclose(x[:, :, 120:], 0.0)                   # padding is 0 AFTER normalising
    wide = preprocess(np.zeros((10, 1000, 3), np.uint8))     # wider than 320: squeezed, no pad
    assert np.allclose(wide, -1.0)


def test_preprocess_accepts_grayscale():
    from awiros_ocr import preprocess
    assert preprocess(np.zeros((30, 90), np.uint8)).shape == (3, 48, 320)


def test_ctc_decode_merges_repeats_and_reports_real_probabilities(reader):
    T, C = 40, len(reader.ctc_chars)
    probs = np.full((T, C), 1e-4, np.float32)
    seq = ["M", "M", "<blank>", "H", "1", "1", "<blank>", "1", "2"] + ["<blank>"] * (T - 9)
    for t, ch in enumerate(seq):
        probs[t, reader.ctc_chars.index(ch)] = 0.9
    probs[3, reader.ctc_chars.index("h")] = 0.05             # lower case adds to 'H'
    probs[8, reader.ctc_chars.index("Z")] = 0.08             # an alternative for '2'
    probs /= probs.sum(1, keepdims=True)
    r = reader._decode(probs)
    assert r["raw"] == "MH112"                                # blank separates the two 1s
    assert [c["char"] for c in r["chars"]] == list("MH112")
    h = r["chars"][1]
    assert h["p"] == pytest.approx(probs[3, reader.ctc_chars.index("H")]
                                   + probs[3, reader.ctc_chars.index("h")], rel=1e-5)
    alt_char, alt_p = r["chars"][4]["alts"][1]             # runner-up at the '2' frame
    assert alt_char == "Z" and alt_p == pytest.approx(        # upper + lower case summed
        float(probs[8, reader.ctc_chars.index("Z")] + probs[8, reader.ctc_chars.index("z")]))
    assert r["weakest"] == min(c["p"] for c in r["chars"])


def test_reads_awiros_sample_plates(reader):
    folder = cfg.AWIROS_DIR / "images"
    files = [folder / f for f in AWIROS_SAMPLES if (folder / f).exists()]
    if not files:
        pytest.skip("Awiros sample images not downloaded")
    reads = reader.read_plate_chars([cv2.imread(str(f)) for f in files])
    got = {f.name: ipv.normalize(r["raw"]) for f, r in zip(files, reads)}
    wrong = {k: v for k, v in got.items() if v != AWIROS_SAMPLES[k]}
    assert not wrong, wrong
    assert any(r["two_line"] for r in reads)                  # dual-row sample read whole
    assert all(0.0 < r["avg"] <= 1.0 and r["infer_ms"] >= 0 for r in reads)


def test_cache_and_empty_crop(reader):
    img = cv2.imread(str(ROOT / "samples" / "plate_crop_1.jpg"))
    a, b, c = reader.read_chars([img, img.copy(), np.zeros((0, 0, 3), np.uint8)])
    assert a["raw"] == b["raw"]
    assert c["error"] and c["raw"] == ""


def test_inference_error_is_reported_not_raised(reader, monkeypatch):
    def boom(_batch):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(reader, "_forward", boom)
    reader._cache.clear()
    r, = reader.read_chars([np.random.default_rng(1).integers(0, 255, (40, 160, 3), np.uint8)])
    assert r["raw"] == "" and "simulated failure" in r["error"]


def test_missing_weights_raise_clear_error(tmp_path):
    from awiros_ocr import AwirosReader, AwirosUnavailable
    with pytest.raises(AwirosUnavailable, match="model.safetensors"):
        AwirosReader("cpu", weights=tmp_path / "model.safetensors")


def test_detailed_status_mapping():
    import plate_decision as pdec

    def plate(status, reasons, valid=True, raw="MH12DE1433", text="MH12DE1433"):
        return {"raw": raw, "decision": {"status": status, "reasons": reasons, "text": text,
                                         "validation": {"is_valid_format": valid}}}
    assert pdec.detailed_status(plate("SUCCESS", [])) == "SUCCESS"
    assert pdec.detailed_status(plate("REVIEW", ["UNCERTAIN_CHARACTER"])) == "LOW_CONFIDENCE"
    assert pdec.detailed_status(plate("REVIEW", ["ML_VERIFIER_LOW_PROBABILITY",
                                                 "LOW_AVERAGE_CONFIDENCE"])) == "LOW_CONFIDENCE"
    assert pdec.detailed_status(plate("REVIEW", ["CONFLICTING_READINGS", "UNCERTAIN_CHARACTER"])) \
        == "MANUAL_REVIEW_REQUIRED"
    assert pdec.detailed_status(plate("REVIEW", ["PARTIAL_OR_INVALID_PLATE_TEXT"], valid=False)) \
        == "INVALID_FORMAT"
    assert pdec.detailed_status(plate("REJECT", ["NOT_A_REGISTRATION_NUMBER"], valid=False,
                                      raw="PETROL", text="PETROL")) == "INVALID_FORMAT"
    assert pdec.detailed_status(plate("REJECT", ["UNUSABLE_CROP"], valid=False, raw="", text="")) \
        == "OCR_FAILED"


def test_validator_never_forces_characters():
    v = ipv.validate("MH12DE143B")                      # B in the number: invalid, not "fixed"
    assert not v["is_valid_format"] and v["normalized_text"] == "MH12DE143B"
    assert ipv.normalize("tn-38 ab 1234") == "TN38AB1234"
    for good in ("TN38AB1234", "MH12DE1433", "KA03MN5678", "KL07CD1234", "HR26DK9876",
                 "AP39BC4567"):
        assert ipv.validate(good)["is_valid_format"], good


# ============================================================================ benchmark
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_engine(name, device):
    if name == "awiros":
        from awiros_ocr import AwirosReader
        return AwirosReader(device)
    from plate_reader import PARSeqReader
    r = PARSeqReader("parseq", "cuda" if device in ("cuda", "0") else "cpu")
    r.engine_name = "PARSeq"
    return r


def gpu_peak_mb(name, reader):
    if name == "awiros":
        return reader.gpu_memory_mb()
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
    except Exception:
        return None


def summarize(rows):
    lab = [r for r in rows if r["gt"]]
    out = {"n": len(rows), "labelled": len(lab)}
    if lab:
        out["exact_pct"] = round(100 * sum(r["pred"] == r["gt"] for r in lab) / len(lab), 2)
        out["char_acc_pct"] = round(100 * statistics.mean(
            max(0.0, 1 - lev(r["pred"], r["gt"]) / len(r["gt"])) for r in lab), 2)
        out["cer_pct"] = round(100 * sum(lev(r["pred"], r["gt"]) for r in lab)
                               / sum(len(r["gt"]) for r in lab), 2)
    unread = [r for r in rows if r["category"] == "unreadable"]
    if unread:   # lower is better: a "valid plate" read from an unreadable image is a hallucination
        out["valid_plate_output_pct"] = round(
            100 * sum(ipv.validate(r["pred"])["is_valid_format"] for r in unread) / len(unread), 2)
    ms = [r["ms"] for r in rows]
    out["ms_per_crop_mean"] = round(statistics.mean(ms), 2)
    out["ms_per_crop_p95"] = round(sorted(ms)[int(0.95 * (len(ms) - 1))], 2)
    return out


def benchmark(args):
    import evaluate as ev
    items = ev.load_items(argparse.Namespace(manifest=args.manifest, folder=args.folder,
                                             labels=None, every=1, limit=args.limit))
    with_src = {}
    for m in args.manifest or []:
        m = Path(m).expanduser()
        for r in csv.DictReader(open(m, newline="", encoding="utf-8")):
            with_src[str(m.parent / r["path"])] = r.get("plate_source", "")
    rng = np.random.default_rng(0)
    work = []
    for it in items:
        img = cv2.imread(it["path"])
        if img is None:
            continue
        gt = it["gt"] if ipv.validate(it["gt"])["is_valid_format"] else ""
        src = with_src.get(it["path"], it["label_source"])
        if src == "llm_invalid":
            cat = "unreadable"
        elif not gt:
            continue                                   # no label, not known to be unreadable
        else:
            cat = "clean"
        work.append(dict(path=it["path"], img=img, gt=gt, category=cat, label_source=src or "none",
                         day_night=it["day_night"], vehicle_type=it["vehicle_type"],
                         layout="dual_row" if (img.shape[1] / img.shape[0]) < cfg.TWO_LINE_ASPECT
                         else "single_row"))
    if args.degrade:
        clean = [w for w in work if w["category"] == "clean"][: args.degrade_limit]
        for kind in ev.DEGRADATIONS:
            for w in clean:
                work.append(dict(w, img=ev.degrade(w["img"], kind, rng), category=kind))
    print(f"{len(work)} crops ({sum(w['category'] == 'clean' for w in work)} clean labelled)")

    engines = args.engines.split(",")
    preds, report = [], {"settings": {k: v for k, v in vars(args).items()}, "engines": {}}
    for name in engines:
        t_load = time.perf_counter()
        reader = load_engine(name, args.device)
        load_s = time.perf_counter() - t_load
        reader.read_plate_chars([work[0]["img"]])                     # warm-up, not timed
        reader._cache.clear()
        for k, w in enumerate(work):                                  # latency: batch of 1
            t0 = time.perf_counter()
            r = reader.read_plate_chars([w["img"]])[0]
            ms = (time.perf_counter() - t0) * 1000
            reader._cache.clear()
            preds.append({"engine": name, "file": w["path"], "category": w["category"],
                          "layout": w["layout"], "day_night": w["day_night"],
                          "vehicle_type": w["vehicle_type"], "label_source": w["label_source"],
                          "gt": w["gt"], "pred": ipv.normalize(r["raw"]), "raw": r["raw"],
                          "avg_conf": round(r["avg"], 4), "weakest": round(r["weakest"], 4),
                          "ms": round(ms, 2)})
            if (k + 1) % 200 == 0:
                print(f"  {name}: {k + 1}/{len(work)}", flush=True)
        imgs = [w["img"] for w in work[:256]]                         # throughput: batched
        t0 = time.perf_counter()
        reader.read_plate_chars(imgs)
        thr = len(imgs) / (time.perf_counter() - t0)
        reader._cache.clear()
        report["engines"][name] = {"load_seconds": round(load_s, 2),
                                   "throughput_crops_per_s": round(thr, 1),
                                   "gpu_peak_mb": gpu_peak_mb(name, reader),
                                   "device": getattr(reader, "device", args.device)}
        del reader

    groups = defaultdict(list)
    for p in preds:
        e = p["engine"]
        groups[(e, "category", p["category"])].append(p)
        if p["category"] == "clean":
            groups[(e, "all_clean", "all")].append(p)
            groups[(e, "layout", p["layout"])].append(p)
            groups[(e, "day_night", p["day_night"] or "?")].append(p)
            groups[(e, "label_source", p["label_source"])].append(p)
            if p["label_source"] != "parseq":
                groups[(e, "label_source", "NOT_parseq (unbiased subset)")].append(p)
    res = {f"{e}|{k}={v}": summarize(ps) for (e, k, v), ps in sorted(groups.items())}
    report["results"] = res
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with open(out / "predictions.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(preds[0]))
        wr.writeheader()
        wr.writerows(preds)

    cols = ["labelled", "exact_pct", "char_acc_pct", "cer_pct", "valid_plate_output_pct",
            "ms_per_crop_mean", "ms_per_crop_p95"]
    keys = sorted({k.split("|", 1)[1] for k in res})
    lines = ["# Awiros ANPR OCR vs PARSeq (OCR only, identical crops)", "",
             "| slice | engine | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k in keys:
        for e in engines:
            r = res.get(f"{e}|{k}")
            if r:
                lines.append(f"| {k} | {e} | " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines += ["", "| engine | device | load s | throughput crops/s (batched) | GPU peak MB |",
              "|---|---|---|---|---|"]
    for e, v in report["engines"].items():
        lines.append(f"| {e} | {v['device']} | {v['load_seconds']} | {v['throughput_crops_per_s']} "
                     f"| {v['gpu_peak_mb'] if v['gpu_peak_mb'] is not None else 'n/a (CPU)'} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}/report.md, report.json, predictions.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--manifest", action="append")
    ap.add_argument("--folder", action="append")
    ap.add_argument("--engines", default="awiros,parseq")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--degrade-limit", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "awiros_vs_parseq"))
    args = ap.parse_args()
    if not args.benchmark:
        raise SystemExit(pytest.main([__file__, "-q"]))
    benchmark(args)


if __name__ == "__main__":
    main()
