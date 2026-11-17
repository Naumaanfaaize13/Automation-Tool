"""One command to check that PlateVision works on this machine.

    python verify.py              # environment + model files + unit tests + real-pipeline smoke test
    python verify.py --accuracy   # also score the held-out test split (needs the split's images)
    python verify.py --quick      # skip the unit tests

Exit code 0 = everything passed, 1 = something failed (the failing step is printed in red).
It changes nothing: no logs, uploads or models are written."""
import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/
YOLO_WEIGHTS = ROOT / "runs" / "detect" / "runs" / "vehicle_plate" / "weights" / "best.pt"
EXPECTED = ROOT / "tests" / "fixtures" / "expected_reads.json"
TEST_SPLIT = ROOT / "training" / "splits" / "test.csv"

# Minimum numbers for --accuracy on the 124-image test split. Measured 2026-11-17 with the
# deployed fine-tuned model: see the README "Verify" section. A drop below these = regression.
MIN_EXACT = 0.72              # share of images whose top reading equals the label
MIN_SUCCESS_PRECISION = 0.85  # share of SUCCESS plates whose text is correct

GREEN, RED, YELLOW, END = ("\033[32m", "\033[31m", "\033[33m", "\033[0m") if sys.stdout.isatty() else ("",) * 4
failures, warnings = [], []


def ok(msg):
    print(f"  {GREEN}OK{END}    {msg}")


def fail(msg):
    failures.append(msg)
    print(f"  {RED}FAIL{END}  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  {YELLOW}WARN{END}  {msg}")


def step(title):
    print(f"\n== {title}")


def check_environment():
    step("Python packages")
    # paddle before torch: see pipeline.Pipeline.__init__
    for mod in ("paddle", "numpy", "cv2", "torch", "ultralytics", "safetensors", "flask",
                "sklearn", "joblib", "openpyxl", "yaml"):
        try:
            m = importlib.import_module(mod)
            ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            fail(f"{mod} cannot be imported: {e}")


def check_files():
    step("Model files")
    import anpr_config as cfg
    need = {
        "YOLO plate detector": YOLO_WEIGHTS,
        "Awiros OCR weights (in use)": Path(cfg.AWIROS_WEIGHTS),
        "Awiros character dictionary": Path(cfg.AWIROS_DICT),
        "PaddleOCR clone (ppocr package)": Path(cfg.PADDLEOCR_DIR) / "ppocr",
    }
    for name, p in need.items():
        (ok if p.exists() else fail)(f"{name}: {p}" + ("" if p.exists() else "  (missing)"))
    if Path(cfg.AWIROS_WEIGHTS) != Path(cfg.AWIROS_FINETUNED_WEIGHTS):
        warn("the fine-tuned model is NOT in use (models/awiros_finetuned/model.safetensors missing "
             "or AWIROS_USE_FINETUNED=0); accuracy will be lower")

    cal = Path(cfg.HYBRID_CALIBRATION_PATH)
    if not cal.is_file():
        warn(f"hybrid calibration file missing: {cal}")
    elif Path(cfg.AWIROS_WEIGHTS).is_file():
        from hybrid_verifier import weights_sha256
        want = json.loads(cal.read_text()).get("weights_sha256")
        if want == weights_sha256(cfg.AWIROS_WEIGHTS):
            ok("hybrid calibration matches the loaded OCR weights")
        else:
            warn("hybrid calibration was fitted for different weights, so the gate runs uncalibrated. "
                 "Rerun bench/cache_reads.py and bench/analyze_hybrid.py after retraining")
    ml = ROOT / "models" / "ocr_verifier_hgb.joblib"
    (ok if ml.is_file() else warn)(f"ML verifier: {ml}" + ("" if ml.is_file() else " (missing; rules only)"))


def run_unit_tests():
    step("Unit tests (pytest tests/)")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                       cwd=ROOT, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    if r.returncode == 0:
        ok(f"{last}  [{time.time() - t0:.0f}s]")
    else:
        print(r.stdout[-3000:], r.stderr[-2000:])
        fail(f"unit tests: {last}")


def load_pipeline():
    import pipeline as pl
    t0 = time.time()
    P = pl.Pipeline(YOLO_WEIGHTS, "cpu", use_resolver=False)
    ok(f"pipeline loaded in {time.time() - t0:.1f}s  (OCR: {Path(str(P.reader.weights)).parent.name}, "
       f"{P.reader.inference_mode}; ML verifier: {P.verifier is not None}; hybrid gate: {P.hybrid is not None})")
    return P


def smoke_test(P):
    """Fixed images in the repo must give exactly the readings recorded in expected_reads.json."""
    import cv2
    step("Real pipeline on the sample images (tests/fixtures/expected_reads.json)")
    for item in json.loads(EXPECTED.read_text()):
        img = cv2.imread(str(ROOT / item["image"]))
        if img is None:
            fail(f"{item['image']}: cannot open")
            continue
        t0 = time.time()
        plates = P.analyze(img)
        dt = time.time() - t0
        got = (plates[0]["text"], plates[0]["status"]) if plates else ("", "NO PLATE")
        want = (item["text"], item["status"])
        if got == want:
            ok(f"{item['image']}: {got[0] or '-'} {got[1]}  [{dt:.1f}s]")
        else:
            fail(f"{item['image']}: expected {want[0]} {want[1]}, got {got[0] or '-'} {got[1]}")


def accuracy_test(P):
    import cv2
    step("Accuracy on the held-out test split (training/splits/test.csv)")
    rows = list(csv.DictReader(open(TEST_SPLIT)))
    rows = [r for r in rows if Path(r["path"]).is_file()]
    if not rows:
        warn("test-split images are not on this machine (paths in training/splits/test.csv); skipped")
        return
    n = exact = succ = succ_ok = 0
    t0 = time.time()
    for r in rows:
        plates = P.analyze(cv2.imread(r["path"]))
        text, status = (plates[0]["text"], plates[0]["status"]) if plates else ("", "NO PLATE")
        n += 1
        exact += text == r["label"]
        if status == "SUCCESS":
            succ += 1
            succ_ok += text == r["label"]
    ex, prec = exact / n, (succ_ok / succ if succ else 0.0)
    print(f"        {n} images, {(time.time() - t0) / n:.2f}s/image")
    (ok if ex >= MIN_EXACT else fail)(f"exact match {exact}/{n} = {ex:.1%}  (minimum {MIN_EXACT:.0%})")
    (ok if prec >= MIN_SUCCESS_PRECISION else fail)(
        f"SUCCESS precision {succ_ok}/{succ} = {prec:.1%}  (minimum {MIN_SUCCESS_PRECISION:.0%}); "
        f"{succ - succ_ok} wrong plates shown as SUCCESS")
    ok(f"SUCCESS coverage {succ}/{n} = {succ / n:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accuracy", action="store_true", help="also score the test split")
    ap.add_argument("--quick", action="store_true", help="skip the unit tests")
    a = ap.parse_args()
    os.chdir(ROOT)

    check_environment()
    check_files()
    if not a.quick:
        run_unit_tests()
    if not failures:
        step("Loading the production pipeline")
        try:
            P = load_pipeline()
            smoke_test(P)
            if a.accuracy:
                accuracy_test(P)
        except Exception as e:  # noqa: BLE001
            fail(f"pipeline error: {type(e).__name__}: {e}")
    else:
        warn("pipeline checks skipped because of the failures above")

    print()
    if failures:
        print(f"{RED}VERIFY FAILED{END}: {len(failures)} problem(s), {len(warnings)} warning(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{GREEN}VERIFY PASSED{END}" + (f" with {len(warnings)} warning(s)" if warnings else ""))


if __name__ == "__main__":
    main()
