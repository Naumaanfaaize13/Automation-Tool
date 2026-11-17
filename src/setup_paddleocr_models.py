"""Download the PaddleOCR DET / CLS / REC inference models used as the second OCR engine.

    python src/setup_paddleocr_models.py            # download what is missing, verify what is there
    python src/setup_paddleocr_models.py --check    # only verify, never download

Models (official PaddleOCR inference exports for Paddle 3.0.0+, loaded with the PaddleOCR clone
that Awiros already uses - the `paddleocr` / `paddlex` pip packages are NOT needed):
  DET  PP-OCRv5_mobile_det          text-line detection inside the YOLO plate crop
  CLS  PP-LCNet_x1_0_textline_ori   0 / 180 degree text-line orientation
  REC  en_PP-OCRv5_mobile_rec       English/digit recognition (a different, smaller network
                                    than the PP-OCRv5 *server* rec that Awiros was trained from)

No duplicate downloads: a model whose folder already holds inference.json + inference.pdiparams
(matching models/paddleocr/manifest.json when present) is skipped. Archives are extracted with
tarfile's 'data' filter (no absolute paths, no '..', no links outside the folder) and deleted.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import anpr_config as cfg

BASE = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
MODELS = {
    "det": "PP-OCRv5_mobile_det",
    "cls": "PP-LCNet_x1_0_textline_ori",
    "rec": "en_PP-OCRv5_mobile_rec",
}
REQUIRED = ("inference.json", "inference.pdiparams", "inference.yml")
MIN_PADDLE = (3, 0)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_runtime():
    """Paddle >= 3.0 (PIR .json models) and the PaddleOCR clone's tools/infer runners."""
    problems = []
    try:
        import paddle
        ver = tuple(int(x) for x in paddle.__version__.split(".")[:2])
        if ver < MIN_PADDLE:
            problems.append(f"paddlepaddle {paddle.__version__} is too old (need >= 3.0)")
        info = f"paddlepaddle {paddle.__version__}, CUDA build: {paddle.device.is_compiled_with_cuda()}"
    except ImportError:
        problems.append("paddlepaddle is not installed")
        info = "paddlepaddle missing"
    for rel in ("ppocr/__init__.py", "tools/infer/predict_det.py", "tools/infer/predict_cls.py",
                "tools/infer/predict_rec.py", "tools/infer/utility.py"):
        if not (Path(cfg.PADDLEOCR_DIR) / rel).is_file():
            problems.append(f"PaddleOCR clone incomplete: {cfg.PADDLEOCR_DIR / rel} missing")
    for mod in ("pyclipper", "shapely"):      # DB post-processing of the detector
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"{mod} is not installed (pip install {mod})")
    return info, problems


def model_ok(folder):
    return all((folder / f).is_file() for f in REQUIRED)


def download(name, dest):
    url = f"{BASE}/{name}_infer.tar"
    print(f"  downloading {url}")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        tar_path = Path(tmp) / f"{name}.tar"
        with urllib.request.urlopen(url, timeout=120) as r, open(tar_path, "wb") as f:
            shutil.copyfileobj(r, f)
        with tarfile.open(tar_path) as t:
            t.extractall(tmp, filter="data")
        src = Path(tmp) / f"{name}_infer"
        if not model_ok(src):
            raise RuntimeError(f"{name}: archive does not contain {', '.join(REQUIRED)}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only, do not download")
    args = ap.parse_args()

    info, problems = check_runtime()
    print(f"Runtime: {info}; PaddleOCR clone: {cfg.PADDLEOCR_DIR}")
    if problems:
        print("Not compatible:\n  " + "\n  ".join(problems))
        return 1

    root = Path(cfg.PADDLEOCR_MODEL_DIR)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    failed = False
    for role, name in MODELS.items():
        folder = root / name
        known = manifest.get(name, {}).get("sha256", {})
        if model_ok(folder) and (not known or all(
                sha256(folder / f) == h for f, h in known.items())):
            print(f"{role.upper()} {name}: present, skipped")
        elif args.check:
            print(f"{role.upper()} {name}: MISSING or changed"); failed = True
            continue
        else:
            print(f"{role.upper()} {name}:")
            download(name, folder)
        manifest[name] = {"role": role, "url": f"{BASE}/{name}_infer.tar",
                          "sha256": {f: sha256(folder / f) for f in REQUIRED}}
    if not args.check:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print("OK" if not failed else "Some models are missing: run without --check.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
