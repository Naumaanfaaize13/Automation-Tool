# PlateVision

Batch Indian number-plate reader: YOLO11n (`best.pt`) finds the plate, OpenCV prepares the crop,
**Awiros ANPR OCR** (PP-OCRv5, PaddlePaddle) reads it, an Indian registration validator checks it,
a HistGradientBoosting verifier scores it, and a Flask dashboard shows everything with manual
review and Excel export. (PARSeq was the OCR engine until 2026-11-05; it is kept only as a
benchmark baseline - see "OCR engine" below.)

The system prefers **no answer over a wrong answer**: anything uncertain is returned as
`MANUAL_REVIEW_REQUIRED` with, at most, a suggestion.

## Hybrid accuracy update (2026-11-11)

Full report: `bench/reports/HYBRID_REPORT.md`. Measured on 124 untouched, human-audited test images:

| | Exact match | Auto-approved precision | Coverage |
|---|---|---|---|
| Before (original Awiros) | 65.3 % | 91.3 % | 64.5 % |
| Now, default (strict gate) | 75.0 % | 97.6 % | 33.9 % |
| Now, `HYBRID_AUTO_MIN_P=0.85` (balanced) | 75.0 % | 95.7 % | 56.5 % |

The 97 % exact-match target is **not** reached. 27 of the 31 remaining errors are two-row bike plates;
more verified training plates are the way forward.

* **Fine-tuned Awiros** (`models/awiros_finetuned/model.safetensors`, loaded automatically): the
  original backbone is kept, and the SVTR neck + CTC head were trained on 225 audited plates. The original checkpoint is untouched.
* **Hybrid gate** (`hybrid_verifier.py`): a SUCCESS also needs a calibrated P(correct) of at least the
  threshold in `models/calibration/awiros_calibration.json`. It can only send a plate to manual review; it never approves on its own and never changes text.
* **CPU speed**: denormal weights are zeroed when the model loads (identical output, 67x faster eager mode).
* New result columns: `calibrated_probability`, `hybrid_reasons`, `tta_result`, `ocr_model`
  (older logs are upgraded automatically, with a backup).
* Rollback: `AWIROS_USE_FINETUNED=0` (original model), `HYBRID_VERIFIER=0` (no gate).
* After retraining: `bench/cache_reads.py` → `bench/analyze_hybrid.py` refits calibration and thresholds.

## Run

The Awiros files live outside the project in `AWIROS_DIR` (looked up in this order: env var
`AWIROS_DIR`, `./awiros`, `../Awiros-ANPR-OCR`, `~/Awiros-ANPR-OCR` - on Windows that is
`C:\Users\Lenovo\Awiros-ANPR-OCR`). It needs `model.safetensors`, `en_dict.txt` and a PaddleOCR
clone in `AWIROS_DIR\PaddleOCR` (the model is built with PaddleOCR's `ppocr` package).

Windows (RTX 3050):

```powershell
cd C:\Users\Lenovo\Awiros-ANPR-OCR
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
cd C:\Users\Lenovo\Vehicle_Number_Detection
.venv\Scripts\activate
pip install -r requirements.txt
# GPU: replace the CPU wheel with the CUDA one matching your driver (see paddlepaddle.org.cn), e.g.
pip uninstall -y paddlepaddle
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python server.py --device cuda   # YOLO and Awiros on the GPU; each falls back to CPU if CUDA is missing
```

Linux:

```bash
cd ~/Documents/Vehicle_Number_Detection
python3 -m venv .venv-linux
.venv-linux/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-linux/bin/pip install -r requirements.txt
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git ../Awiros-ANPR-OCR/PaddleOCR
.venv-linux/bin/python server.py
```

Check the OCR model alone: `python src/awiros_ocr.py path/to/plate_crops/`.

Open http://127.0.0.1:5000. Flags: `--weights PATH`, `--device cpu|cuda`, `--host`, `--port`.
The ML verifier needs `scikit-learn` and `joblib` (added to requirements.txt).

## Processing workflow

1. **Detect** (YOLO11 `best.pt`, unchanged). Touching boxes are merged into one plate.
   Tick *"Images are already plate crops"* to skip detection (e.g. pump-camera plate crops).
   **Automatic crop mode** (`AUTO_CROP_MODE` in `anpr_config.py`): when an image is itself
   plate-shaped (aspect 0.9-7.5, at most 1000 px) and YOLO found nothing, or a box covering
   >= 20 % of it, the whole image is read as well and the better reading is kept. YOLO tends to
   find a box *inside* an already-cropped plate and cut characters off. If the two readings
   confidently disagree, the plate goes to manual review.
2. **Read the original crop** with Awiros ANPR OCR and keep the real per-character probabilities
   (CTC softmax at each character's strongest frame, upper/lower case merged). Dual-row plates
   are read whole (the model was trained on them; PARSeq needed them split in two). If it is a valid, very confident
   plate, stop. Otherwise read wider crops and each detector box.
3. **Measure crop quality** (brightness, contrast, sharpness, noise, size, tilt) and **retry**
   with only the OpenCV variants that fit: perspective/deskew for tilt, low-light enhancement,
   CLAHE, denoise, sharpen, upscale; then grayscale/adaptive/Otsu as a last round.
   At most 6 variants in 2 rounds, batched in one OCR call each. Identical crops and identical
   uploaded files are processed once (cache).
4. **Validate** every reading (`indian_plate_validator.py`): known state/UT code, RTO code,
   0-3 series letters, 4-digit number (not 0000), or BH series (year 21+).
5. **Position-aware alternatives**: a character is only swapped for another one the model itself
   gave real probability to (e.g. `0` for `O`), at most 2 swaps, never adding or removing
   characters. Such a reading is **only a suggestion** (REVIEW), never SUCCESS.
6. **Decide** (`plate_decision.py`):
   * **SUCCESS**: valid format with no warnings, average ≥ 90 %, every character ≥ 80 %,
     no other confident reading disagrees, detection and crop OK. Only SUCCESS gets a
     `final_registration_number`.
   * **REVIEW** (`MANUAL_REVIEW_REQUIRED`): a plausible plate that fails any check; the
     reasons are listed.
   * **REJECT**: text that is not a registration number (brand names, stickers, signs).
   * **NO PLATE**: the detector found nothing.
7. **ML verifier** (`ocr_verifier.py`, a scikit-learn `HistGradientBoostingClassifier` in
   `models/ocr_verifier_hgb.joblib`): for readings that passed the hard rules (valid format,
   read directly, no conflict, usable crop) it estimates P(reading is correct) from 35 features
   (character probabilities, agreement between attempts, retries, image quality, detection) and
   replaces the fixed 90 % / 80 % thresholds: SUCCESS when P >= the trained threshold (0.923) and
   no character is below 30 %. It can also hold back a rule-SUCCESS. It never changes the text.
   Results on held-out images: `evaluation/ml_verifier_vs_rules.md`.
   Retrain: `python scripts/train_verifier.py --manifest ...` (add `--reuse` to skip re-running OCR).
   Run without it: `python server.py --no-ml`.
8. **Manual verification** in the dashboard: correct or confirm; the OCR result is never
   overwritten, and every action goes to `results/corrections.csv` (audit trail).

Format validity is not proof that a registration exists. That needs an official lookup (VAHAN),
which is not part of this project.

Thresholds are in `src/anpr_config.py`. They are starting values: calibrate them with `scripts/evaluate.py`.

## Project layout

```
server.py                 dashboard (Flask app, background jobs, API)   -> python server.py
verify.py                 one-command health check                     -> python verify.py
static/index.html         dashboard page
src/                      core code used by the server
  anpr_config.py            all thresholds and model paths
  pipeline.py               detection, OCR attempts, retries, dedupe, saving
  awiros_ocr.py             OCR engine: Awiros ANPR OCR (loading, CTC decoding, confidences, GPU/CPU)
  plate_preprocess.py       OpenCV quality assessment and enhancement variants
  image_enhance.py          enhancement plans for the hybrid gate
  indian_plate_validator.py normalisation, state/RTO/series/number validation, candidates
  plate_decision.py         SUCCESS / REVIEW / REJECT decision with reasons
  ocr_verifier.py           HistGradientBoosting verifier (features, apply)
  ocr_calibration.py        calibrated P(correct) for the hybrid gate
  hybrid_verifier.py        final gate on SUCCESS
  review_resolver.py        second layer for REVIEW plates (off by config)
  dashboard_log.py          results/log.csv, corrections, Excel export
  finetune_awiros.py        fine-tunes Awiros on verified plates (started from the dashboard)
  plate_reader.py           PARSeq wrapper, benchmark baseline only
  paddle_ocr_engine.py, setup_paddleocr_models.py   optional PaddleOCR engine (off)
scripts/                  command-line tools (training, labelling, evaluation)
  train_yolo.py             train the YOLO plate detector (needs dataset/)
  finetune_awiros_head.py   CPU fine-tuning of the OCR head (used for the deployed model)
  train_verifier.py, train_resolver.py   train the ML verifier / review resolver
  label_tool.py             web page to verify plate labels before fine-tuning
  evaluate.py, legacy_pipeline.py        accuracy on labelled images, before vs after
  benchmark_awiros_vs_parseq.py          OCR benchmark against PARSeq
bench/                    benchmark scripts + reports (bench/reports/HYBRID_REPORT.md)
evaluation/               older evaluation reports
models/                   small models in git (ML verifier, calibration); large weights stay local
runs/detect/runs/vehicle_plate/weights/best.pt   YOLO plate detector
training/splits/          train / val / calib / test splits (labels)
samples/                  sample images used by verify.py
tests/                    unit tests (python -m pytest tests)
requirements.txt          runtime packages; requirements/ = optional extras
```

Core modules import each other by plain name (`import pipeline`); `server.py`, `verify.py`,
`scripts/` and `bench/` add `src/` to the import path themselves, so run them from any folder.

## API (additions; existing endpoints unchanged)

| Method and path | Purpose |
|---|---|
| `POST /api/jobs` | extra optional form fields: `weakest` (0.3-0.99), `crop_mode` (1/0) |
| `GET /api/results?status=REJECT` / `CORRECTED` | new filters |
| `POST /api/results/<row_id>/review` | JSON `{corrected_text, reason, reviewer}`; 400 if not a valid format |
| `GET /api/results/<row_id>/audit` | correction history for a row |
| `GET /files/enhanced/<name>` | enhanced crop used for the reading |

## OCR speed on CPU

Paddle's eager mode does not use oneDNN for the backbone convolutions (~7 s per crop). On CPU
`awiros_ocr.py` therefore exports the model once to a static inference graph
(`models/awiros_inference/`, rebuilt automatically when the weights change) and runs it with
oneDNN: ~150 ms per crop, identical output (max difference 2e-5). `AWIROS_FAST_CPU=0` disables it.

## Fine-tuning the OCR model on your plates

**Everyday loop (dashboard):** open a REVIEW plate, type or confirm the correct number, and
save. The row then shows **✓ VERIFIED** with your text as the final registration (the OCR
output stays in the audit trail), and the un-enhanced crop it was read from
(`results/train_crops/`) becomes a training sample. Once 150+ plates are verified, the
*Improve the OCR model* panel trains a new model (`--device cuda` server) and installs it
without a restart, but only if it beats the current model on held-out verified plates.

The manifest labels are **unaudited** (PARSeq / LLM / "hiconf" sources), so they are not used
for training directly. Steps:

1. **Verify labels**: `python scripts/label_tool.py --manifest ~/Desktop/plates/car/manifest.csv
   --manifest ~/Desktop/plates/bike/manifest.csv --manifest ~/Desktop/plates/others/manifest.csv
   --folder ~/Desktop/correct`, open http://127.0.0.1:5001. Plates where the model and the
   dataset label disagree come first (the useful training examples). Results go to
   `training/verified_labels.csv`. Corrections made in the dashboard are used as well.
2. **Train** (GPU: Windows RTX 3050; CPU is only good for a `--max-steps 2` smoke test):
   `python src/finetune_awiros.py --device cuda --epochs 30 --deploy`.
   A held-out set (split by vehicle) is read with the original model first; the new model is
   written to `models/awiros_finetuned/model.safetensors` **only if it beats the original** on
   that set. `awiros_ocr.py` then uses it automatically (`AWIROS_USE_FINETUNED=0` = original).
   Report: `models/awiros_finetuned/report.json`.
3. Re-run `evaluate.py` on the full pipeline and check the SUCCESS precision before relying on it.
   Copy `training/` and `models/awiros_finetuned/` between machines as needed.

## Measure accuracy

```bash
python scripts/evaluate.py --manifest ~/Desktop/plates/car/manifest.csv \
                   --manifest ~/Desktop/plates/bike/manifest.csv --detect both --degrade
python scripts/evaluate.py --folder my_images        # ground truth = file name before the first '_'
```

Writes `evaluation/report.md`, `report.json`, `predictions.csv`. Labels must be checked by a
person for the numbers to mean "accuracy"; system-generated labels only measure agreement.

## Test

```bash
pip install pytest playwright && playwright install chromium
python -m pytest tests -q
```

## Verify (one command)

```bash
.venv-linux/bin/python verify.py              # packages, model files, unit tests, real pipeline on sample images
.venv-linux/bin/python verify.py --accuracy   # + score the 124-image held-out test split
```

It ends with `VERIFY PASSED` (exit code 0) or `VERIFY FAILED` and the list of problems.
The sample images must give exactly the readings in `tests/fixtures/expected_reads.json`;
`--accuracy` fails if exact match drops below 72 % or SUCCESS precision below 85 %.
Measured 2026-11-17: exact 93/124 = 75.0 %, SUCCESS 99/124, of which 87 are correct (87.9 %).

## What is NOT in git (copy these separately to a new machine)

The repository holds the code, the small models and the reports. These stay on disk only:

| Path | Why | Needed to run? |
|---|---|---|
| `models/awiros_finetuned/model.safetensors` | 143 MB, above GitHub's 100 MB limit | yes (else the original Awiros weights are used, lower accuracy) |
| `../Awiros-ANPR-OCR/` (model, `en_dict.txt`, PaddleOCR clone) | separate download | yes |
| `models/awiros_inference/` | rebuilt automatically from the weights | no |
| `dataset/` | deleted 2026-11-17; YOLO training data (Roboflow "YOLOv8 number plate detection v1"), only `scripts/train_yolo.py` needs it | no |
| `results/`, `bench/cache/` | dashboard uploads/logs, caches | no |
| `backups/`, `backup_*/`, `.venv*/` | old copies and environments | no |
