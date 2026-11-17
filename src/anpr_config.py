"""All tunable thresholds in one place. These are STARTING values, not calibrated ones:
run evaluate.py on a labelled set and adjust. Confidence thresholds alone do not guarantee accuracy."""

# ---- decision (SUCCESS / REVIEW / REJECT) --------------------------------------------
ACCEPT_AVG_CONF = 0.90        # geometric mean of per-character probabilities
ACCEPT_WEAKEST_CHAR = 0.80    # lowest single-character probability
CONFLICT_MIN_CONF = 0.80      # another valid reading this confident = unresolved conflict
DET_CONF_MIN_SUCCESS = 0.30   # YOLO box confidence needed for SUCCESS (not used in crop mode)

# ---- automatic crop mode --------------------------------------------------------------
# Pump-camera uploads are often already plate crops; YOLO then finds a box INSIDE the plate and
# cuts characters off. When an image looks like a crop, the whole image is read as well and the
# better reading is kept (a confident disagreement between the two goes to manual review).
AUTO_CROP_MODE = True
AUTO_CROP_ASPECT = (0.9, 7.5)     # width / height of an image that may be a plate crop
AUTO_CROP_MAX_SIDE = 1000         # px: larger images are full scenes
AUTO_CROP_MIN_BOX_FRACTION = 0.20 # a YOLO box covering >= this share of the image = crop

# ---- plate geometry / crop quality ----------------------------------------------------
MIN_CROP_HEIGHT = 12          # px: smaller crops cannot be read reliably
MIN_CROP_WIDTH = 30
PLATE_ASPECT_RANGE = (0.9, 7.5)   # width / height of a plausible plate box
TWO_LINE_ASPECT = 2.2         # below this width/height a plate is read as two lines

# ---- OCR attempts ---------------------------------------------------------------------
BASE_H_PADS = (0.06, 0.25, 0.50)  # horizontal padding (fraction of box width)
V_PAD = 0.12
EARLY_EXIT_AVG = 0.95         # tight crop this confident and valid -> skip the wider crops
EARLY_EXIT_WEAKEST = 0.85
MAX_RETRY_VARIANTS = 6        # extra preprocessing variants per plate (one batched PARSeq call)
MAX_RETRY_ROUNDS = 2
OCR_CACHE_SIZE = 4096         # identical crops are read once
IMAGE_CACHE_SIZE = 512        # identical uploaded files are analysed once

# ---- position-aware candidate generation ---------------------------------------------
ALT_MIN_PROB = 0.02           # an alternative character needs at least this model probability
ALT_MIN_PROB_NON_CONFUSABLE = 0.15
MAX_SUBSTITUTIONS = 2
CONFUSABLE = {frozenset(p) for p in (("O", "0"), ("D", "0"), ("Q", "0"), ("I", "1"), ("L", "1"),
                                     ("T", "1"), ("B", "8"), ("S", "5"), ("Z", "2"), ("G", "6"),
                                     ("A", "4"), ("U", "V"), ("M", "H"), ("W", "V"), ("K", "X"),
                                     ("V", "Y"), ("H", "U"))}

# ---- registration format ------------------------------------------------------------
SERIES_WARN_LETTERS = set("IO")   # not seen in series letters; a read containing them goes to review
WARN_EMPTY_SERIES = True          # plates with no series letters are rare; a dropped letter is common
RTO_DIGITS = {"DL": (1, 2)}       # state -> (min, max) RTO digits. Default (2, 2)
ALLOW_LEGACY_STATE_CODES = True   # OR, UA, DN: still on older vehicles

# ---- ML verifier (HistGradientBoostingClassifier, see ocr_verifier.py) ----------------
USE_ML_VERIFIER = True        # used when models/ocr_verifier_hgb.joblib exists
USE_REVIEW_RESOLVER = False   # OFF (2026-11-11 hybrid plan: it cross-checks with PARSeq, which is prohibited). second layer for REVIEW plates (review_resolver.py), used when
                              # models/review_resolver_hgb.joblib exists (train_resolver.py)
RESOLVER_TARGET_PRECISION = 0.98
ML_MIN_WEAKEST_CHAR = 0.30    # hard floor: the ML verifier never accepts a character below this
ML_TARGET_PRECISION = 0.96    # chosen on held-out data; labels are ~3% wrong, so 0.97 was unreachable
ML_VERIFIER_CAN_DEMOTE = False  # the verifier was trained on PARSeq readings, not Awiros: it may
                                # promote a borderline read to SUCCESS, but not overrule a reading
                                # that already passed every rule (format, avg >= 90 %, each char >= 80 %)

# ---- OCR engine: Awiros ANPR OCR (PP-OCRv5 SVTR_HGNet, PaddlePaddle) -------------------
# The only model paths in the project live here. AWIROS_DIR holds model.safetensors,
# en_dict.txt and (optionally) the PaddleOCR clone that provides the `ppocr` package.
# Override with the environment variables AWIROS_DIR / PADDLEOCR_DIR.
import os as _os
from pathlib import Path as _Path

OCR_ENGINE_NAME = "Awiros ANPR OCR"


def _first_existing(paths):
    for p in paths:
        if p and _Path(p).expanduser().exists():
            return _Path(p).expanduser()
    return _Path(paths[-1]).expanduser()


_ROOT = _Path(__file__).resolve().parents[1]
AWIROS_DIR = _first_existing([
    _os.environ.get("AWIROS_DIR"),
    _ROOT / "awiros",                                  # inside the project
    _ROOT.parent / "Awiros-ANPR-OCR",                  # next to the project (Linux layout)
    _Path.home() / "Awiros-ANPR-OCR",                  # C:\Users\Lenovo\Awiros-ANPR-OCR (Windows)
])
AWIROS_ORIGINAL_WEIGHTS = AWIROS_DIR / "model.safetensors"
# finetune_awiros.py --deploy writes a fine-tuned model here; it is used automatically when present
# (only deployed if it beat the original on held-out verified plates). AWIROS_USE_FINETUNED=0 = original.
AWIROS_FINETUNED_WEIGHTS = _ROOT / "models" / "awiros_finetuned" / "model.safetensors"
AWIROS_WEIGHTS = _Path(_os.environ["AWIROS_WEIGHTS"]).expanduser() if _os.environ.get("AWIROS_WEIGHTS") else (
    AWIROS_FINETUNED_WEIGHTS
    if AWIROS_FINETUNED_WEIGHTS.is_file() and _os.environ.get("AWIROS_USE_FINETUNED", "1") != "0"
    else AWIROS_ORIGINAL_WEIGHTS)
AWIROS_DICT = AWIROS_DIR / "en_dict.txt"
PADDLEOCR_DIR = _first_existing([_os.environ.get("PADDLEOCR_DIR"), AWIROS_DIR / "PaddleOCR"])
AWIROS_INPUT_SHAPE = (3, 48, 320)   # C, H, W from the model card / PP-OCRv5_server_rec.yml
AWIROS_BATCH_SIZE = 32              # crops per forward pass; lower it if the GPU runs out of memory
# On CPU the model is exported once to a static Paddle inference graph and run with oneDNN:
# ~150 ms per crop instead of ~7 s in eager mode (identical output, max diff 2e-5).
# Set AWIROS_FAST_CPU=0 to force the eager model. The export is rebuilt when the weights change.
AWIROS_FAST_CPU = _os.environ.get("AWIROS_FAST_CPU", "1") != "0"
AWIROS_INFERENCE_DIR = _ROOT / "models" / "awiros_inference"
AWIROS_CPU_THREADS = max(1, min(8, (_os.cpu_count() or 4)))
# 58 % of the checkpoint's values are denormal floats (~1e-39), which make eager CPU math ~100x
# slower. They are zeroed in memory when the model is loaded (identical output; file unchanged).
AWIROS_FLUSH_DENORMALS = _os.environ.get("AWIROS_FLUSH_DENORMALS", "1") != "0"

# ---- detailed OCR status (shown next to SUCCESS / REVIEW / REJECT / NO PLATE) -----------
# SUCCESS, LOW_CONFIDENCE, INVALID_FORMAT, MANUAL_REVIEW_REQUIRED, DETECTION_FAILED, OCR_FAILED
LOW_CONFIDENCE_REASONS = {"LOW_AVERAGE_CONFIDENCE", "UNCERTAIN_CHARACTER", "ML_VERIFIER_LOW_PROBABILITY",
                          "LOW_CALIBRATED_PROBABILITY", "UNCERTAIN_CHARACTER_CALIBRATED"}

# ---- hybrid automatic-verification gate (hybrid_verifier.py, 2026-11-11) ------------------
# Final gate on SUCCESS: calibrated P(correct) + selective same-image TTA + expected states.
# It can only send a SUCCESS to manual review; it never approves or changes text.
# HYBRID_VERIFIER=0 turns it off (the previous behaviour, exactly).
HYBRID_VERIFIER_ENABLED = _os.environ.get("HYBRID_VERIFIER", "1") != "0"
HYBRID_CALIBRATION_PATH = _ROOT / "models" / "calibration" / "awiros_calibration.json"
# Defaults only; the calibration file (fitted on the calibration split, thresholds chosen on
# calibration + validation, never on test) overrides them for the weights it was fitted on.
HYBRID_AUTO_MIN_P = 0.90          # calibrated P(exactly correct) needed for SUCCESS
# Operating point. Default = the calibration file's threshold, chosen as the highest coverage with
# ZERO false approvals on calib+val (fine-tuned model: 0.90 -> test precision 97.6 %, coverage 34 %).
# HYBRID_AUTO_MIN_P=0.85 = "balanced" (2 false approvals in 146 selection images; see
# bench/reports/HYBRID_REPORT.md for its measured test numbers).
# 2026-11-11: user chose "balanced" as the default (test: SUCCESS 70/124, 3 wrong approvals).
# HYBRID_AUTO_MIN_P=0.90 = strict (42/124, 1 wrong). HYBRID_VERIFIER=0 = gate off (90/124, 7 wrong).
HYBRID_AUTO_MIN_P_OVERRIDE = float(_os.environ.get("HYBRID_AUTO_MIN_P") or 0.85)
# 2026-11-11: user asked to stop LOW_CALIBRATED_PROBABILITY / UNCERTAIN_CHARACTER_CALIBRATED from
# sending plates to review. The gate still computes and logs the calibrated probability and its
# reasons (column hybrid_reasons), but only demotes when HYBRID_GATE_DEMOTES=1.
# Measured on test: no demotion -> 90/124 SUCCESS, 7 wrong; demotion at 0.85 -> 70/124, 3 wrong.
# Auto-crop conflict check (pipeline._pick_crop_reading): YOLO-box reading vs whole-image reading.
#   "success_only": only another reading that the rules also passed as SUCCESS can block (a truncated
#                   box crop that the rules already sent to review no longer vetoes a good reading)
#   "any":          previous behaviour - any valid-format reading with conf >= CONFLICT_MIN_CONF blocks
#   "off":          never block; the best reading is kept
CROP_CONFLICT_MODE = _os.environ.get("CROP_CONFLICT_MODE", "off")   # user choice 2026-11-11; "any" = previous
# Conflict between OCR attempts on the SAME crop (plate_decision.decide: "Different crops gave
# different confident readings"). Kept ON: measured 2026-11-11, turning it off approved 7 more
# val+calib plates of which 6 were wrong (test: +6 approved, +5 wrong). ATTEMPT_CONFLICT_CHECK=0 = off.
ATTEMPT_CONFLICT_CHECK = _os.environ.get("ATTEMPT_CONFLICT_CHECK", "1") == "1"
HYBRID_GATE_DEMOTES = _os.environ.get("HYBRID_GATE_DEMOTES", "0") == "1"
HYBRID_MIN_CHAR_P = 0.50          # weakest temperature-scaled character probability
HYBRID_TTA_GATE = 0.98            # below this calibrated P, re-read the crop with TTA variants
HYBRID_TTA_VARIANTS = ("clahe", "clahe_bilateral", "gaussian")
# Site setting: registrations from other states are read but never auto-approved (manual review).
# The labelled IOCL data is 98.4 % PY/TN; confident misreads mostly produce other state codes.
# Empty tuple = no regional check (use this for deployments outside Puducherry / Tamil Nadu).
AUTO_APPROVE_STATE_CODES = tuple(x for x in _os.environ.get("AUTO_APPROVE_STATE_CODES", "PY,TN").split(",") if x)

# ---- second OCR engine: PaddleOCR DET -> CLS -> REC (mobile models) -----------------------
# Downloaded once by setup_paddleocr_models.py into PADDLEOCR_MODEL_DIR; run with the same
# PaddleOCR clone (PADDLEOCR_DIR) and PaddlePaddle as Awiros. PADDLE_OCR_ENABLED=0 turns it off.
PADDLEOCR_MODEL_DIR = _ROOT / "models" / "paddleocr"
PADDLE_DET_MODEL = PADDLEOCR_MODEL_DIR / "PP-OCRv5_mobile_det"
PADDLE_CLS_MODEL = PADDLEOCR_MODEL_DIR / "PP-LCNet_x1_0_textline_ori"
PADDLE_REC_MODEL = PADDLEOCR_MODEL_DIR / "en_PP-OCRv5_mobile_rec"
PADDLE_OCR_ENABLED = _os.environ.get("PADDLE_OCR_ENABLED", "0") == "1"   # OFF: the 2026-11-11 hybrid prompt prohibits PaddleOCR as an OCR engine
# oneDNN per model on CPU (checked with paddlepaddle 3.3.1, Phase 2): the DET export fails under
# oneDNN ("ConvertPirAttribute2RuntimeAttribute not support ArrayAttribute<DoubleAttribute>"),
# so DET runs on plain CPU kernels (~18 ms on a plate crop); CLS / REC are 4-5x faster with it.
PADDLE_ONEDNN = {"det": False, "cls": True, "rec": True}
PADDLE_CPU_THREADS = max(1, min(4, (_os.cpu_count() or 4) // 2))   # Awiros keeps its own threads
# DET on plate crops: long side resized to this (inference.yml default 960; a plate crop is small,
# so it is enlarged), DB box score / unclip from the model card unless calibrated otherwise.
PADDLE_DET_RESIZE_LONG = 320      # Phase 3 grid on 120 real crops: 320 + pad 0.3 -> 115/120 good, 960 + no pad -> see report
PADDLE_DET_BOX_THRESH = 0.5
PADDLE_DET_UNCLIP = 2.0        # wider boxes: the end characters are not cut off
PADDLE_DET_PAD = 0.3            # border added around the plate crop before DET (fraction of height)
