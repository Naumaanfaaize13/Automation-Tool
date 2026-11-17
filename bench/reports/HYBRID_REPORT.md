# Hybrid ANPR accuracy work — report (2026-11-11)

**Bottom line:** the 97 % end-to-end target was **not reached**. On the untouched test split
(124 human-audited images), exact-match rose from **65.3 % → 75.0 %** (76.9 % without the 3 images
whose label is visibly wrong). Automatic-approval precision rose from **91.3 % → 97.6 %**, but
coverage fell from 64.5 % to 33.9 %. All numbers come from the real production pipeline
(`bench/run_e2e.py`). None are estimates.

## 1. What is deployed

```
YOLO11 best.pt (unchanged) → auto-crop logic (unchanged) → original crop first
→ adaptive OpenCV retries only when needed (unchanged) → Awiros OCR, FINE-TUNED head
→ Indian format validation + rules (unchanged) → existing HGB ML verifier (unchanged, cannot demote)
→ NEW hybrid gate: temperature-scaled + Platt-calibrated P(correct) → SUCCESS / REVIEW
```

| Component | File | Status |
|---|---|---|
| Fine-tuned Awiros (backbone frozen, SVTR neck + CTC head trained) | `models/awiros_finetuned/model.safetensors` | deployed (auto-loaded) |
| Original Awiros checkpoint | `~/Documents/Awiros-ANPR-OCR/model.safetensors` | untouched (sha256 f9f1264e…) |
| Calibration (T = 1.5 + Platt), fitted on the calib split | `models/calibration/awiros_calibration.json` | bound to the deployed weights by sha256 |
| Hybrid gate | `hybrid_verifier.py` | on (`HYBRID_VERIFIER=0` turns it off) |
| Denormal flush (speed fix) | `awiros_ocr.py: flush_denormals` | on, output bit-identical |
| YOLO11, ML verifier, APIs, dashboard, Excel, auth | — | unchanged; new log columns appended via the existing migration |

## 2. Data and splits (`bench/split_dataset.py`, `dataset_audit.json`)

- 500 images, 498 labelled (2 missing labels), all `audited=yes`. All are 640-px-wide pump-camera
  plate crops: 315 bike / 176 car, 259 day / 239 night, 323 two-row, 490 PY/TN.
- Split by vehicle group (no near-duplicates found): **train 225 / val 75 / calib 74 / test 124**.
  Test was never used for training, model selection, calibration or threshold choice.
- **Label errors:** 6 images show a different registration than their "verified" label (e.g. image
  `PY01CM1918`, label `PY01CY7642`), checked by eye. They are listed in
  `suspected_label_errors.csv` and were **not corrected**; please re-check them. 3 are in test and
  count as errors in the headline numbers. The training split was screened too; none were found.

## 3. Results on the untouched test split (124 images, real pipeline)

| # | Configuration | Exact | CER | Auto-approved | Precision | Coverage | False accepts | Mean s/img |
|---|---|---|---|---|---|---|---|---|
| 1 | Original Awiros, plain read of the image | 62.1 % | 9.6 % | — | — | — | — | — |
| 2 | Original production pipeline (YOLO + OpenCV + original Awiros + rules + ML) | 65.3 % | 9.8 % | 80 | 91.3 % | 64.5 % | 7 | 0.39 |
| 3 | Fine-tuned Awiros, plain read | 71.0 % | 6.2 % | — | — | — | — | — |
| 4 | Production pipeline + fine-tuned Awiros (gate off) | **75.0 %** | 5.9 % | 90 | 92.2 % | 72.6 % | 7 | 0.33 |
| 5a | **Full hybrid, default (strict)** | **75.0 %** | 5.9 % | 42 | **97.6 %** | 33.9 % | **1** | 0.33 |
| 5b | Full hybrid, `HYBRID_AUTO_MIN_P=0.85` (balanced) | 75.0 % | 5.9 % | 70 | 95.7 % | 56.5 % | 3 | 0.34 |

Without the 3 confirmed label mismatches (121 images): exact 76.9 %, and 5b's precision is 98.5 %
(2 of its 3 "false accepts" are correct readings of a wrongly labelled image).

- Exact matches: 93 correct, 31 wrong. The gate never changes text, only status.
- False rejections, i.e. correct readings sent to review: 52 (5a), 26 (5b), 10 (4).
- YOLO found a box in 120 of 124 test images. Detection precision can't be measured because the
  dataset has no box annotations. The images are already plate crops, and YOLO-box crops read much
  worse than the whole image (val: 32 % vs 65 %), so auto-crop mode reading the whole image is correct.

**Operating point.** 5a is the default because the rule was declared before testing: the most
approvals with *zero* false approvals on calib + val. 5b allows 2 false approvals in those 146
images. One environment variable switches between them; the choice is yours.

## 4. What each technique contributed (measured, not assumed)

| Technique | Result |
|---|---|
| **Awiros fine-tuning** | The main gain: +9.7 points exact on test, CER 9.8 → 5.9 %, two-row plates 42 → 60 of 94 on val+calib |
| Full fine-tune vs frozen backbone | val+calib: frozen backbone 75.2 %, full 74.5 %, frozen at lower LR 73.8 %. The frozen-backbone model was chosen (best, and it keeps the released backbone) |
| Blanket OpenCV enhancement | **Hurts** (val: CLAHE 65.3 → 54.7 %, low-light 58.7 %, adaptive threshold 34.7 %; bilateral/Gaussian ±0). So the original crop is always read first and variants are only fallbacks |
| Temperature scaling + Platt | Plate-level ECE on test 0.114 → 0.070, NLL 0.41 → 0.36; character ECE 0.022 → 0.007. It ranks readings well (AUC 0.85), and the gate is built on it. Calibration does not change accuracy |
| Selective TTA (same image) | Implemented. On the fine-tuned model it never improved the calib+val trade-off, so the selection switched it off (`tta_variants: []`). Earlier, on the original model, it gained 1 of 75 val plates. That is noise |
| Region check (PY/TN only) | Implemented. It helped the original model, which turned PY into KA/MP/MH, but not the fine-tuned one (only 2 state-code errors on test), so the selection switched it off |
| Existing ML verifier | Kept and unchanged. It was trained on PARSeq readings: on Awiros test readings its AUC is 0.75, vs 0.87 for Awiros' own confidence. It promoted 0 test plates. Any promotion must still pass the calibrated gate, so it can't approve a plate alone. `ML_VERIFIER_CAN_DEMOTE = False` is unchanged |
| Indian format validation | Unchanged (existing validator, never swaps characters to force a valid format) |

## 5. Failure analysis (test, `failures_test_final.csv`)

31 wrong readings: **27 are two-row (bike) plates.**

| Category | Count |
|---|---|
| Incorrect character, two-row (H↔M, A↔B, U↔D, E↔L, 1↔7, 2↔7, 6↔0 …) | 19 |
| Missing character, two-row (e.g. `PY01CX8284` → `PY01X8284`) | 5 |
| Wrong state code, two-row | 2 |
| Label mismatch (the image shows another plate) | 3 |
| Incorrect character, one-row | 2 (1 auto-approved: `PY01CH5600` → `PY01CW5600`, P = 0.901) |

Only 1 of the 31 failures had any image-quality flag (DARK). No failure traced back to detection or
crop size. **What limits accuracy is the OCR model on two-row plates, trained on only 225 images.**

## 6. Speed fix found along the way

58 % of the released Awiros weights (21.8M of 37.3M values) are *denormal* floats (~1e-39). x86
CPUs process those about 100× slower. Zeroing them in memory gives bit-identical output (max
difference 0.0) and makes eager CPU inference **67× faster** (105 s → 1.6 s per 16 crops). That turned
a 16.8-hour training epoch into about 40 s, which is what made fine-tuning possible on this laptop. The file on disk is unchanged.

## 7. GPU

This Linux laptop has no NVIDIA GPU, so **no GPU numbers were measured**. The CUDA paths already
existed and were not changed: `python server.py --device cuda`, and for training
`python finetune_awiros.py --device cuda ...` or `python finetune_awiros_head.py --device gpu ...`.
CPU (i7-13620H): 0.33 s per image end-to-end (median 0.18 s, p95 0.80 s).

## 8. What would move toward 97 %

1. **More verified two-row plates.** 225 training images gave +10 points. Every confirmed or corrected
   plate in the dashboard is training data (`finetune_awiros_head.py --labels …`).
2. Fix the 6 mismatched labels and re-check the rest; a label-error rate of about 1–2 % caps measurable accuracy.
3. Retrain the full model on the RTX 3050 with more data. The full fine-tune was about as good as the
   head-only one here; with more data it should pull ahead.
4. Re-fit calibration and thresholds after every retrain: `bench/cache_reads.py` → `bench/analyze_hybrid.py`.
   The gate only uses a calibration whose sha256 matches the loaded weights.

## 9. Reproduce

```bash
.venv-linux/bin/python finetune_awiros_head.py --out models/awiros_finetuned_head          # ~15 min CPU
.venv-linux/bin/python bench/cache_reads.py --splits val calib test --weights models/awiros_finetuned/model.safetensors --tag head
.venv-linux/bin/python bench/analyze_hybrid.py --tag head --weights models/awiros_finetuned/model.safetensors
.venv-linux/bin/python bench/run_e2e.py --split test --tag final_strict
```

## 10. Rollback

| To undo | Do |
|---|---|
| Fine-tuned model | `AWIROS_USE_FINETUNED=0 python server.py`, or delete `models/awiros_finetuned/model.safetensors` |
| Hybrid gate | `HYBRID_VERIFIER=0 python server.py` |
| Denormal flush | `AWIROS_FLUSH_DENORMALS=0` (identical output, only slower) |
| Everything | source backup `backups/pre_hybrid_20261111/source.tgz` (+ checkpoint hashes); the original Awiros/YOLO/ML files were never modified |
