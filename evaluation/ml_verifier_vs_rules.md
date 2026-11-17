# ML verifier (HistGradientBoostingClassifier) vs fixed rules

Data: 600 real pump plate images (Desktop/plates manifests), labels unaudited (~3% visibly wrong).
Split by original image: 70% train, 30% held out (256 labelled real images; 700 with degraded copies).
Threshold chosen by 5-fold grouped cross-validation on the train part only.

Held-out AUC (ranking correct vs wrong readings): ML 0.920, average confidence 0.727, weakest character 0.711.

| Held-out, clean real images (n=256) | auto-accepted | precision | wrong accepted |
|---|---|---|---|
| Rules: avg >= 0.90, weakest >= 0.80 | 25.0% | 95.3% | 3 |
| Rules: avg >= 0.85, weakest >= 0.60 | 34.8% | 91.0% | 8 |
| **ML, target precision 0.96 (deployed setting)** | **25.0%** | **98.4%** | **1** |
| ML, target precision 0.95 | 34.8% | 95.5% | 4 |

| Held-out, clean + degraded (n=700) | auto-accepted | precision | wrong accepted |
|---|---|---|---|
| Rules: avg >= 0.90, weakest >= 0.80 | 25.7% | 95.0% | 9 |
| **ML, target 0.96** | **28.1%** | **98.5%** | **3** |
| ML, target 0.95 | 37.6% | 97.0% | 8 |

Small numbers: 1 vs 3 errors on 256 images is a direction, not a precise rate.
Deployed model: refit on all eligible rows, threshold 0.923 from grouped CV on all data (models/ocr_verifier_hgb.json).
