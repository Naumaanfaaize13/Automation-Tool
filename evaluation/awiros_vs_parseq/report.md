# Awiros ANPR OCR vs PARSeq (OCR only, identical crops)

| slice | engine | labelled | exact_pct | char_acc_pct | cer_pct | valid_plate_output_pct | ms_per_crop_mean | ms_per_crop_p95 |
|---|---|---|---|---|---|---|---|---|
| all_clean=all | awiros | 454 | 71.59 | 88.89 | 11.08 |  | 5859.0 | 7323.71 |
| all_clean=all | parseq | 454 | 26.21 | 71.6 | 30.01 |  | 226.57 | 320.81 |
| category=blur | awiros | 60 | 85.0 | 95.17 | 4.84 |  | 7096.95 | 7171.74 |
| category=blur | parseq | 60 | 20.0 | 74.17 | 28.38 |  | 222.49 | 303.19 |
| category=clean | awiros | 454 | 71.59 | 88.89 | 11.08 |  | 5859.0 | 7323.71 |
| category=clean | parseq | 454 | 26.21 | 71.6 | 30.01 |  | 226.57 | 320.81 |
| category=lowres | awiros | 60 | 85.0 | 95.17 | 4.84 |  | 7109.1 | 7214.05 |
| category=lowres | parseq | 60 | 20.0 | 73.67 | 29.22 |  | 199.93 | 284.14 |
| category=night | awiros | 60 | 76.67 | 92.17 | 7.85 |  | 7121.41 | 7249.46 |
| category=night | parseq | 60 | 30.0 | 72.67 | 29.05 |  | 215.85 | 323.79 |
| category=noise | awiros | 60 | 80.0 | 94.0 | 6.01 |  | 7127.56 | 7217.11 |
| category=noise | parseq | 60 | 21.67 | 73.5 | 28.38 |  | 188.39 | 277.0 |
| category=occlude | awiros | 60 | 0.0 | 59.93 | 40.07 |  | 7181.22 | 7328.72 |
| category=occlude | parseq | 60 | 0.0 | 29.56 | 71.29 |  | 206.04 | 306.23 |
| category=tilt | awiros | 60 | 85.0 | 94.5 | 5.51 |  | 7147.19 | 7372.44 |
| category=tilt | parseq | 60 | 31.67 | 74.67 | 27.88 |  | 221.38 | 333.54 |
| category=unreadable | awiros | 1 | 0.0 | 90.0 | 10.0 | 42.86 | 5240.9 | 7110.93 |
| category=unreadable | parseq | 1 | 0.0 | 60.0 | 40.0 | 0.0 | 269.02 | 308.89 |
| day_night=day | awiros | 27 | 62.96 | 87.24 | 12.6 |  | 7203.07 | 7330.05 |
| day_night=day | parseq | 27 | 22.22 | 78.19 | 22.14 |  | 188.0 | 261.98 |
| day_night=night | awiros | 427 | 72.13 | 89.0 | 10.98 |  | 5774.01 | 7321.74 |
| day_night=night | parseq | 427 | 26.46 | 71.18 | 30.5 |  | 229.01 | 321.12 |
| label_source=NOT_parseq (unbiased subset) | awiros | 249 | 65.06 | 85.44 | 14.41 |  | 5865.01 | 7336.21 |
| label_source=NOT_parseq (unbiased subset) | parseq | 249 | 15.66 | 64.8 | 36.97 |  | 232.33 | 321.12 |
| label_source=finalize | awiros | 84 | 34.52 | 70.8 | 28.88 |  | 5260.46 | 7322.73 |
| label_source=finalize | parseq | 84 | 13.1 | 52.26 | 50.85 |  | 245.27 | 320.81 |
| label_source=hiconf | awiros | 153 | 86.93 | 97.17 | 2.83 |  | 6244.95 | 7351.09 |
| label_source=hiconf | parseq | 153 | 18.3 | 74.54 | 26.45 |  | 223.29 | 333.85 |
| label_source=llm | awiros | 12 | 0.0 | 38.33 | 61.67 |  | 5252.53 | 7054.27 |
| label_source=llm | parseq | 12 | 0.0 | 28.33 | 75.0 |  | 257.08 | 301.4 |
| label_source=parseq | awiros | 205 | 79.51 | 93.09 | 6.98 |  | 5851.71 | 7310.98 |
| label_source=parseq | parseq | 205 | 39.02 | 79.86 | 21.45 |  | 219.57 | 307.08 |
| layout=dual_row | awiros | 113 | 42.48 | 72.92 | 26.98 |  | 5223.12 | 7140.64 |
| layout=dual_row | parseq | 113 | 1.77 | 32.34 | 74.01 |  | 295.86 | 362.24 |
| layout=single_row | awiros | 341 | 81.23 | 94.19 | 5.81 |  | 6069.72 | 7336.21 |
| layout=single_row | parseq | 341 | 34.31 | 84.61 | 15.43 |  | 203.61 | 244.82 |

| engine | device | load s | throughput crops/s (batched) | GPU peak MB |
|---|---|---|---|---|
| awiros | cpu | 11.28 | 0.1 | n/a (CPU) |
| parseq | cpu | 13.44 | 6.6 | n/a (CPU) |
