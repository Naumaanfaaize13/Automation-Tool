"""Dataset audit + fixed, leakage-free split of the audited traxflow plate images.

    python bench/split_dataset.py [--manifest ~/Downloads/traxflow_local_images_2026-11-10/manifest.csv]

Ground truth = the human-audited `plate_number` (audited=yes, verdict=correct). Rows with a blank
plate_number are reported as missing labels and left out. Images are grouped so that the same
vehicle (same plate number) and near-duplicate photos (dHash Hamming distance <= 6) always land in
the same split. Groups are assigned per stratum (vehicle_type x day_night) with a fixed seed:
    train 45 %  val 15 %  calib 15 %  test 25 %
The test split is never used for tuning or training. Writes training/splits/*.csv and
bench/reports/dataset_audit.json."""
import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/
import indian_plate_validator as ipv  # noqa: E402
import plate_preprocess as pp  # noqa: E402

FRACS = (("train", 0.45), ("val", 0.15), ("calib", 0.15), ("test", 0.25))
SEED = 20261111


def dhash(img, n=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="~/Downloads/traxflow_local_images_2026-11-10/manifest.csv")
    ap.add_argument("--out", default=str(ROOT / "training" / "splits"))
    a = ap.parse_args()
    man = Path(a.manifest).expanduser()
    rows = list(csv.DictReader(open(man, newline="", encoding="utf-8")))
    audit = {"manifest": str(man), "rows": len(rows)}
    items, missing, unreadable = [], [], []
    for r in rows:
        p = man.parent / r["path"]
        label = ipv.normalize(r["plate_number"])
        if not label:
            missing.append(r["path"]); continue
        img = cv2.imread(str(p))
        if img is None:
            unreadable.append(r["path"]); continue
        q = pp.assess(img)
        v = ipv.validate(label)
        items.append({"path": str(p), "label": label, "vehicle_type": r["vehicle_type"],
                      "day_night": r["day_night"], "camera_id": r["camera_id"],
                      "plate_source": r["plate_source"], "txn": r["fcc_transaction_id"],
                      "h": img.shape[0], "w": img.shape[1], "dhash": dhash(img),
                      "quality": pp.quality_label(q), "flags": q["flags"],
                      "format": v.get("format") or ("VALID" if v.get("is_valid_format") else "INVALID"),
                      "valid": bool(v.get("is_valid_format")), "two_line": img.shape[1] / img.shape[0] < 2.2})
    n = len(items)
    uf = UF(n)
    by_label = collections.defaultdict(list)
    for i, it in enumerate(items):
        by_label[it["label"]].append(i)
    dup_labels = {k: len(v) for k, v in by_label.items() if len(v) > 1}
    for idx in by_label.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    near = []
    for i in range(n):
        for j in range(i + 1, n):
            d = bin(items[i]["dhash"] ^ items[j]["dhash"]).count("1")
            if d <= 6:
                near.append((Path(items[i]["path"]).name, Path(items[j]["path"]).name, d,
                             items[i]["label"] == items[j]["label"]))
                uf.union(i, j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    # stratified group assignment
    rng = random.Random(SEED)
    strata = collections.defaultdict(list)
    for g, idx in groups.items():
        it = items[idx[0]]
        strata[(it["vehicle_type"], it["day_night"])].append(g)
    assign = {}
    for key in sorted(strata):
        gs = sorted(strata[key]); rng.shuffle(gs)
        tot = sum(len(groups[g]) for g in gs)
        filled = {s: 0 for s, _ in FRACS}
        for g in gs:
            # put the group where the split is furthest below its target share
            s = max(FRACS, key=lambda sf: sf[1] * tot - filled[sf[0]])[0]
            assign[g] = s
            filled[s] += len(groups[g])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for s, _ in FRACS:
        sel = [items[i] for g, idx in groups.items() if assign[g] == s for i in idx]
        sel.sort(key=lambda it: it["path"])
        counts[s] = len(sel)
        with open(out / f"{s}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "group", "vehicle_type", "day_night", "camera_id",
                        "plate_source", "quality"])
            for it in sel:
                i = items.index(it)
                w.writerow([it["path"], it["label"], uf.find(i), it["vehicle_type"],
                            it["day_night"], it["camera_id"], it["plate_source"], it["quality"]])
    C = collections.Counter
    audit.update(
        labelled=n, missing_labels=missing, unreadable=unreadable,
        duplicate_plate_numbers=dup_labels, near_duplicate_pairs=near,
        groups=len(groups), splits=counts, seed=SEED,
        incorrect_labels="not detectable automatically: every row is audited=yes, verdict=correct",
        invalid_format_labels=[it["label"] for it in items if not it["valid"]],
        height_px={"min": min(it["h"] for it in items), "median": int(np.median([it["h"] for it in items])),
                   "max": max(it["h"] for it in items)},
        width_px={"min": min(it["w"] for it in items), "median": int(np.median([it["w"] for it in items])),
                  "max": max(it["w"] for it in items)},
        two_line_shape=sum(it["two_line"] for it in items),
        vehicle_type=C(it["vehicle_type"] for it in items), day_night=C(it["day_night"] for it in items),
        camera=C(it["camera_id"] for it in items), label_source=C(it["plate_source"] for it in items),
        quality=C(it["quality"].split(":")[0] for it in items),
        quality_flags=C(f for it in items for f in it["flags"]),
        format=C(it["format"] for it in items), state=C(it["label"][:2] for it in items),
        split_by_vehicle_type={s: C(items[i]["vehicle_type"] for g, idx in groups.items()
                                    if assign[g] == s for i in idx) for s, _ in FRACS})
    rep = ROOT / "bench" / "reports"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=list))
    print(json.dumps({k: audit[k] for k in ("rows", "labelled", "missing_labels", "groups", "splits",
                                           "duplicate_plate_numbers", "invalid_format_labels",
                                           "height_px", "width_px", "two_line_shape", "quality",
                                           "quality_flags", "format", "vehicle_type", "day_night",
                                           "label_source")}, indent=1, default=list))
    print("near-duplicate pairs:", len(near))
    for x in near[:15]:
        print("  ", x)


if __name__ == "__main__":
    main()
