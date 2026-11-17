"""Fine-tune the Awiros ANPR OCR model on your own, hand-verified plates.

Training data (only labels a person has confirmed):
  * training/verified_labels.csv  - written by label_tool.py (status "ok" rows)
  * results/log.csv               - dashboard rows with a manual correction / confirmation
                                    (the saved plate crop + the reviewer's text)
  * --labels FILE.csv             - any other CSV with columns path,label

A held-out validation set (split by vehicle/transaction, so near-duplicate photos of one
vehicle never sit on both sides) is read with the original model first. The fine-tuned
model is saved after every epoch that improves validation exact-match, and it is only
deployed (models/awiros_finetuned/model.safetensors, picked up by awiros_ocr.py) when it
beats the original model on that held-out set: `--deploy`.

What is trained: the CTC path (backbone + SVTR neck + CTC head), i.e. exactly what inference
uses. BatchNorm statistics are frozen (small dataset); the first backbone stages can be frozen
with --freeze-stages. The NRTR guidance head is not used at inference and is left unchanged.

GPU strongly recommended (Windows RTX 3050: --device cuda). On CPU Paddle's eager mode needs
several seconds per image, so CPU is only usable for a smoke test (--max-steps 2).

    python src/finetune_awiros.py --device cuda --epochs 30 --deploy
    python src/finetune_awiros.py --max-steps 2          # smoke test (CPU)
"""
import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import anpr_config as cfg
import indian_plate_validator as ipv

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- data
def load_samples(args):
    rows = {}
    vl = Path(args.verified)
    if vl.is_file():
        for r in csv.DictReader(open(vl, newline="", encoding="utf-8")):
            rows[r["path"]] = r                       # last decision per image wins
    samples = {p: ipv.normalize(r["label"]) for p, r in rows.items() if r["status"] == "ok"}
    n_tool = len(samples)
    n_dash = 0
    if args.dashboard and dl_log().is_file():
        res = dl_log().parent
        for r in csv.DictReader(open(dl_log(), newline="", encoding="utf-8")):
            t = ipv.normalize(r.get("manually_corrected_text", ""))
            if not t:
                continue
            # prefer the un-enhanced crop the reading came from; older rows only have the box crop
            for sub, col in (("train_crops", "train_crop_file"), ("crops", "crop_file")):
                f = res / sub / (r.get(col) or "")
                if r.get(col) and f.is_file():
                    samples[str(f)] = t
                    n_dash += 1
                    break
    n_extra = 0
    for f in args.labels or []:
        f = Path(f).expanduser()
        for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
            p = Path(r["path"]).expanduser()
            p = p if p.is_absolute() else f.parent / p
            if r.get("label"):
                samples[str(p)] = ipv.normalize(r["label"])
                n_extra += 1
    out = [(p, t) for p, t in samples.items() if t and Path(p).is_file()]
    print(f"samples: {len(out)} (label tool {n_tool}, dashboard {n_dash}, extra {n_extra})")
    return out


def _abs(csv_file, path):
    """A path from a labels CSV: absolute, or relative to the CSV's folder (portable to Windows)."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else Path(csv_file).expanduser().parent / p)


def dl_log():
    import dashboard_log as dl
    return dl.LOG


def group_of(path):
    """Images of one transaction / vehicle share the prefix before the first '__'."""
    name = Path(path).name
    return name.split("__")[0] if "__" in name else name


def split(samples, val_frac, seed):
    groups = sorted({group_of(p) for p, _ in samples})
    random.Random(seed).shuffle(groups)
    val_g = set(groups[: max(1, int(round(len(groups) * val_frac)))])
    tr = [s for s in samples if group_of(s[0]) not in val_g]
    va = [s for s in samples if group_of(s[0]) in val_g]
    return tr, va


def augment(img, rng):
    """Mild, plate-realistic augmentation (camera blur, night noise, tilt, framing)."""
    h, w = img.shape[:2]
    if rng.random() < 0.7:                                    # framing: crop / pad the edges
        m = lambda: int(rng.uniform(-0.06, 0.08) * w)
        l, r_, t, b = m(), m(), int(rng.uniform(-0.06, 0.1) * h), int(rng.uniform(-0.06, 0.1) * h)
        img = cv2.copyMakeBorder(img, max(t, 0), max(b, 0), max(l, 0), max(r_, 0),
                                 cv2.BORDER_REPLICATE)
        H, W = img.shape[:2]
        img = img[max(-t, 0):H - max(-b, 0), max(-l, 0):W - max(-r_, 0)]
        h, w = img.shape[:2]
    if rng.random() < 0.5:                                    # small rotation / perspective
        a = rng.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:
        d = 0.06
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32(rng.uniform(-d, d, (4, 2)) * [w, h])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:                                    # brightness / contrast
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.6, 1.4), beta=rng.uniform(-40, 40))
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.2:                                    # motion blur
        k = rng.choice([3, 5, 7])
        ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, ker)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, rng.uniform(3, 15), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                                    # low resolution
        f = rng.uniform(0.35, 0.8)
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))))
        img = cv2.resize(small, (w, h))
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(25, 80))])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.1:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return img


def sample_weights(train, balance):
    """Oversample rare state codes and dual-row plates (the model's weak spots)."""
    if not balance:
        return None
    state = lambda t: re.match(r"[A-Z]{2}", t).group(0) if re.match(r"[A-Z]{2}", t) else "?"
    cnt = Counter(state(t) for _, t in train)
    w = []
    for p, t in train:
        im = cv2.imread(p)
        dual = im is not None and im.shape[1] / max(im.shape[0], 1) < cfg.TWO_LINE_ASPECT
        w.append((1.0 / math.sqrt(cnt[state(t)])) * (2.0 if dual else 1.0))
    w = np.array(w); return w / w.sum()


# ---------------------------------------------------------------------------- model
def ctc_logits(model, x):
    """Backbone -> SVTR neck -> CTC head (training mode: raw logits, N x T x C)."""
    f = model.backbone(x)
    h = model.head
    if getattr(h, "use_pool", False):
        f = h.pool(f.reshape([0, 3, -1, h.in_channels]).transpose([0, 3, 1, 2]))
    return h.ctc_head(h.ctc_encoder(f))


def evaluate(reader, val, batch=32):
    """Exact-match / character accuracy of the reader on (path, label) pairs."""
    from awiros_ocr import preprocess
    ok, chars, total, wrong = 0, 0, 0, []
    for s in range(0, len(val), batch):
        chunk = val[s:s + batch]
        x = np.stack([preprocess(cv2.imread(p)) for p, _ in chunk])
        probs = reader._forward(x)
        for (p, gt), pr in zip(chunk, probs):
            pred = ipv.normalize(reader._decode(pr)["raw"])
            ok += pred == gt
            d = lev(pred, gt)
            chars += max(len(gt) - d, 0); total += len(gt)
            if pred != gt:
                wrong.append((Path(p).name, gt, pred))
    return {"exact": ok / max(len(val), 1), "char_acc": chars / max(total, 1), "n": len(val),
            "wrong": wrong}


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def save_weights(model, path):
    from safetensors.numpy import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.numpy() for k, v in model.state_dict().items()}, str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verified", default=str(ROOT / "training" / "verified_labels.csv"))
    ap.add_argument("--labels", action="append", help="extra CSV(s) with path,label")
    ap.add_argument("--no-dashboard", dest="dashboard", action="store_false",
                    help="do not use the dashboard's manual corrections")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--freeze-stages", type=int, default=1,
                    help="freeze the stem and the first N backbone stages (0-4)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-labels", help="fixed validation CSV (path,label), e.g. training/splits/val.csv; "
                    "its images are removed from training. Default: random split by vehicle")
    ap.add_argument("--exclude-labels", action="append", default=[],
                    help="CSV(s) whose images must never be trained on (e.g. training/splits/test.csv)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: stop after N epochs without a better validation exact-match")
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--min-samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "models" / "awiros_finetuned"))
    ap.add_argument("--deploy", action="store_true",
                    help="install the best model only if it beats the original on validation")
    args = ap.parse_args()

    samples = load_samples(args)
    if len(samples) < args.min_samples and not args.max_steps:
        raise SystemExit(f"Only {len(samples)} verified samples; need at least {args.min_samples} "
                         f"(run label_tool.py, or lower --min-samples).")
    excl = set()
    for f in args.exclude_labels:
        excl |= {_abs(f, r["path"]) for r in csv.DictReader(open(Path(f).expanduser(), encoding="utf-8"))}
    samples = [s for s in samples if s[0] not in excl]
    if args.val_labels:
        vf = Path(args.val_labels).expanduser()
        val = [(_abs(vf, r["path"]), ipv.normalize(r["label"]))
               for r in csv.DictReader(open(vf, encoding="utf-8")) if r.get("label")]
        vset = {p for p, _ in val}
        train = [s for s in samples if s[0] not in vset]
        val = [s for s in val if Path(s[0]).is_file() and s[0] not in excl]
    else:
        train, val = split(samples, args.val_frac, args.seed)
    if excl:
        print(f"excluded from training/validation: {len(excl)} images")
    if args.max_steps:
        val = val[:8]                               # smoke test: keep the eager CPU eval short
    print(f"train {len(train)}  val {len(val)} (split by vehicle)")

    cfg.AWIROS_FAST_CPU = False                     # train the eager model
    import paddle
    from awiros_ocr import AwirosReader, preprocess
    reader = AwirosReader(args.device, weights=cfg.AWIROS_ORIGINAL_WEIGHTS)
    model = reader.model
    out = Path(args.out)
    report = {"settings": vars(args), "train": len(train), "val": len(val), "epochs": []}

    base = evaluate(reader, val) if val else None
    if base:
        print(f"original model on validation: exact {base['exact']:.3f}  char {base['char_acc']:.3f}")
        report["original"] = {k: v for k, v in base.items() if k != "wrong"}

    # what to train
    stages = list(model.backbone.stages)
    frozen = [model.backbone.stem] + stages[: args.freeze_stages] if args.freeze_stages >= 0 else []
    frozen_ids = {id(p) for layer in frozen for p in layer.parameters()}
    nrtr_ids = {id(p) for n, p in model.named_parameters() if "gtc_head" in n or "before_gtc" in n}
    params = [p for p in model.parameters() if id(p) not in frozen_ids | nrtr_ids]
    keep = {id(q) for q in params}
    for p in model.parameters():
        p.stop_gradient = id(p) not in keep
    print(f"trainable parameters: {sum(int(np.prod(p.shape)) for p in params) / 1e6:.1f}M")

    steps_per_epoch = max(1, math.ceil(len(train) / args.batch))
    total = steps_per_epoch * args.epochs
    sched = paddle.optimizer.lr.LinearWarmup(
        paddle.optimizer.lr.CosineAnnealingDecay(args.lr, T_max=max(1, total)),
        warmup_steps=min(steps_per_epoch, total // 10 + 1), start_lr=args.lr / 10, end_lr=args.lr)
    opt = paddle.optimizer.AdamW(sched, parameters=params, weight_decay=1e-4,
                                 grad_clip=paddle.nn.ClipGradByGlobalNorm(5.0))
    ctc = paddle.nn.CTCLoss(blank=0, reduction="none")
    idx = {c: i for i, c in enumerate(reader.ctc_chars)}
    rng = np.random.default_rng(args.seed)
    w = sample_weights(train, args.balance)
    best = base["exact"] if base else -1.0
    best_epoch, step = 0, 0
    T = None

    for ep in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()                       # frozen BatchNorm statistics
        order = rng.choice(len(train), len(train), replace=w is not None, p=w)
        t0, losses = time.time(), []
        for s in range(0, len(order), args.batch):
            chunk = [train[i] for i in order[s:s + args.batch]]
            imgs, labels, lens = [], [], []
            for p, t in chunk:
                im = cv2.imread(p)
                if im is None:
                    continue
                imgs.append(preprocess(augment(im, rng)))
                labels.append([idx[c] for c in t]); lens.append(len(t))
            if not imgs:
                continue
            L = max(lens)
            lab = np.array([l + [0] * (L - len(l)) for l in labels], np.int32)
            logits = ctc_logits(model, paddle.to_tensor(np.stack(imgs)))
            T = logits.shape[1]
            loss = ctc(logits.transpose([1, 0, 2]), paddle.to_tensor(lab),
                       paddle.to_tensor([T] * len(imgs), dtype="int64"),
                       paddle.to_tensor(lens, dtype="int64")).mean()
            loss.backward(); opt.step(); opt.clear_grad(); sched.step()
            losses.append(float(loss)); step += 1
            if args.max_steps and step >= args.max_steps:
                break
        model.eval()
        ev = evaluate(reader, val) if val else {"exact": 0, "char_acc": 0}
        row = {"epoch": ep, "loss": round(float(np.mean(losses)) if losses else 0, 4),
               "val_exact": round(ev["exact"], 4), "val_char": round(ev["char_acc"], 4),
               "seconds": round(time.time() - t0, 1)}
        report["epochs"].append(row)
        print(row)
        if ev["exact"] > best:
            best, best_epoch = ev["exact"], ep
            save_weights(model, out / "candidate.safetensors")
            report["best"] = dict(row, wrong=ev.get("wrong", [])[:50])
        if args.max_steps and step >= args.max_steps:
            break
        if args.patience and ep - max(best_epoch, 0) >= args.patience:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    report["best_epoch"] = best_epoch
    deployed = False
    if args.deploy and best_epoch and base and best > base["exact"]:
        (out / "candidate.safetensors").replace(out / "model.safetensors")
        deployed = True
    report["deployed"] = deployed
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    if base:
        print(f"\noriginal exact {base['exact']:.3f} -> best {best:.3f} (epoch {best_epoch}); "
              f"{'DEPLOYED to ' + str(out / 'model.safetensors') if deployed else 'not deployed'}")
    print(f"report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
