"""CPU-feasible Awiros fine-tuning: frozen backbone, trained recognition head.

Full fine-tuning (finetune_awiros.py) needs a GPU: on this laptop's CPU one epoch over ~225
images takes many hours. This script fine-tunes the part of Awiros that maps visual features to
characters (SVTR neck + CTC head, 5.3M of 37M parameters) and keeps the PPHGNetV2 backbone
exactly as released:

  1. the original backbone is exported once to a static oneDNN graph (fast on CPU, identical math);
  2. backbone features are computed once per image: 1 clean + --aug augmented copies (the same
     plate-realistic augmentation as finetune_awiros.py) for training, clean only for validation;
  3. the neck + CTC head are trained on those features with CTC loss, AdamW, cosine LR,
     early stopping on validation exact-match (validation = training/splits/val.csv by default);
  4. the best weights (original backbone + new head) are written to
     models/awiros_finetuned/candidate_head.safetensors, and with --deploy copied to
     models/awiros_finetuned/model.safetensors ONLY if they beat the original on validation.

The original model.safetensors is never modified. The untouched test split is not read here.

    python scripts/finetune_awiros_head.py --train training/splits/train.csv --val training/splits/val.csv --deploy
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path
import sys  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/

import cv2
import numpy as np

import anpr_config as cfg
import indian_plate_validator as ipv



def read_csv(p):
    p = Path(p).expanduser()
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        q = Path(r["path"]).expanduser()
        q = q if q.is_absolute() else p.parent / q
        lab = ipv.normalize(r.get("label", ""))
        if lab and q.is_file():
            out.append((str(q), lab))
    return out


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def backbone_predictor(reader, out_dir):
    """Static oneDNN graph of the ORIGINAL backbone (exported once)."""
    import paddle
    from paddle import inference as pi
    out_dir = Path(out_dir)
    mf, pf = out_dir / "inference.json", out_dir / "inference.pdiparams"
    if not (mf.is_file() and pf.is_file()):
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = [paddle.static.InputSpec([None] + list(cfg.AWIROS_INPUT_SHAPE), "float32", name="x")]
        paddle.jit.save(paddle.jit.to_static(reader.model.backbone, input_spec=spec, full_graph=True),
                        str(out_dir / "inference"))
    c = pi.Config(str(mf), str(pf))
    if reader.device == "gpu":
        c.enable_use_gpu(256, 0)
    else:
        c.disable_gpu(); c.enable_mkldnn(); c.set_cpu_math_library_num_threads(cfg.AWIROS_CPU_THREADS)
    c.disable_glog_info()
    p = pi.create_predictor(c)
    return p


def features(pred, batch):
    h = pred.get_input_handle(pred.get_input_names()[0])
    h.reshape(list(batch.shape)); h.copy_from_cpu(np.ascontiguousarray(batch, np.float32))
    pred.run()
    return pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(ROOT / "training" / "splits" / "train.csv"))
    ap.add_argument("--val", default=str(ROOT / "training" / "splits" / "val.csv"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--aug", type=int, default=8, help="augmented copies per training image")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-encoder", action="store_true", default=True,
                    help="train the SVTR neck as well as the CTC classifier (default)")
    ap.add_argument("--head-only", dest="train_encoder", action="store_false",
                    help="train only the final CTC classifier layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "models" / "awiros_finetuned"))
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()

    import paddle
    cfg.AWIROS_FAST_CPU = False
    from awiros_ocr import AwirosReader, preprocess
    from finetune_awiros import augment, save_weights
    from safetensors.numpy import load_file

    train, val = read_csv(a.train), read_csv(a.val)
    vset = {p for p, _ in val}
    train = [s for s in train if s[0] not in vset]
    print(f"train {len(train)} images x {a.aug + 1} views, val {len(val)}", flush=True)
    reader = AwirosReader(a.device, weights=cfg.AWIROS_ORIGINAL_WEIGHTS)
    model = reader.model
    model.eval()
    bb = backbone_predictor(reader, ROOT / "models" / "awiros_backbone_inference")
    rng = np.random.default_rng(a.seed)

    t0 = time.time()

    def feats(items, n_aug):
        X, Y = [], []
        for k in range(0, len(items), 16):
            chunk = items[k:k + 16]
            imgs = [cv2.imread(p) for p, _ in chunk]
            for rep in range(n_aug + 1):
                xb = np.stack([preprocess(im if rep == 0 else augment(im, rng)) for im in imgs])
                X.append(features(bb, xb).astype(np.float16))
                Y += [t for _, t in chunk]
        return np.concatenate(X), Y

    Xv, Yv = feats(val, 0)
    Xt, Yt = feats(train, a.aug)
    print(f"features: train {Xt.shape} val {Xv.shape} in {time.time() - t0:.0f}s", flush=True)

    h = model.head
    enc, cls = h.ctc_encoder, h.ctc_head
    chars = reader.ctc_chars
    idx = {c: i for i, c in enumerate(chars)}

    def logits(xf, train_mode):
        enc.train() if train_mode else enc.eval()
        cls.train() if train_mode else cls.eval()
        o = cls(enc(paddle.to_tensor(xf.astype(np.float32))))
        return o["res"] if isinstance(o, dict) else o

    def evaluate():
        ok, edits, tot, wrong = 0, 0, 0, []
        with paddle.no_grad():
            for k in range(0, len(Xv), 64):
                lg = logits(Xv[k:k + 64], False)
                pr = paddle.nn.functional.softmax(lg, -1).numpy() if not _is_softmaxed(lg) else lg.numpy()
                for p, gt in zip(pr, Yv[k:k + 64]):
                    pred = ipv.normalize(reader._decode(p)["raw"])
                    ok += pred == gt; e = lev(pred, gt); edits += e; tot += len(gt)
                    if pred != gt:
                        wrong.append((gt, pred))
        return ok / max(len(Yv), 1), 1 - edits / max(tot, 1), wrong

    def _is_softmaxed(lg):
        s = lg.numpy()[0].sum(-1)
        return bool(np.allclose(s, 1.0, atol=1e-3))

    base_exact, base_char, _ = evaluate()
    print(f"original model on validation: exact {base_exact:.4f}  char {base_char:.4f}", flush=True)
    params = list(cls.parameters()) + (list(enc.parameters()) if a.train_encoder else [])
    keep = {id(p) for p in params}
    for p in model.parameters():
        p.stop_gradient = id(p) not in keep
    steps = math.ceil(len(Xt) / a.batch) * a.epochs
    sched = paddle.optimizer.lr.LinearWarmup(paddle.optimizer.lr.CosineAnnealingDecay(a.lr, T_max=steps),
                                             warmup_steps=max(1, steps // 20), start_lr=a.lr / 10, end_lr=a.lr)
    opt = paddle.optimizer.AdamW(sched, parameters=params, weight_decay=1e-4,
                                 grad_clip=paddle.nn.ClipGradByGlobalNorm(5.0))
    ctc = paddle.nn.CTCLoss(blank=0, reduction="none")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"method": "frozen original backbone, trained " + ("SVTR neck + CTC head" if a.train_encoder else "CTC head"),
              "settings": vars(a), "train_images": len(train), "train_views": int(len(Xt)), "val": len(val),
              "original": {"exact": base_exact, "char": base_char}, "epochs": []}
    best, best_ep = base_exact, 0
    for ep in range(1, a.epochs + 1):
        order = rng.permutation(len(Xt)); losses = []; t1 = time.time()
        for k in range(0, len(order), a.batch):
            ib = order[k:k + a.batch]
            labs = [[idx[c] for c in Yt[i]] for i in ib]
            L = max(len(l) for l in labs)
            lab = np.array([l + [0] * (L - len(l)) for l in labs], np.int32)
            lg = logits(Xt[ib], True)
            T = lg.shape[1]
            # training-mode CTC head returns raw logits
            loss = ctc(lg.transpose([1, 0, 2]), paddle.to_tensor(lab),
                       paddle.to_tensor([T] * len(ib), dtype="int64"),
                       paddle.to_tensor([len(l) for l in labs], dtype="int64")).mean()
            loss.backward(); opt.step(); opt.clear_grad(); sched.step()
            losses.append(float(loss))
        ex, ch, wrong = evaluate()
        row = {"epoch": ep, "loss": round(float(np.mean(losses)), 4), "val_exact": round(ex, 4),
               "val_char": round(ch, 4), "seconds": round(time.time() - t1, 1)}
        report["epochs"].append(row); print(row, flush=True)
        if ex > best:
            best, best_ep = ex, ep
            save_weights(model, out / "candidate_head.safetensors")
            report["best"] = dict(row, wrong=wrong[:60])
        if ep - best_ep >= a.patience:
            print(f"early stopping (no improvement for {a.patience} epochs)"); break
    report["best_epoch"] = best_ep
    deployed = False
    if a.deploy and best_ep and best > base_exact:
        # sanity: the saved backbone must be bit-identical to the original
        orig = load_file(str(cfg.AWIROS_ORIGINAL_WEIGHTS)); new = load_file(str(out / "candidate_head.safetensors"))
        from awiros_ocr import flush_denormals
        same = all(np.array_equal(flush_denormals(orig[k]), new[k]) for k in orig if k.startswith("backbone."))
        report["backbone_identical_to_original"] = bool(same)
        if same:
            import shutil
            shutil.copyfile(out / "candidate_head.safetensors", out / "model.safetensors")
            deployed = True
    report["deployed"] = deployed
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"original {base_exact:.4f} -> best {best:.4f} (epoch {best_ep}); "
          f"{'DEPLOYED ' + str(out / 'model.safetensors') if deployed else 'not deployed'}")


if __name__ == "__main__":
    main()
