"""Awiros ANPR OCR: the project's text-recognition engine (replaces PARSeq).

Model: Awiros/anpr-ocr (Hugging Face), PP-OCRv5 SVTR_HGNet = PPHGNetV2_B4 backbone +
MultiHead(CTCHead, NRTRHead), 37.3M parameters, weights in model.safetensors, character set
en_dict.txt (0-9 A-Z a-z, plus space). The architecture is built with PaddleOCR's own `ppocr`
package, exactly as the official test.py does; no weights are changed.

How the prediction is produced (checked against the PaddleOCR source):
  * In eval mode PaddleOCR's MultiHead returns ONLY the CTC head output (already softmaxed).
    The NRTR head is a training-time guidance head (GTC); the official inference never uses it,
    so neither does this module. There is no voting between heads.
  * Input: BGR crop resized to height 48 keeping aspect (width ceil(48*w/h), at most 320),
    scaled to [-1, 1], then padded on the right with 0 up to 320 - PaddleOCR's
    `resize_norm_img`, the transform PP-OCRv5 is trained with. The official test.py pads with
    black BEFORE normalising (= -1); on Awiros' own sample images that drops a character on
    4 of 9 plates, while the PaddleOCR transform reads all 9 correctly (see README).
  * Decoding: greedy CTC (argmax per frame, merge repeats, drop blanks).

Confidence (genuine model probabilities, nothing invented):
  * per character: the CTC softmax probability of that character at its strongest frame,
    with upper and lower case summed (as the plate is upper-case only);
  * `alts`: the other characters' probabilities at that same frame (used for review suggestions);
  * `avg`: geometric mean of the per-character probabilities; `weakest`: their minimum;
  * `ctc_mean_conf`: PaddleOCR's own CTCLabelDecode confidence (mean of the first-frame
    probabilities), i.e. the number the official test.py prints.
  CTC cannot express a character the model skipped entirely, so a dropped character is not
  visible in these numbers; the registration validator and the review rules cover that case.
"""
import hashlib
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

import anpr_config as cfg

log = logging.getLogger("awiros_ocr")

ENGINE_NAME = cfg.OCR_ENGINE_NAME
_GROUPS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Architecture of the released checkpoint (identical to MODEL_CONFIG in Awiros' test.py).
# CTC head: 64 classes (63 dictionary characters + blank). NRTR head: 67 (+1 added internally).
MODEL_ARCH = {
    "model_type": "rec",
    "algorithm": "SVTR_HGNet",
    "Transform": None,
    "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
    "Head": {
        "name": "MultiHead",
        "out_channels_list": {"CTCLabelDecode": 64, "NRTRLabelDecode": 67},
        "head_list": [
            {"CTCHead": {"Neck": {"name": "svtr", "dims": 120, "depth": 2, "hidden_dims": 120,
                                  "kernel_size": [1, 3], "use_guide": True},
                         "Head": {"fc_decay": 1e-05}}},
            {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
        ],
    },
}


def flush_denormals(a):
    """Set float32 values smaller than the smallest normal float (|x| < 1.18e-38) to 0.

    58 % of the released checkpoint's values (21.8M of 37.3M) are such denormals - channels that
    weight decay shrank to ~1e-39. x86 CPUs process denormal arithmetic ~100x slower, which made
    the eager model take ~105 s per 16 crops on CPU (training: 16.8 h per epoch). After the flush:
    1.6 s, with bit-identical CTC probabilities (max difference 0.0, checked 2026-11-11). The file
    on disk is never changed; this only affects the in-memory copy."""
    if getattr(a, "dtype", None) != np.float32:
        return a
    tiny = np.finfo(np.float32).tiny
    m = (a != 0) & (np.abs(a) < tiny)
    if not m.any():
        return a
    a = a.copy()
    a[m] = 0.0
    return a


class AwirosUnavailable(RuntimeError):
    """Weights, dictionary or a dependency (paddle, ppocr, safetensors) is missing."""


def preprocess(img_bgr, shape=cfg.AWIROS_INPUT_SHAPE):
    """BGR uint8 crop -> float32 CHW tensor in [-1, 1], PaddleOCR `resize_norm_img`."""
    c, h, w = shape
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    ih, iw = img_bgr.shape[:2]
    rw = min(w, int(math.ceil(h * iw / float(max(ih, 1)))))
    rw = max(rw, 1)
    resized = cv2.resize(img_bgr, (rw, h)).astype("float32")
    resized = resized.transpose((2, 0, 1)) / 255.0
    resized = (resized - 0.5) / 0.5
    out = np.zeros((c, h, w), dtype=np.float32)
    out[:, :, :rw] = resized
    return out


def is_two_line(crop):
    """Square-ish crops are usually dual-row plates. Awiros reads both rows in one pass,
    so this is only reported, never used to split the crop."""
    h, w = crop.shape[:2]
    return (w / max(h, 1)) < cfg.TWO_LINE_ASPECT


class AwirosReader:
    """Loads the model once; `read_chars` / `read_plate_chars` keep the interface the pipeline
    used with PARSeq, so detection, preprocessing, validation and decisions are unchanged."""

    engine_name = ENGINE_NAME

    def __init__(self, device="cpu", weights=None, dict_path=None, paddleocr_dir=None,
                 batch_size=cfg.AWIROS_BATCH_SIZE):
        weights = Path(weights or cfg.AWIROS_WEIGHTS)
        dict_path = Path(dict_path or cfg.AWIROS_DICT)
        paddleocr_dir = Path(paddleocr_dir or cfg.PADDLEOCR_DIR)
        for p, what in ((weights, "model.safetensors"), (dict_path, "en_dict.txt")):
            if not p.is_file():
                raise AwirosUnavailable(
                    f"Awiros {what} not found at {p}. Download https://huggingface.co/Awiros/anpr-ocr "
                    f"into {cfg.AWIROS_DIR} or set AWIROS_DIR.")
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")  # no GPU pre-allocation
        try:
            import paddle
            from safetensors.numpy import load_file
        except ImportError as e:
            raise AwirosUnavailable(f"{e.name} is not installed: pip install paddlepaddle "
                                    f"(or paddlepaddle-gpu) safetensors") from e
        if not (paddleocr_dir / "ppocr" / "__init__.py").is_file():
            raise AwirosUnavailable(
                f"PaddleOCR (ppocr package) not found at {paddleocr_dir}. Run: git clone --depth 1 "
                f"https://github.com/PaddlePaddle/PaddleOCR.git \"{paddleocr_dir}\"")
        if str(paddleocr_dir) not in sys.path:
            sys.path.append(str(paddleocr_dir))
        try:
            from ppocr.modeling.architectures import build_model
        except ImportError as e:
            raise AwirosUnavailable(f"ppocr could not be imported ({e}). Check the PaddleOCR "
                                    f"clone and run: pip install -r requirements.txt") from e

        self.paddle = paddle
        want_gpu = str(device).lower() in ("cuda", "gpu", "0")
        if want_gpu and paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            paddle.set_device("gpu:0")
            self.device = "gpu"
        else:
            if want_gpu:
                log.warning("Paddle has no CUDA here (install paddlepaddle-gpu); Awiros runs on CPU.")
            paddle.set_device("cpu")
            self.device = "cpu"

        import copy
        self.model = build_model(copy.deepcopy(MODEL_ARCH))
        state = {k: paddle.to_tensor(flush_denormals(v) if cfg.AWIROS_FLUSH_DENORMALS else v)
                 for k, v in load_file(str(weights)).items()}
        missing, unexpected = self._load_state(state)
        if missing or unexpected:
            raise AwirosUnavailable(f"Checkpoint does not match the architecture "
                                    f"(missing {missing[:3]}, unexpected {unexpected[:3]})")
        self.model.eval()
        self.weights = weights
        self._predictor = None
        self.inference_mode = "eager"
        if self.device == "cpu" and cfg.AWIROS_FAST_CPU:
            try:
                self._predictor = self._build_cpu_predictor(weights)
            except Exception:
                log.exception("Fast CPU inference unavailable; using the (slow) eager model")

        chars = [ln.rstrip("\r\n") for ln in open(dict_path, encoding="utf-8")]
        chars = [c for c in chars if c] + [" "]                 # use_space_char=True
        self.ctc_chars = ["<blank>"] + chars                     # CTC class 0 = blank
        if len(self.ctc_chars) != MODEL_ARCH["Head"]["out_channels_list"]["CTCLabelDecode"]:
            raise AwirosUnavailable(f"{dict_path} has {len(chars)} characters; the model expects 63")
        G = np.zeros((len(self.ctc_chars), len(_GROUPS)), dtype=np.float32)
        for i, ch in enumerate(self.ctc_chars):
            if len(ch) == 1 and ch.upper() in _GROUPS:
                G[i, _GROUPS.index(ch.upper())] = 1.0
        self._group_mat = G
        self.batch_size = max(1, int(batch_size))
        self._cache = OrderedDict()
        self.cache_hits = 0
        self.images_read = 0
        self.total_infer_ms = 0.0
        self.last_batch_ms = 0.0
        t0 = time.perf_counter()                    # warm-up: the first call builds kernels
        self._forward(np.zeros((1,) + tuple(cfg.AWIROS_INPUT_SHAPE), np.float32))
        self.warmup_ms = (time.perf_counter() - t0) * 1000.0
        log.info("Awiros ANPR OCR loaded from %s on %s", weights, self.device)

    def _build_cpu_predictor(self, weights):
        """Export the eager model once to a static inference graph (keyed by the weights file's
        size and mtime) and load it with oneDNN. Eager Paddle does not use oneDNN for the
        backbone convolutions, which makes it ~50x slower on CPU."""
        import json
        from paddle import inference as pi
        paddle = self.paddle
        st = weights.stat()
        tag = {"weights": str(weights), "size": st.st_size, "mtime": int(st.st_mtime),
               "paddle": paddle.__version__}
        out = Path(cfg.AWIROS_INFERENCE_DIR)
        if Path(weights).resolve() != Path(cfg.AWIROS_ORIGINAL_WEIGHTS).resolve():
            # one export per weights file, so benchmarking another model never overwrites (or
            # races with) the export the server is using
            out = out / ("w_" + hashlib.blake2b(json.dumps(tag, sort_keys=True).encode(),
                                                digest_size=6).hexdigest())
        model_file, params = out / "inference.json", out / "inference.pdiparams"
        tag_file = out / "source.json"
        if not (model_file.is_file() and params.is_file() and tag_file.is_file()
                and json.loads(tag_file.read_text()) == tag):
            out.mkdir(parents=True, exist_ok=True)
            spec = [paddle.static.InputSpec([None] + list(cfg.AWIROS_INPUT_SHAPE), "float32", name="x")]
            static = paddle.jit.to_static(self.model, input_spec=spec, full_graph=True)
            paddle.jit.save(static, str(out / "inference"))
            tag_file.write_text(json.dumps(tag))
            log.info("Exported Awiros inference graph to %s", out)
        c = pi.Config(str(model_file), str(params))
        c.disable_gpu()
        c.enable_mkldnn()
        c.set_cpu_math_library_num_threads(cfg.AWIROS_CPU_THREADS)
        c.disable_glog_info()
        pred = pi.create_predictor(c)
        self._in = pred.get_input_handle(pred.get_input_names()[0])
        self._out = pred.get_output_handle(pred.get_output_names()[0])
        self.inference_mode = "static+oneDNN"
        return pred

    def _load_state(self, state):
        own = self.model.state_dict()
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        self.model.set_state_dict(state)
        return missing, unexpected

    def describe(self):
        return {"engine": ENGINE_NAME, "device": self.device, "weights": str(self.weights),
                "input_shape": list(cfg.AWIROS_INPUT_SHAPE), "decoder": "CTC (greedy)",
                "inference": self.inference_mode}

    def gpu_memory_mb(self):
        """Peak GPU memory used by Paddle, or None on CPU."""
        if self.device != "gpu":
            return None
        try:
            return self.paddle.device.cuda.max_memory_allocated() / 2 ** 20
        except Exception:
            return None

    # ------------------------------------------------------------------ inference
    def _forward(self, batch):
        """batch: float32 NCHW -> CTC probabilities N x T x 64. Halves the batch on GPU OOM."""
        paddle = self.paddle
        if self._predictor is not None:
            self._in.reshape(list(batch.shape))
            self._in.copy_from_cpu(np.ascontiguousarray(batch, dtype=np.float32))
            self._predictor.run()
            return self._out.copy_to_cpu()
        try:
            with paddle.no_grad():
                out = self.model(paddle.to_tensor(batch))
            if isinstance(out, dict):           # defensive: eval MultiHead returns the CTC tensor
                out = out["ctc"]
            return out.numpy()
        except (MemoryError, RuntimeError) as e:
            if len(batch) > 1 and "memory" in str(e).lower():
                half = len(batch) // 2
                log.warning("Out of memory on a batch of %d; retrying in halves", len(batch))
                return np.concatenate([self._forward(batch[:half]), self._forward(batch[half:])])
            raise

    def _decode(self, probs):
        """probs: T x 64 for one image -> result dict (see module docstring)."""
        ids = probs.argmax(-1)
        raw, chars, first_frame_p = "", [], []
        t = 0
        T = len(ids)
        while t < T:
            c = int(ids[t])
            e = t
            while e + 1 < T and int(ids[e + 1]) == c:
                e += 1
            if c != 0:                                   # not blank
                tok = self.ctc_chars[c]
                raw += tok
                first_frame_p.append(float(probs[t, c]))
                if self._group_mat[c].any():             # a plate character, not a space
                    f = t + int(np.argmax(probs[t:e + 1, c]))
                    g = probs[f] @ self._group_mat       # upper + lower case summed
                    order = np.argsort(-g)[:4]
                    alts = [(_GROUPS[j], float(g[j])) for j in order]
                    gi = _GROUPS.index(tok.upper())
                    chars.append({"char": _GROUPS[gi], "p": float(g[gi]), "alts": alts})
            t = e + 1
        ps = [ch["p"] for ch in chars]
        return {"raw": raw, "chars": chars,
                "avg": math.exp(sum(math.log(max(p, 1e-9)) for p in ps) / len(ps)) if ps else 0.0,
                "weakest": min(ps) if ps else 0.0,
                "ctc_mean_conf": float(np.mean(first_frame_p)) if first_frame_p else 0.0,
                # full CTC probability matrix (T x 64): needed for temperature-scaled confidence
                # (ocr_calibration.py / hybrid_verifier.py). float16 to keep the OCR cache small.
                "probs": probs.astype(np.float16),
                "engine": ENGINE_NAME, "error": ""}

    @staticmethod
    def _key(im):
        return hashlib.blake2b(im.tobytes() + str(im.shape).encode(), digest_size=16).digest()

    def read_chars(self, bgr_images, cache_size=cfg.OCR_CACHE_SIZE):
        """List of BGR crops -> list of result dicts:
            {"raw", "chars": [{"char", "p", "alts"}], "avg", "weakest", "ctc_mean_conf",
             "infer_ms", "engine", "error"}
        Identical crops are read once (cache). An inference error does not raise: the affected
        crops come back with empty text and `error` set (status OCR_FAILED downstream)."""
        results = [None] * len(bgr_images)
        todo = []
        for i, im in enumerate(bgr_images):
            if im is None or im.size == 0:
                results[i] = self._failed("empty crop")
                continue
            k = self._key(im)
            if k in self._cache:
                self._cache.move_to_end(k)
                results[i] = dict(self._cache[k], infer_ms=0.0)
                self.cache_hits += 1
            else:
                todo.append((i, k, im))
        for s in range(0, len(todo), self.batch_size):
            chunk = todo[s:s + self.batch_size]
            try:
                batch = np.stack([preprocess(im) for _, _, im in chunk])
                t0 = time.perf_counter()
                probs = self._forward(batch)
                ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:                        # keep the job alive, report the failure
                log.exception("Awiros inference failed on %d crop(s)", len(chunk))
                for i, _, _ in chunk:
                    results[i] = self._failed(f"{type(e).__name__}: {e}")
                continue
            self.last_batch_ms = ms
            self.total_infer_ms += ms
            self.images_read += len(chunk)
            for (i, k, _), p in zip(chunk, probs):
                res = self._decode(p)
                res["infer_ms"] = ms / len(chunk)
                results[i] = res
                self._cache[k] = res
                if len(self._cache) > cache_size:
                    self._cache.popitem(last=False)
        return results

    @staticmethod
    def _failed(msg):
        return {"raw": "", "chars": [], "avg": 0.0, "weakest": 0.0, "ctc_mean_conf": 0.0,
                "infer_ms": 0.0, "engine": ENGINE_NAME, "error": msg}

    def read_plate_chars(self, crops, two_line="auto"):
        """Read many plate crops in batched calls. Dual-row plates are read whole (the model
        was trained on them); `two_line` is only recorded."""
        reads = self.read_chars(crops)
        return [dict(r, two_line=bool(is_two_line(c)) if two_line == "auto" else bool(two_line))
                for r, c in zip(reads, crops)]

    def read(self, bgr_images):
        """List of BGR crops -> list of (normalised text, avg confidence)."""
        from indian_plate_validator import normalize
        return [(normalize(r["raw"]), r["avg"]) for r in self.read_chars(bgr_images)]


def main():
    """Read plate crops from the command line: python src/awiros_ocr.py crop1.jpg dir/ ..."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Awiros ANPR OCR on plate crops")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = []
    for p in map(Path, args.paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in
                        (".jpg", ".jpeg", ".png", ".bmp", ".webp")) if p.is_dir() else [p]
    reader = AwirosReader(args.device)
    pairs = [(f, cv2.imread(str(f))) for f in files]
    pairs = [(f, im) for f, im in pairs if im is not None]
    for (f, _), r in zip(pairs, reader.read_plate_chars([im for _, im in pairs])):
        print(json.dumps({"image": f.name, "raw": r["raw"], "avg_conf": round(r["avg"], 4),
                          "weakest_char": round(r["weakest"], 4),
                          "ctc_mean_conf": round(r["ctc_mean_conf"], 4),
                          "infer_ms": round(r["infer_ms"], 2), "two_line": r["two_line"]}))


if __name__ == "__main__":
    main()
