"""Second, independent OCR engine: PaddleOCR DET -> CLS -> REC on the YOLO plate crop.

Models (downloaded once by setup_paddleocr_models.py, paths in anpr_config):
  DET  PP-OCRv5_mobile_det          finds the text lines inside the plate crop (2 on dual-row plates)
  CLS  PP-LCNet_x1_0_textline_ori   0 / 180 degree orientation of each line        (Phase 4)
  REC  en_PP-OCRv5_mobile_rec       reads each line                                 (Phase 5)

They run with paddle.inference and the pre/post-processing operators of the PaddleOCR clone that
Awiros already uses (`ppocr`); every step's parameters come from that model's own inference.yml.
No `paddleocr` / `paddlex` pip package, no PARSeq, nothing shared with Awiros except the Paddle
runtime. GPU is used when asked for and available (paddlepaddle-gpu), otherwise CPU.
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import anpr_config as cfg

log = logging.getLogger("paddle_ocr_engine")
ENGINE_NAME = "PaddleOCR (PP-OCRv5 mobile DET/CLS/REC)"


class PaddleOCRUnavailable(RuntimeError):
    """A model, the PaddleOCR clone or paddlepaddle is missing."""


def _import_ppocr():
    d = Path(cfg.PADDLEOCR_DIR)
    if not (d / "ppocr" / "__init__.py").is_file():
        raise PaddleOCRUnavailable(f"PaddleOCR clone (ppocr) not found at {d}")
    if str(d) not in sys.path:
        sys.path.append(str(d))


def _number(v):
    """inference.yml writes some scales as fractions, e.g. "1./255."."""
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/")
        return float(a) / float(b)
    return float(v)


def resolve_device(device):
    """'cuda' only if asked for and paddle can actually use a GPU; otherwise 'cpu'."""
    if str(device) not in ("cuda", "0", "gpu"):
        return "cpu"
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "cuda"
    except Exception:            # noqa: BLE001 - any CUDA problem means CPU
        pass
    log.warning("PaddleOCR: CUDA requested but not available, using CPU")
    return "cpu"


def load_model(folder, role, device):
    """paddle.inference predictor + the model's inference.yml. role: det / cls / rec."""
    folder = Path(folder)
    for f in ("inference.json", "inference.pdiparams", "inference.yml"):
        if not (folder / f).is_file():
            raise PaddleOCRUnavailable(f"{folder / f} missing: run python src/setup_paddleocr_models.py")
    try:
        from paddle import inference
    except ImportError as e:
        raise PaddleOCRUnavailable("paddlepaddle is not installed") from e
    conf = yaml.safe_load((folder / "inference.yml").read_text(encoding="utf-8"))
    c = inference.Config(str(folder / "inference.json"), str(folder / "inference.pdiparams"))
    if device == "cuda":
        c.enable_use_gpu(256, 0)
    else:
        c.disable_gpu()
        c.set_cpu_math_library_num_threads(cfg.PADDLE_CPU_THREADS)
        c.enable_mkldnn() if cfg.PADDLE_ONEDNN.get(role, False) else c.disable_mkldnn()
    c.disable_glog_info()
    c.switch_ir_optim(True)    # (enable_memory_optim fails with these PIR exports in paddle 3.3.1)
    p = inference.create_predictor(c)
    return p, conf


def run(predictor, batch):
    """One forward pass; returns the first output as numpy."""
    h = predictor.get_input_handle(predictor.get_input_names()[0])
    h.reshape(list(batch.shape))
    h.copy_from_cpu(np.ascontiguousarray(batch, dtype=np.float32))
    predictor.run()
    return predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()


def order_lines(boxes):
    """Reading order of text lines: top to bottom; boxes whose vertical centres are within half a
    line height of each other are one row, read left to right."""
    if not boxes:
        return []
    items = sorted(boxes, key=lambda b: (b["center"][1], b["center"][0]))
    rows, cur = [], [items[0]]
    for b in items[1:]:
        ref = cur[-1]
        if abs(b["center"][1] - ref["center"][1]) < 0.5 * min(b["height"], ref["height"]):
            cur.append(b)
        else:
            rows.append(cur); cur = [b]
    rows.append(cur)
    return [b for row in rows for b in sorted(row, key=lambda b: b["center"][0])]


def crop_quad(img, points):
    """Perspective-rectified crop of one detected quad (PaddleOCR's get_rotate_crop_image):
    upright, and turned 90 degrees when the quad is much taller than wide."""
    import cv2
    pts = np.asarray(points, dtype=np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h / float(w) >= 1.5:
        out = np.rot90(out)
    return np.ascontiguousarray(out)


class PaddleDet:
    """PP-OCRv5_mobile_det: text-line quads inside a plate crop, with DB box scores."""

    def __init__(self, device="cpu", folder=None):
        _import_ppocr()
        from ppocr.postprocess.db_postprocess import DBPostProcess
        self.device = device
        self.predictor, conf = load_model(folder or cfg.PADDLE_DET_MODEL, "det", device)
        self.model_name = conf["Global"]["model_name"]
        ops = {next(iter(op)): (next(iter(op.values())) or {}) for op in conf["PreProcess"]["transform_ops"]}
        norm = ops["NormalizeImage"]
        self.mean = np.array(norm["mean"], np.float32).reshape(1, 1, 3)
        self.std = np.array(norm["std"], np.float32).reshape(1, 1, 3)
        self.scale = _number(norm.get("scale", 1 / 255.0))   # "1./255." in the yml
        self.resize_long = cfg.PADDLE_DET_RESIZE_LONG or ops["DetResizeForTest"].get("resize_long", 960)
        post = dict(conf["PostProcess"])
        post.pop("name", None)
        post.update(box_thresh=cfg.PADDLE_DET_BOX_THRESH, unclip_ratio=cfg.PADDLE_DET_UNCLIP)
        self.post = DBPostProcess(**post)

    def preprocess(self, img_bgr):
        """inference.yml: DetResizeForTest(resize_long) -> NormalizeImage(hwc) -> ToCHWImage.
        Same arithmetic as ppocr.data.imaug (resize_image_type2: long side to resize_long, both
        sides rounded up to a multiple of 128), without importing ppocr.data, whose training
        augmentations need albumentations."""
        import cv2
        h, w = img_bgr.shape[:2]
        ratio = float(self.resize_long) / max(h, w)
        rh = max(128, (int(h * ratio) + 127) // 128 * 128)
        rw = max(128, (int(w * ratio) + 127) // 128 * 128)
        img = cv2.resize(img_bgr, (rw, rh)).astype(np.float32)
        img = (img * self.scale - self.mean) / self.std
        return img.transpose(2, 0, 1), (h, w)

    def detect(self, img_bgr):
        """-> list of {"points": 4x2 [x,y] (clockwise from top-left), "score", "center",
        "height", "width"} in reading order. Empty list = no text line found."""
        if img_bgr is None or img_bgr.size == 0:
            return []
        if img_bgr.ndim == 2:
            import cv2
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        H, W = img_bgr.shape[:2]
        # Tight plate crops: the text fills the image, but DB detection was trained on scenes
        # with context around text and breaks such lines up or misses them. A replicated border
        # (fraction of the crop height) restores that context; boxes are shifted back after.
        pad = int(round(cfg.PADDLE_DET_PAD * H))
        if pad:
            import cv2
            img_bgr = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x, (src_h, src_w) = self.preprocess(img_bgr)
        pred = run(self.predictor, x[None])[:, 0]
        mask = pred[0] > self.post.thresh
        if self.post.dilation_kernel is not None:
            import cv2
            mask = cv2.dilate(mask.astype(np.uint8), self.post.dilation_kernel)
        quads, scores = self.post.boxes_from_bitmap(pred[0], mask, src_w, src_h)
        out = []
        for q, s in zip(quads, scores):
            q = np.asarray(q, dtype=np.float32) - pad
            q[:, 0] = q[:, 0].clip(0, W - 1)
            q[:, 1] = q[:, 1].clip(0, H - 1)
            w = float(np.linalg.norm(q[0] - q[1]))
            h = float(np.linalg.norm(q[0] - q[3]))
            if w < 3 or h < 3:
                continue
            out.append({"points": q.tolist(), "score": float(s), "center": q.mean(0).tolist(),
                        "height": h, "width": w})
        return order_lines(out)


class PaddleOCREngine:
    """DET -> CLS -> REC on one plate crop. Phase 3: DET only."""

    engine_name = ENGINE_NAME

    def __init__(self, device="cpu"):
        self.device = resolve_device(device)
        t0 = time.perf_counter()
        self.det = PaddleDet(self.device)
        self.load_ms = (time.perf_counter() - t0) * 1000.0

    def detect_lines(self, plate_bgr):
        """-> (lines, crops, det_ms): quads in reading order and their rectified crops."""
        t0 = time.perf_counter()
        lines = self.det.detect(plate_bgr)
        crops = [crop_quad(plate_bgr, ln["points"]) for ln in lines]
        return lines, crops, (time.perf_counter() - t0) * 1000.0

    def describe(self):
        return {"engine": self.engine_name, "device": self.device,
                "det": self.det.model_name}
