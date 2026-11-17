"""Image -> YOLO11 plate boxes -> merged plate -> Awiros ANPR OCR (several crops, then
quality-driven OpenCV retries) -> Indian format validation -> SUCCESS / REVIEW / REJECT
(+ ML verifier) -> saved row.

The OCR engine is Awiros (awiros_ocr.py). `ocr_engine="parseq"` exists only so evaluate.py can
measure the old engine on the same images; the server never uses it.

No web code here. `analyze()` does no file I/O (used by evaluate.py); `process()` = analyze + save."""
import hashlib
import time
from collections import OrderedDict
from pathlib import Path

import cv2

import anpr_config as cfg
import dashboard_log as dl
import plate_decision as pdec
import plate_preprocess as pp
from plate_reader import pad_box

PREVIEW_W = 800                 # boxed preview images are shrunk to this width
MERGE_GAP = 10                  # px: boxes closer than this belong to the same plate
_RANK = {pdec.SUCCESS: 3, pdec.REVIEW: 2, pdec.REJECT: 1}


def _union(g):
    return [min(b[0] for b in g), min(b[1] for b in g),
            max(b[2] for b in g), max(b[3] for b in g), max(b[4] for b in g)]


def merge_boxes(boxes, gap=MERGE_GAP):
    """Merge boxes that touch or overlap into one box around the whole plate.
    boxes: list of [x1, y1, x2, y2, conf]. Returns list of (union_box, member_boxes)."""
    groups = [[b] for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            a = _union(groups[i])
            for j in range(i + 1, len(groups)):
                b = _union(groups[j])
                if (a[0] - gap < b[2] and b[0] - gap < a[2]
                        and a[1] - gap < b[3] and b[1] - gap < a[3]):
                    groups[i] += groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [(_union(g), g) for g in groups]


def _crop(frame, b, px, py=cfg.V_PAD):
    H, W = frame.shape[:2]
    bw, bh = b[2] - b[0], b[3] - b[1]
    x1 = max(0, int(b[0] - bw * px)); x2 = min(W, int(b[2] + bw * px))
    y1 = max(0, int(b[1] - bh * py)); y2 = min(H, int(b[3] + bh * py))
    return frame[y1:y2, x1:x2]


def looks_like_plate_crop(W, H, boxes):
    """True when the uploaded image is itself (most likely) a plate crop, e.g. a pump-camera
    plate image: plate-shaped, and YOLO found nothing or a box covering a large part of it.
    A plate in a full vehicle photo covers only a few percent of the image."""
    if not (cfg.AUTO_CROP_ASPECT[0] <= W / max(H, 1) <= cfg.AUTO_CROP_ASPECT[1]):
        return False
    if max(W, H) > cfg.AUTO_CROP_MAX_SIDE:
        return False
    if not boxes:
        return True
    big = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / float(W * H)
    return big >= cfg.AUTO_CROP_MIN_BOX_FRACTION


class Pipeline:
    def __init__(self, weights, device="cpu", use_ml=cfg.USE_ML_VERIFIER, ocr_engine="awiros",
                 use_resolver=True, use_hybrid=True):
        if ocr_engine == "awiros":
            # Paddle must be loaded BEFORE PyTorch (ultralytics). The other way round, Paddle's
            # oneDNN kernels run with PyTorch's OpenMP runtime and Awiros becomes ~25x slower
            # after the first YOLO call (measured 2026-11-11: 3 crops 13 s -> 0.5 s). Same output.
            try:
                import paddle  # noqa: F401
            except ImportError:
                pass
        from ultralytics import YOLO

        self.device = "cuda" if str(device) in ("0", "cuda") else "cpu"
        self.detector = YOLO(str(weights))                  # loaded once
        if ocr_engine == "awiros":
            from awiros_ocr import AwirosReader
            self.reader = AwirosReader(self.device)          # loaded once
        elif ocr_engine == "parseq":                        # benchmark baseline only
            from plate_reader import PARSeqReader
            self.reader = PARSeqReader("parseq", self.device)
            self.reader.engine_name = "PARSeq"
        else:
            raise ValueError(f"unknown OCR engine {ocr_engine!r}")
        self.ocr_engine = getattr(self.reader, "engine_name", ocr_engine)
        self._image_cache = OrderedDict()
        self.verifier = None
        if use_ml:
            from ocr_verifier import OCRVerifier
            self.verifier = OCRVerifier.load_default()      # None until train_verifier.py has run
        self.hybrid = None
        if use_hybrid and ocr_engine == "awiros":
            from hybrid_verifier import HybridVerifier
            self.hybrid = HybridVerifier.load_default(self.reader)
        self.ocr_model = Path(str(getattr(self.reader, "weights", ""))).parent.name or self.ocr_engine
        self.resolver = None
        if use_resolver and cfg.USE_REVIEW_RESOLVER:
            from review_resolver import ReviewResolver
            self.resolver = ReviewResolver.load_default(self.device)  # None until train_resolver.py

    # ------------------------------------------------------------------ OCR attempts
    def _read(self, named_imgs, timer=None):
        t0 = time.perf_counter()
        reads = self.reader.read_plate_chars([im for _, im in named_imgs])
        if timer is not None:
            timer[0] += (time.perf_counter() - t0) * 1000.0
        return [dict(r, variant=n, img=im) for (n, im), r in zip(named_imgs, reads)]

    def read_plate(self, frame, union_box, members, det_conf=None, crop_mode=False,
                   accept_avg=cfg.ACCEPT_AVG_CONF, accept_weakest=cfg.ACCEPT_WEAKEST_CHAR):
        """All OCR attempts + decision for one plate. Returns a plate dict (no file I/O)."""
        t0 = time.time()
        bw, bh = union_box[2] - union_box[0], union_box[3] - union_box[1]
        tight = frame if crop_mode else _crop(frame, union_box, cfg.BASE_H_PADS[0])
        quality = pp.assess(tight)
        ocr_ms = [0.0]
        kw = dict(det_conf=det_conf, quality=quality, accept_avg=accept_avg,
                  accept_weakest=accept_weakest,
                  box_aspect=None if crop_mode else bw / max(bh, 1))

        attempts = self._read([("original", tight)], ocr_ms)
        a = attempts[0]
        from indian_plate_validator import find_in_text
        v0 = find_in_text(a["raw"])
        early = (v0["is_valid_format"] and not v0["warnings"] and a["avg"] >= cfg.EARLY_EXIT_AVG
                 and a["weakest"] >= cfg.EARLY_EXIT_WEAKEST)
        if not early and not crop_mode:
            more = [(f"original_pad{int(px * 100)}", _crop(frame, union_box, px))
                    for px in cfg.BASE_H_PADS[1:]]
            if len(members) > 1:
                more += [(f"member{k}", _crop(frame, m, cfg.BASE_H_PADS[0]))
                         for k, m in enumerate(members)]
            attempts += self._read([(n, im) for n, im in more if im.size], ocr_ms)
        base_n = len(attempts)
        d = pdec.decide(attempts, **kw)
        base_status = d["status"]

        retries, rounds = 0, 0
        if d["status"] != pdec.SUCCESS and "UNUSABLE" not in quality["flags"]:
            for names in pp.plan_variants(quality)[: cfg.MAX_RETRY_ROUNDS]:
                names = names[: cfg.MAX_RETRY_VARIANTS - retries]
                if not names:
                    continue
                variants = pp.make_variants(tight, names)
                if not variants:
                    continue
                rounds += 1
                attempts += self._read(variants, ocr_ms)
                retries += len(variants)
                d = pdec.decide(attempts, **kw)
                if d["status"] == pdec.SUCCESS:
                    break

        H, W = frame.shape[:2]
        if crop_mode:
            box = (0, 0, W, H)
        else:
            box = pad_box(int(union_box[0]), int(union_box[1]), int(union_box[2]),
                          int(union_box[3]), W, H)
        chosen = d.get("chosen") or attempts[0]
        # un-enhanced source of the chosen reading: training data once a person verifies the text
        v = chosen.get("variant", "original")
        train_img = chosen["img"] if v.startswith(("original", "member")) else tight
        rule_status = d["status"]
        plate = {
            "box": box, "det_conf": det_conf, "crop": frame[box[1]:box[3], box[0]:box[2]],
            "enhanced": chosen["img"], "train_crop": train_img, "decision": d, "status": d["status"], "text": d["text"],
            "final_text": d["final_text"], "raw": chosen["raw"], "chars": chosen["chars"],
            "avg": d.get("avg", 0.0), "weakest": d.get("weakest", 0.0),
            "preprocessing": chosen["variant"], "ocr_attempts": len(attempts),
            "base_attempts": base_n, "retry_count": retries, "retry_rounds": rounds,
            "base_status": base_status, "quality": quality, "seconds": time.time() - t0,
            "two_line": bool(chosen.get("two_line")), "crop_mode": crop_mode,
            "box_aspect": kw["box_aspect"], "rule_status": rule_status, "ml_probability": None,
            "ocr_engine": self.ocr_engine, "ocr_ms": ocr_ms[0],
            "ocr_error": "; ".join(sorted({a.get("error", "") for a in attempts} - {""})),
        }
        if self.verifier is not None:
            self.verifier.apply(plate)      # may change status / final_text / reasons only
            plate.update(status=d["status"], final_text=d["final_text"])
        if self.resolver is not None and plate["status"] == pdec.REVIEW:
            # second layer: Awiros vote + PARSeq cross-check + ML on the un-enhanced crops
            seen, views = set(), []
            for att in attempts:
                if str(att.get("variant", "")).startswith(("original", "member")) and id(att["img"]) not in seen:
                    seen.add(id(att["img"])); views.append(att["img"])
            self.resolver.apply(plate, attempts, views or [tight])
        plate.update(calibrated_probability=None, hybrid_reasons=[], tta_result="", _chosen=chosen)
        plate["ocr_status"] = pdec.detailed_status(plate)
        return plate

    # ------------------------------------------------------------------ whole image
    def analyze(self, frame, det_conf=0.25, accept_avg=cfg.ACCEPT_AVG_CONF,
                accept_weakest=cfg.ACCEPT_WEAKEST_CHAR, crop_mode=False):
        """Returns a list of plate dicts (empty = no plate detected). No file I/O.
        The hybrid gate runs on the FINAL plates only, so it can change a status but never which
        reading (text) is chosen."""
        plates = self._analyze(frame, det_conf, accept_avg, accept_weakest, crop_mode)
        for p in plates:
            chosen = p.pop("_chosen", None)
            if self.hybrid is not None and chosen is not None:
                self.hybrid.apply(p, chosen)        # may only turn SUCCESS into REVIEW
                p["ocr_status"] = pdec.detailed_status(p)
        return plates

    def _analyze(self, frame, det_conf, accept_avg, accept_weakest, crop_mode):
        H, W = frame.shape[:2]
        whole = [0, 0, W, H, None]
        kw = dict(accept_avg=accept_avg, accept_weakest=accept_weakest)
        if crop_mode:
            return [self.read_plate(frame, whole, [whole], crop_mode=True, **kw)]
        res = self.detector.predict(frame, conf=det_conf, iou=0.4, verbose=False,
                                    device=self.device)[0].boxes
        raw = [list(map(float, b.xyxy[0].tolist())) + [float(b.conf[0])] for b in res]
        plates = [self.read_plate(frame, u, m, det_conf=u[4], **kw) for u, m in merge_boxes(raw)]
        if cfg.AUTO_CROP_MODE and looks_like_plate_crop(W, H, raw):
            full = self.read_plate(frame, whole, [whole], crop_mode=True, **kw)
            if not plates and full["status"] == pdec.REJECT:
                return []                       # no box and no registration-like text: NO PLATE
            return [self._pick_crop_reading(plates, full)]
        # the same registration from two boxes (duplicate detection) -> keep the stronger one
        best = {}
        for p in plates:
            k = p["text"] if p["status"] != pdec.REJECT and p["text"] else id(p)
            if k not in best or (_RANK[p["status"]], p["avg"]) > (_RANK[best[k]["status"]], best[k]["avg"]):
                best[k] = p
        return sorted(best.values(), key=lambda p: (-_RANK[p["status"]], -p["avg"]))

    @staticmethod
    def _pick_crop_reading(plates, whole):
        """The image is (probably) one plate crop: choose between the YOLO-box readings and the
        whole-image reading. If the other side read a different, confident, valid registration,
        the choice is unresolved and goes to manual review (never a silent pick)."""
        if not plates:
            whole["auto_crop"] = True
            return whole
        pool = plates + [whole]
        # status, then a reading without format warnings (e.g. a discontinued state code like DN
        # beside TN), then confidence
        best = max(pool, key=lambda p: (_RANK[p["status"]],
                                        not (p["decision"].get("validation") or {}).get("warnings"),
                                        p["avg"]))
        best["auto_crop"] = best is whole
        if best["status"] == pdec.SUCCESS and cfg.CROP_CONFLICT_MODE != "off":
            for p in pool:
                v = p["decision"].get("validation") or {}
                if cfg.CROP_CONFLICT_MODE == "success_only" and p["status"] != pdec.SUCCESS:
                    continue
                if (p is not best and v.get("is_valid_format") and p["text"] != best["text"]
                        and p["avg"] >= cfg.CONFLICT_MIN_CONF):
                    d = best["decision"]
                    d["reasons"] = list(d["reasons"]) + ["CONFLICTING_READINGS"]
                    d.update(status=pdec.REVIEW, final_text="",
                             manual_review_status=pdec.MANUAL_REVIEW_REQUIRED)
                    best.update(status=pdec.REVIEW, final_text="")
                    best["ocr_status"] = pdec.detailed_status(best)
                    break
        return best

    # ------------------------------------------------------------------ dashboard entry point
    def process(self, path, batch, det_conf=0.25, accept_thr=cfg.ACCEPT_AVG_CONF,
                display_name=None, accept_weakest=cfg.ACCEPT_WEAKEST_CHAR, crop_mode=False):
        """Read one image file and log one row per plate (or one NO PLATE row).
        Returns the plate dicts. Raises ValueError if the file is not a readable image."""
        t_image = time.time()
        name = display_name or Path(path).name
        data = Path(path).read_bytes()
        key = (hashlib.blake2b(data, digest_size=16).hexdigest(), round(det_conf, 3),
               round(accept_thr, 3), round(accept_weakest, 3), crop_mode)
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"{name}: could not open the image")
        H, W = frame.shape[:2]
        thumb = dl.save_thumb(frame)

        duplicate_of = ""
        if key in self._image_cache:                       # identical file seen before
            plates, duplicate_of = self._image_cache[key]
            self._image_cache.move_to_end(key)
        else:
            plates = self.analyze(frame, det_conf, accept_thr, accept_weakest, crop_mode)
            self._image_cache[key] = (plates, name)
            if len(self._image_cache) > cfg.IMAGE_CACHE_SIZE:
                self._image_cache.popitem(last=False)

        annotated = frame.copy()
        thick = max(2, W // 400)
        colors = {pdec.SUCCESS: (0, 200, 0), pdec.REVIEW: (0, 165, 255), pdec.REJECT: (60, 60, 230)}
        for p in plates:
            d = p["decision"]
            v = d["validation"]
            dl.save_result(
                batch, name, thumb, p["text"], p["avg"], p["weakest"], p["status"], p["crop"],
                enhanced_bgr=p["enhanced"], train_bgr=p.get("train_crop"),
                det_conf=p["det_conf"], raw_ocr_text=p["raw"],
                normalized_ocr_text="".join(c["char"] for c in p["chars"]),
                final_registration_number=p["final_text"],
                format_validation_status="VALID" if v.get("is_valid_format") else (v.get("reason") or "INVALID"),
                state_code=v.get("state_code") or "", rto_code=v.get("rto_code") or "",
                registration_series=v.get("series") or "",
                registration_digits=v.get("registration_number") or "",
                char_confidences="|".join(f"{c['char']}:{c['p']:.4f}" for c in p["chars"]),
                substitutions="|".join(f"{s['pos']}:{s['from']}>{s['to']}:{s['p_to']:.3f}"
                                       for s in d.get("substitutions", [])),
                candidates="|".join(f"{c['text']}:{c['direct_support']}:{c['corrected_support']}:{c['best_avg']}"
                                    for c in d.get("candidates", [])),
                retry_count=p["retry_count"], ocr_attempts=p["ocr_attempts"],
                preprocessing_method=p["preprocessing"],
                quality_flags="|".join(p["quality"]["flags"]),
                review_reasons="|".join(d["reasons"]),
                manual_review_status=d["manual_review_status"], duplicate_of=duplicate_of,
                ml_probability="" if p.get("ml_probability") is None else f"{p['ml_probability']:.4f}",
                rule_status=p.get("rule_status", ""),
                ocr_engine=p.get("ocr_engine", self.ocr_engine), ocr_status=p.get("ocr_status", ""),
                ocr_time_ms="" if duplicate_of else f"{p.get('ocr_ms', 0.0):.1f}",
                processing_seconds=f"{time.time() - t_image:.3f}",
                image_quality=pp.quality_label(p["quality"]),
                resolver_probability="" if p.get("resolver_probability") is None
                else f"{p['resolver_probability']:.4f}",
                resolved_by=(d.get("resolved_by", "") + " (was: " + ", ".join(d.get("resolved_reasons", [])) + ")")
                if d.get("resolved_by") else "",
                calibrated_probability="" if p.get("calibrated_probability") is None
                else f"{p['calibrated_probability']:.4f}",
                hybrid_reasons="|".join(p.get("hybrid_reasons") or []),
                tta_result=p.get("tta_result", ""), ocr_model=self.ocr_model)
            x1, y1, x2, y2 = p["box"]
            if p["box"] != (0, 0, W, H):
                color = colors[p["status"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
                cv2.putText(annotated, p["text"] or "?", (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.8, W / 900), color, thick, cv2.LINE_AA)
        if not plates:
            dl.save_result(batch, name, thumb, duplicate_of=duplicate_of,   # NO PLATE row
                           ocr_engine=self.ocr_engine, ocr_status=pdec.DETECTION_FAILED,
                           processing_seconds=f"{time.time() - t_image:.3f}")
        if W > PREVIEW_W:
            annotated = cv2.resize(annotated, (PREVIEW_W, int(H * PREVIEW_W / W)),
                                   interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dl.ANNOT / thumb), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return plates
