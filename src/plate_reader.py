"""
Vehicle number plate reader: YOLO11 (your best.pt) detects the plate,
PARSeq reads the characters, and the full vehicle number is the output.

Usage:
    python src/plate_reader.py --source car.jpg
    python src/plate_reader.py --source folder_of_images/
    python src/plate_reader.py --source video.mp4
    python src/plate_reader.py --source 0                  # webcam

    python src/plate_reader.py --source car.jpg --weights best.pt --parseq parseq --device cpu
"""
import argparse
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------------------------------
# PARSeq
# --------------------------------------------------------------------------
class PARSeqReader:
    """Thin wrapper around the official PARSeq (baudm/parseq) torch.hub model.

    variant: 'parseq' (default, best accuracy), 'parseq_tiny' (fastest),
             'parseq_patch16_224'.
    The weights download once from GitHub and are cached by torch.hub.
    """

    def __init__(self, variant="parseq", device="cpu"):
        import torch
        from PIL import Image  # noqa: F401
        from torchvision import transforms as T

        self.torch = torch
        self.device = device
        local = Path(__file__).resolve().parents[1] / "parseq_repo"
        if (local / "hubconf.py").exists():   # use the bundled copy of baudm/parseq
            model = torch.hub.load(str(local), variant, source="local", pretrained=True)
        else:
            model = torch.hub.load("baudm/parseq", variant, pretrained=True, trust_repo=True)
        self.model = model.eval().to(device)

        h, w = tuple(self.model.hparams.img_size)  # (32, 128)
        self.tf = T.Compose([
            T.Resize((h, w), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

        # Map each model output class to an upper-case A-Z0-9 group, so 'o' and 'O' count as
        # the same plate character. Output class i corresponds to tokenizer._itos[i].
        itos = self.model.tokenizer._itos
        n_out = self.model.head.out_features if hasattr(self.model, "head") else len(itos) - 2
        self._groups = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        M = torch.zeros(n_out, len(self._groups))
        for i in range(n_out):
            t = itos[i].upper()
            if len(t) == 1 and t in self._groups:
                M[i, self._groups.index(t)] = 1.0
        self._group_mat = M.to(device)
        self._eos = self.model.tokenizer.eos_id
        self._itos = itos
        from collections import OrderedDict
        self._cache = OrderedDict()
        self.cache_hits = 0

    def _key(self, im):
        import hashlib
        return hashlib.blake2b(im.tobytes() + str(im.shape).encode(), digest_size=16).digest()

    def read_chars(self, bgr_images, cache_size=4096):
        """List of BGR crops -> list of dicts with real per-character probabilities:
            {"raw": str (decoded text, original case and symbols),
             "chars": [{"char": "A", "p": 0.97, "alts": [("A", .97), ("4", .02), ...]}, ...],
             "avg": geometric mean of chars' p, "weakest": min p}
        'p' is PARSeq's softmax probability at that position, summed over upper/lower case.
        Positions whose best token is a symbol (space, '-', '.') are dropped, as normalisation
        drops them. Identical crops are read once (cache)."""
        from PIL import Image

        results = [None] * len(bgr_images)
        todo = []
        for i, im in enumerate(bgr_images):
            k = self._key(im)
            if k in self._cache:
                self._cache.move_to_end(k)
                results[i] = self._cache[k]
                self.cache_hits += 1
            else:
                todo.append((i, k, im))
        if todo:
            batch = self.torch.stack([
                self.tf(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for _, _, im in todo
            ]).to(self.device)
            with self.torch.inference_mode():
                probs = self.model(batch).softmax(-1)
                grouped = (probs @ self._group_mat).cpu()
                top_ids = probs.argmax(-1).cpu()
            for (i, k, _), g, ids in zip(todo, grouped, top_ids):
                raw, chars = "", []
                for pos in range(len(ids)):
                    tid = int(ids[pos])
                    if tid == self._eos:
                        break
                    raw += self._itos[tid]
                    if g[pos].sum() < 0.5:          # the model is mostly predicting a symbol here
                        continue
                    vals, idx = g[pos].topk(4)
                    alts = [(self._groups[int(j)], float(v)) for v, j in zip(vals, idx)]
                    chars.append({"char": alts[0][0], "p": alts[0][1], "alts": alts})
                ps = [c["p"] for c in chars]
                res = {"raw": raw, "chars": chars,
                       "avg": math.exp(sum(math.log(max(p, 1e-9)) for p in ps) / len(ps)) if ps else 0.0,
                       "weakest": min(ps) if ps else 0.0}
                results[i] = res
                self._cache[k] = res
                if len(self._cache) > cache_size:
                    self._cache.popitem(last=False)
        return results

    def read_plate_chars(self, crops, two_line="auto"):
        """Read many plate crops at once (one batched PARSeq call). Two-line plates
        (aspect < TWO_LINE_ASPECT) are read as top + bottom halves and joined."""
        jobs, layout = [], []
        for crop in crops:
            tl = is_two_line(crop) if two_line == "auto" else two_line
            if tl:
                h = crop.shape[0]
                layout.append((len(jobs), 2))
                jobs += [crop[: h // 2 + 2], crop[h // 2 - 2:]]
            else:
                layout.append((len(jobs), 1))
                jobs.append(crop)
        reads = self.read_chars(jobs)
        out = []
        for start, n in layout:
            if n == 1:
                out.append(dict(reads[start], two_line=False))
                continue
            a, b = reads[start], reads[start + 1]
            chars = a["chars"] + b["chars"]
            ps = [c["p"] for c in chars]
            out.append({"raw": a["raw"] + " / " + b["raw"], "chars": chars, "two_line": True,
                        "avg": math.exp(sum(math.log(max(p, 1e-9)) for p in ps) / len(ps)) if ps else 0.0,
                        "weakest": min(ps) if ps else 0.0})
        return out

    def read(self, bgr_images):
        """List of BGR crops -> list of (text, confidence). Kept for compatibility."""
        return [(t, c) for t, c, _ in self.read_detailed(bgr_images)]

    def read_detailed(self, bgr_images):
        """List of BGR crops -> list of (text, avg_char_conf, weakest_char_conf).
        avg_char_conf is the geometric mean of the per-character probabilities, so it
        does not shrink just because the plate has many characters."""
        from PIL import Image

        if not bgr_images:
            return []
        batch = self.torch.stack([
            self.tf(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
            for im in bgr_images
        ]).to(self.device)
        with self.torch.inference_mode():
            probs = self.model(batch).softmax(-1)
        texts, token_probs = self.model.tokenizer.decode(probs)
        out = []
        for t, p in zip(texts, token_probs):
            if len(p):
                avg = float(p.prod()) ** (1.0 / len(p))
                weakest = float(p.min())
            else:
                avg = weakest = 0.0
            out.append((t, avg, weakest))
        return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clean_plate_text(s):
    """Uppercase, keep only A-Z / 0-9 (drops spaces, dashes, dots)."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


# Indian format, e.g. TN09AB1234 / MH12DE1433 / DL1CAB1234 / 22BH1234AA
_STATES = ("AN|AP|AR|AS|BR|CH|CG|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|"
           "MZ|NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UK|UA|UP|WB")
_CORE = rf"(?:(?:{_STATES})\d{{1,2}}[A-Z]{{0,3}}\d{{4}}|\d{{2}}BH\d{{4}}[A-Z]{{1,2}})"
_IN_PLATE = re.compile(rf"^{_CORE}$")
_IN_PLATE_ANY = re.compile(_CORE)


def looks_like_indian_plate(s):
    return bool(_IN_PLATE.match(s))


def extract_plate(s):
    """Return the valid plate found inside s (e.g. 'INDRJ14CV0002' -> 'RJ14CV0002'),
    or '' if there is none. Picks the longest match."""
    found = [m.group(0) for m in _IN_PLATE_ANY.finditer(s)]
    return max(found, key=len) if found else ""


def pad_box(x1, y1, x2, y2, w, h, pad=0.06):
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def is_two_line(crop):
    """Two-wheeler / square plates have two rows of characters.
    A normal car plate is ~4:1 wide; anything below ~2.2:1 is treated as two-line."""
    h, w = crop.shape[:2]
    return (w / max(h, 1)) < 2.2


def read_plate(reader, crop, two_line="auto"):
    """Read one plate crop; returns (full_text, confidence)."""
    if two_line == "auto":
        two_line = is_two_line(crop)
    if two_line:
        h = crop.shape[0]
        top, bottom = crop[: h // 2 + 2], crop[h // 2 - 2:]
        (t1, c1, m1), (t2, c2, m2) = reader.read_detailed([top, bottom])
        return clean_plate_text(t1 + t2), min(c1, c2), min(m1, m2)
    (t, c, m), = reader.read_detailed([crop])
    return clean_plate_text(t), c, m


def process_frame(frame, detector, reader, conf, two_line, draw=True):
    """Detect plates in a frame, read each one. Returns list of dicts."""
    H, W = frame.shape[:2]
    results = detector.predict(frame, conf=conf, verbose=False)[0]
    plates = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, W, H)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        text, ocr_conf, _weakest = read_plate(reader, crop, two_line)
        plates.append({
            "text": text,
            "ocr_conf": ocr_conf,
            "det_conf": float(box.conf[0]),
            "box": (x1, y1, x2, y2),
            "valid_format": looks_like_indian_plate(text),
        })
        if draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = text or "?"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            ty = max(th + 8, y1)
            cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 8, ty), (0, 220, 0), -1)
            cv2.putText(frame, label, (x1 + 4, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return plates


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image, folder, video, or 0 for webcam")
    ap.add_argument("--weights", default="best.pt", help="your YOLO plate detector")
    ap.add_argument("--parseq", default="parseq", choices=["parseq", "parseq_tiny", "parseq_patch16_224"])
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--device", default="cpu", help="cpu / cuda / 0")
    ap.add_argument("--two-line", default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--out", default="output", help="folder for annotated results")
    ap.add_argument("--show", action="store_true", help="open a preview window")
    args = ap.parse_args()

    from ultralytics import YOLO
    detector = YOLO(args.weights)
    torch_device = "cuda" if args.device in ("0", "cuda") else "cpu"
    reader = PARSeqReader(args.parseq, torch_device)
    two_line = {"auto": "auto", "yes": True, "no": False}[args.two_line]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = args.source

    # ---- images / folder ----
    p = Path(src)
    if src.isdigit() is False and (p.is_dir() or p.suffix.lower() in IMG_EXT):
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
        for f in files:
            frame = cv2.imread(str(f))
            if frame is None:
                print(f"{f.name}: could not read image")
                continue
            plates = process_frame(frame, detector, reader, args.conf, two_line)
            if not plates:
                print(f"{f.name}: no plate detected")
            for pl in plates:
                flag = "" if pl["valid_format"] else "  (unusual format - check)"
                print(f"{f.name}: {pl['text']}  [det {pl['det_conf']:.2f}, ocr {pl['ocr_conf']:.2f}]{flag}")
            cv2.imwrite(str(out_dir / f.name), frame)
            if args.show:
                cv2.imshow("plates", frame)
                cv2.waitKey(0)
        return

    # ---- video / webcam ----
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out_dir / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    seen = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for pl in process_frame(frame, detector, reader, args.conf, two_line):
            # keep the highest-confidence reading of each distinct plate
            if pl["text"] and pl["ocr_conf"] > seen.get(pl["text"], 0):
                if pl["text"] not in seen:
                    print(f"New plate: {pl['text']}")
                seen[pl["text"]] = pl["ocr_conf"]
        writer.write(frame)
        if args.show:
            cv2.imshow("plates", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    print("Plates seen:", ", ".join(seen) or "none")


if __name__ == "__main__":
    main()