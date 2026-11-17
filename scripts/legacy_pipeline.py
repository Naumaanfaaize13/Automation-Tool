"""The previous (v1) decision logic, kept unchanged ONLY so evaluate.py can measure before/after.
Not used by the server."""
import dashboard_log as dl
from pipeline import merge_boxes
from plate_reader import extract_plate, read_plate


def _best_reading(reader, frame, union_box, members):
    H, W = frame.shape[:2]
    cands, confident = [], False
    for k, b in enumerate([union_box] + members):
        if confident:
            break
        bw, bh = b[2] - b[0], b[3] - b[1]
        for px in (0.06, 0.25, 0.5):
            x1 = max(0, int(b[0] - bw * px)); x2 = min(W, int(b[2] + bw * px))
            y1 = max(0, int(b[1] - bh * 0.12)); y2 = min(H, int(b[3] + bh * 0.12))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            raw, c, weakest = read_plate(reader, crop)
            plate = extract_plate(raw)
            cands.append({"raw": raw, "text": plate, "conf": c, "weakest": weakest, "valid": bool(plate)})
            if k == 0 and px == 0.06 and plate and c >= 0.90 and weakest >= 0.50:
                confident = True
                break
    if not cands:
        return None
    valid = [c for c in cands if c["valid"]]
    if valid:
        votes = {}
        for c in valid:
            votes.setdefault(c["text"], []).append(c)
        text, group = max(votes.items(), key=lambda kv: (len(kv[1]), max(g["conf"] for g in kv[1])))
        best = max(group, key=lambda g: g["conf"])
        return {"text": text, "conf": best["conf"], "weakest": best["weakest"], "valid": True}
    best = max(cands, key=lambda c: c["conf"])
    return {"text": best["raw"], "conf": best["conf"], "weakest": best["weakest"], "valid": False}


def analyze(pipeline, frame, det_conf=0.25, accept_thr=0.85, crop_mode=False):
    """v1 behaviour: returns [{status, text, avg, weakest}] (SUCCESS/REVIEW only)."""
    H, W = frame.shape[:2]
    if crop_mode:
        groups = [([0, 0, W, H, 1.0], [[0, 0, W, H, 1.0]])]
    else:
        res = pipeline.detector.predict(frame, conf=det_conf, iou=0.4, verbose=False,
                                        device=pipeline.device)[0].boxes
        groups = merge_boxes([list(map(float, b.xyxy[0].tolist())) + [float(b.conf[0])] for b in res])
    out = []
    for u, m in groups:
        r = _best_reading(pipeline.reader, frame, u, m)
        if r is None:
            continue
        out.append({"status": dl.decide_status(r["valid"], r["conf"], accept_thr, r["weakest"]),
                    "text": r["text"], "avg": r["conf"], "weakest": r["weakest"],
                    "retry_count": 0, "base_status": None})
    return out
