"""Confidence calibration for Awiros readings (hybrid accuracy work, 2026-11-11).

Awiros returns CTC softmax probabilities (N x T x 64). log(p) is the logit up to a per-frame
constant, so Temperature Scaling is applied as softmax(log(p) / T) - genuine temperature scaling
of the model's own output, fitted on the separate labelled CALIBRATION split (never on test).

Two calibrated numbers:
  * character level  - p(char) after temperature T (fitted by NLL on aligned characters);
  * plate level      - P(whole reading is correct): Platt scaling (logistic regression) of the
                       tempered log-probability features, fitted on the calibration split.
Calibration changes confidence estimates only. It does not change any recognised text.

Saved to models/calibration/awiros_calibration.json; the original Awiros weights are untouched."""
import json
import math
from pathlib import Path

import numpy as np

import anpr_config as cfg

CAL_PATH = Path(__file__).resolve().parents[1] / "models" / "calibration" / "awiros_calibration.json"
_GROUPS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class Decoder:
    """Greedy CTC decoding identical to AwirosReader._decode, with an optional temperature.
    Needs only the dictionary file (no model)."""

    def __init__(self, dict_path=None):
        chars = [ln.rstrip("\r\n") for ln in open(dict_path or cfg.AWIROS_DICT, encoding="utf-8")]
        self.ctc_chars = ["<blank>"] + [c for c in chars if c] + [" "]
        G = np.zeros((len(self.ctc_chars), len(_GROUPS)), np.float32)
        for i, ch in enumerate(self.ctc_chars):
            if len(ch) == 1 and ch.upper() in _GROUPS:
                G[i, _GROUPS.index(ch.upper())] = 1.0
        self.G = G

    @staticmethod
    def temper(probs, T):
        if T == 1.0:
            return probs.astype(np.float32)
        z = np.log(np.clip(probs.astype(np.float32), 1e-12, 1.0)) / T
        z -= z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    def decode(self, probs, T=1.0):
        """-> {"raw", "text" (A-Z0-9 upper), "chars": [{"char","p","alts"}], "avg", "weakest"}"""
        probs = self.temper(probs, T)
        ids = probs.argmax(-1)
        raw, chars, t, n = "", [], 0, len(ids)
        while t < n:
            c, e = int(ids[t]), t
            while e + 1 < n and int(ids[e + 1]) == c:
                e += 1
            if c != 0:
                raw += self.ctc_chars[c]
                if self.G[c].any():
                    f = t + int(np.argmax(probs[t:e + 1, c]))
                    g = probs[f] @ self.G
                    gi = _GROUPS.index(self.ctc_chars[c].upper())
                    order = np.argsort(-g)[:4]
                    chars.append({"char": _GROUPS[gi], "p": float(g[gi]),
                                  "alts": [(_GROUPS[j], float(g[j])) for j in order]})
            t = e + 1
        ps = [c["p"] for c in chars]
        return {"raw": raw, "text": "".join(c["char"] for c in chars), "chars": chars,
                "avg": math.exp(sum(math.log(max(p, 1e-9)) for p in ps) / len(ps)) if ps else 0.0,
                "weakest": min(ps) if ps else 0.0}


# ----------------------------------------------------------------------------- metrics
def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if not len(conf):
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def brier(conf, correct):
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def nll(conf, correct):
    c = np.clip(np.asarray(conf, float), 1e-6, 1 - 1e-6)
    y = np.asarray(correct, float)
    return float(-np.mean(y * np.log(c) + (1 - y) * np.log(1 - c)))


def reliability(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                        "mean_conf": round(float(conf[m].mean()), 4),
                        "accuracy": round(float(correct[m].mean()), 4)})
    return out


# ----------------------------------------------------------------------------- fitting
def char_pairs(decoder, probs_list, labels, T):
    """(p, correct) for every character of readings whose length equals the label's."""
    ps, ys = [], []
    for probs, gt in zip(probs_list, labels):
        d = decoder.decode(probs, T)
        if len(d["text"]) != len(gt):
            continue
        for c, g in zip(d["chars"], gt):
            ps.append(c["p"]); ys.append(c["char"] == g)
    return np.array(ps), np.array(ys, float)


def fit_temperature(decoder, probs_list, labels, grid=None):
    """Temperature minimising character-level NLL on the calibration split."""
    grid = grid if grid is not None else np.round(np.concatenate([np.arange(0.5, 1.0, 0.05),
                                                                  np.arange(1.0, 4.01, 0.1)]), 3)
    best = (float("inf"), 1.0)
    for T in grid:
        p, y = char_pairs(decoder, probs_list, labels, float(T))
        if len(p):
            best = min(best, (nll(p, y), float(T)))
    return best[1]


def plate_features(d):
    ps = [c["p"] for c in d["chars"]] or [1e-9]
    lp = np.log(np.clip(ps, 1e-9, 1))
    return [float(lp.sum()), float(lp.min()), float(np.sort(lp)[1] if len(lp) > 1 else lp.min())]


class PlateCalibrator:
    """P(reading exactly correct) = sigmoid(w . features + b) on tempered char probabilities."""

    def __init__(self, T=1.0, w=None, b=0.0):
        self.T, self.w, self.b = float(T), list(w or [0.0, 1.0, 0.0]), float(b)

    def prob(self, d):
        z = float(np.dot(self.w, plate_features(d)) + self.b)
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))

    @classmethod
    def fit(cls, decoded, correct, T, C=1.0):
        from sklearn.linear_model import LogisticRegression
        X = np.array([plate_features(d) for d in decoded])
        y = np.asarray(correct, int)
        if len(set(y)) < 2:
            return cls(T)
        m = LogisticRegression(C=C).fit(X, y)
        return cls(T, m.coef_[0].tolist(), float(m.intercept_[0]))

    def save(self, path=CAL_PATH, **info):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"temperature": self.T, "plate_w": self.w, "plate_b": self.b,
                                    **info}, indent=2))

    @classmethod
    def load(cls, path=CAL_PATH):
        if not Path(path).is_file():
            return None
        d = json.loads(Path(path).read_text())
        return cls(d["temperature"], d["plate_w"], d["plate_b"])
