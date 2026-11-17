"""OpenCV plate-crop quality assessment and enhancement variants.

Variants are chosen from the measured quality of the crop, not all applied blindly.
Every variant returns a 3-channel BGR image the same orientation as the input (PARSeq expects RGB)."""
import math

import cv2
import numpy as np

import anpr_config as cfg

# Quality thresholds (on the grayscale crop). Starting values; see evaluate.py.
DARK_MEAN = 80
LOW_CONTRAST_STD = 38
BLUR_LAPVAR = 120          # variance of Laplacian on a crop resized to 64 px high
NOISE_SIGMA = 9.0
LOW_RES_HEIGHT = 28
TILT_DEG = 3.0


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g


def estimate_tilt(img):
    """Median angle (degrees) of long near-horizontal edges, or 0.0 if none are reliable."""
    g = _gray(img)
    h, w = g.shape
    edges = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(20, w // 6),
                            minLineLength=max(20, int(w * 0.35)), maxLineGap=max(3, w // 40))
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if a > 90:
            a -= 180
        if a < -90:
            a += 180
        if abs(a) <= 30:
            angles.append(a)
    return float(np.median(angles)) if len(angles) >= 2 else 0.0


def assess(img):
    """Measure the crop. Returns a dict of numbers and a list of flags."""
    g = _gray(img)
    h, w = g.shape[:2]
    q = {"height": h, "width": w, "aspect": w / max(h, 1)}
    if h < 4 or w < 4:
        q.update(mean=0, std=0, sharpness=0, noise=0, tilt=0.0, flags=["UNUSABLE"])
        return q
    norm = cv2.resize(g, (max(8, int(w * 64 / h)), 64), interpolation=cv2.INTER_AREA
                      if h > 64 else cv2.INTER_CUBIC)
    q["mean"] = float(g.mean())
    q["std"] = float(g.std())
    q["sharpness"] = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    q["noise"] = float(np.median(np.abs(norm.astype(np.int16) - cv2.medianBlur(norm, 3))) * 1.4826)
    q["tilt"] = estimate_tilt(img)
    flags = []
    if h < cfg.MIN_CROP_HEIGHT or w < cfg.MIN_CROP_WIDTH or q["std"] < 6:
        flags.append("UNUSABLE")
    if q["mean"] < DARK_MEAN:
        flags.append("DARK")
    if q["std"] < LOW_CONTRAST_STD:
        flags.append("LOW_CONTRAST")
    if q["sharpness"] < BLUR_LAPVAR:
        flags.append("BLURRY")
    if q["noise"] > NOISE_SIGMA:
        flags.append("NOISY")
    if h < LOW_RES_HEIGHT:
        flags.append("LOW_RES")
    if abs(q["tilt"]) >= TILT_DEG:
        flags.append("TILTED")
    q["flags"] = flags
    return q


def quality_label(q):
    """One word for the dashboard: GOOD, DEGRADED (some flags) or UNUSABLE."""
    flags = q.get("flags") or []
    if "UNUSABLE" in flags:
        return "UNUSABLE"
    return "DEGRADED: " + ", ".join(flags) if flags else "GOOD"


# ----------------------------------------------------------------------------- variants
def grayscale(img):
    return _bgr(_gray(img))


def clahe(img, clip=2.0):
    lab = cv2.cvtColor(_bgr(img), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def low_light(img):
    """Gamma brighten (gamma picked from the mean) then gentle CLAHE."""
    g = _gray(img)
    mean = max(1.0, float(g.mean()))
    gamma = min(2.5, max(1.0, math.log(128 / 255) / math.log(mean / 255)))
    lut = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in range(256)], np.uint8)
    return clahe(cv2.LUT(_bgr(img), lut), clip=1.5)


def denoise(img):
    return cv2.bilateralFilter(_bgr(img), 7, 40, 7)


def sharpen(img, amount=0.8):
    """Unsharp mask; mild amount to avoid halo artifacts."""
    b = _bgr(img)
    blur = cv2.GaussianBlur(b, (0, 0), 1.2)
    return cv2.addWeighted(b, 1 + amount, blur, -amount, 0)


def upscale(img, f=2.0):
    return sharpen(cv2.resize(_bgr(img), None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), 0.5)


def adaptive_threshold(img):
    g = cv2.GaussianBlur(_gray(img), (3, 3), 0)
    block = max(11, (g.shape[0] // 2) | 1)
    return _bgr(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block, 7))


def otsu(img):
    g = cv2.GaussianBlur(_gray(img), (3, 3), 0)
    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if t.mean() < 110:          # dark text on light plate expected; invert light-on-dark
        t = 255 - t
    return _bgr(t)


def deskew(img, angle=None):
    angle = estimate_tilt(img) if angle is None else angle
    if abs(angle) < 1.0:
        return None
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(_bgr(img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _order(pts):
    s, d = pts.sum(1), np.diff(pts, axis=1)[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def perspective(img):
    """Warp to a rectangle when a clear 4-corner plate outline is found; else None."""
    g = _gray(img)
    h, w = g.shape
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120), None)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(c) < 0.35 * w * h:
            break
        quad = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        src = _order(quad[:, 0].astype(np.float32))
        tw = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
        th = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
        if tw < 20 or th < 8 or not (0.9 <= tw / th <= 8):
            continue
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
        return cv2.warpPerspective(_bgr(img), cv2.getPerspectiveTransform(src, dst), (tw, th),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return None


VARIANTS = {
    "low_light": low_light, "clahe": clahe, "denoise": denoise, "sharpen": sharpen,
    "denoise_sharpen": lambda im: sharpen(denoise(im)), "upscale": upscale,
    "deskew": deskew, "perspective": perspective, "grayscale": grayscale,
    "adaptive_threshold": adaptive_threshold, "otsu": otsu,
}


def plan_variants(quality):
    """Ordered retry plan from the quality flags. Round 1 = targeted fixes, round 2 = generic
    fall-backs (binarisation last: PARSeq is trained on natural images and often reads binary
    images worse)."""
    f = set(quality["flags"])
    r1 = []
    if "TILTED" in f:
        r1 += ["perspective", "deskew"]
    if "DARK" in f:
        r1.append("low_light")
    if "LOW_CONTRAST" in f and "DARK" not in f:
        r1.append("clahe")
    if "NOISY" in f:
        r1.append("denoise_sharpen" if "BLURRY" in f else "denoise")
    elif "BLURRY" in f:
        r1.append("sharpen")
    if "LOW_RES" in f:
        r1.append("upscale")
    r2 = [v for v in ("clahe", "grayscale", "perspective", "sharpen", "adaptive_threshold", "otsu")
          if v not in r1]
    return [r1, r2]


def make_variants(img, names):
    """Apply named variants; skips ones that do not apply (e.g. no outline for perspective)."""
    out = []
    for n in names:
        try:
            v = VARIANTS[n](img)
        except cv2.error:
            v = None
        if v is not None and v.size:
            out.append((n, v))
    return out
