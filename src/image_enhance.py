"""Adaptive OpenCV enhancement for plate crops (hybrid accuracy work, 2026-11-11).

Builds on plate_preprocess.py (whose functions and quality flags stay unchanged) and adds the
variants the hybrid plan asks for: CLAHE + bilateral filtering, Gaussian denoising and controlled
resizing. Every function returns a NEW BGR image (the original crop is never modified) or None
when the transformation does not apply (e.g. no tilt to correct).

`all_variants` is for benchmarking (bench/cache_reads.py); production uses `plan(quality)`,
whose choices are set from measured accuracy on the validation split, not assumed."""
import cv2
import numpy as np

import plate_preprocess as pp


def clahe_bilateral(img, clip=2.0):
    """CLAHE on the L channel, then an edge-preserving bilateral filter."""
    return cv2.bilateralFilter(pp.clahe(img, clip), 5, 30, 5)


def gaussian_denoise(img):
    return cv2.GaussianBlur(pp._bgr(img), (3, 3), 0)


def controlled_resize(img, target_h=96):
    """Enlarge small crops (keeps aspect ratio, cubic); None when already tall enough."""
    h, w = img.shape[:2]
    if h >= target_h:
        return None
    f = target_h / float(h)
    return cv2.resize(pp._bgr(img), (int(round(w * f)), target_h), interpolation=cv2.INTER_CUBIC)


def deskew_perspective(img):
    """Perspective rectification when an outline is found, else deskew, else None."""
    out = pp.perspective(img)
    return out if out is not None else pp.deskew(img)


VARIANTS = {
    "clahe": pp.clahe,
    "bilateral": pp.denoise,
    "clahe_bilateral": clahe_bilateral,
    "gaussian": gaussian_denoise,
    "sharpen": pp.sharpen,
    "low_light": pp.low_light,
    "deskew": pp.deskew,
    "perspective": pp.perspective,
    "upscale": pp.upscale,
    "resize": controlled_resize,
    "grayscale": pp.grayscale,
    "adaptive_threshold": pp.adaptive_threshold,
}


def apply(img, name):
    try:
        out = VARIANTS[name](img)
    except cv2.error:
        return None
    return out if out is not None and out.size else None


def all_variants(img):
    """Every variant that applies to this crop: {name: image}."""
    out = {}
    for n in VARIANTS:
        v = apply(img, n)
        if v is not None:
            out[n] = v
    return out


def quality_class(q):
    """GOOD / ACCEPTABLE / POOR / UNREADABLE from plate_preprocess.assess() flags.
    UNREADABLE = unusable crop; POOR = two or more problems, or blur+dark; ACCEPTABLE = one."""
    f = set(q.get("flags") or [])
    if "UNUSABLE" in f:
        return "UNREADABLE"
    if len(f) >= 2:
        return "POOR"
    return "ACCEPTABLE" if f else "GOOD"


def is_identity(a, b):
    return a is b or (a.shape == b.shape and np.array_equal(a, b))
