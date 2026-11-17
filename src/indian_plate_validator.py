"""Indian registration number normalisation, format validation and position-aware candidates.

Format validation only says the text is SHAPED like an Indian registration number.
It does not prove the number exists: that needs an official lookup (e.g. VAHAN), which is
kept out of this module on purpose.
"""
import math
import re
from datetime import date

import anpr_config as cfg

# State / UT codes in current use (MoRTH). Telangana issues both TS and TG (TG since 2024).
STATE_CODES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
    "DD": "Dadra and Nagar Haveli and Daman and Diu", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
    "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand", "JK": "Jammu and Kashmir",
    "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra",
    "ML": "Meghalaya", "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland",
    "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim",
    "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana", "TG": "Telangana",
    "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}
# Codes no longer issued but still on older vehicles.
LEGACY_STATE_CODES = {"OR": "Odisha (before 2012)", "UA": "Uttarakhand (before 2007)",
                      "DN": "Dadra and Nagar Haveli (before 2020)"}

_STANDARD = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$")
_BH = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")
_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize(raw):
    """Uppercase and keep only A-Z 0-9. Never changes a character into another one."""
    return _ALNUM.sub("", (raw or "").upper())


def _state_name(code):
    if code in STATE_CODES:
        return STATE_CODES[code]
    if cfg.ALLOW_LEGACY_STATE_CODES and code in LEGACY_STATE_CODES:
        return LEGACY_STATE_CODES[code]
    return None


def validate(raw):
    """Validate one complete string. Returns a dict; is_valid_format True only for a full
    registration number. 'warnings' lists things that keep a valid read away from SUCCESS."""
    norm = normalize(raw)
    out = {"raw_text": raw, "normalized_text": norm, "is_valid_format": False,
           "format": None, "state_code": None, "state_name": None, "rto_code": None,
           "series": None, "registration_number": None, "reason": None, "warnings": []}
    if not norm:
        out["reason"] = "EMPTY_TEXT"
        return out
    if len(norm) < 6:
        out["reason"] = "TOO_SHORT"
        return out
    if len(norm) > 11:
        out["reason"] = "TOO_LONG"
        return out

    m = _BH.match(norm)
    if m:
        yy, num, letters = m.groups()
        out.update(format="BH", state_code="BH", state_name="Bharat series", rto_code=yy,
                   series=letters, registration_number=num)
        if not (21 <= int(yy) <= date.today().year % 100):
            out["reason"] = "INVALID_BH_YEAR"
            return out
        if num == "0000":
            out["reason"] = "INVALID_NUMBER"
            return out
        if set(letters) & cfg.SERIES_WARN_LETTERS:
            out["warnings"].append("SERIES_HAS_I_OR_O")
        out["is_valid_format"] = True
        return out

    m = _STANDARD.match(norm)
    if not m:
        out["reason"] = "INVALID_REGISTRATION_FORMAT"
        return out
    state, rto, series, num = m.groups()
    out.update(format="STANDARD", state_code=state, rto_code=rto, series=series,
               registration_number=num)
    name = _state_name(state)
    if name is None:
        out["reason"] = "UNKNOWN_STATE_CODE"
        return out
    out["state_name"] = name
    lo, hi = cfg.RTO_DIGITS.get(state, (2, 2))
    if not (lo <= len(rto) <= hi) or int(rto) == 0:
        out["reason"] = "INVALID_RTO_CODE"
        return out
    if num == "0000":
        out["reason"] = "INVALID_NUMBER"
        return out
    if state in LEGACY_STATE_CODES:
        out["warnings"].append("LEGACY_STATE_CODE")
    if set(series) & cfg.SERIES_WARN_LETTERS:
        out["warnings"].append("SERIES_HAS_I_OR_O")
    if not series and cfg.WARN_EMPTY_SERIES:
        out["warnings"].append("NO_SERIES_LETTERS")
    out["is_valid_format"] = True
    return out


def find_in_text(raw):
    """Validate, and if the whole string is not a plate, look for a plate inside it
    (e.g. 'INDRJ14CV0002' -> 'RJ14CV0002'). Text taken out of a longer string gets the warning
    EXTRACTED_FROM_LONGER_TEXT, except a leading 'IND' (the plate's own marking)."""
    v = validate(raw)
    if v["is_valid_format"]:
        return v
    norm = v["normalized_text"]
    best = None
    for i in range(len(norm)):
        for j in range(len(norm), i + 5, -1):
            if i == 0 and j == len(norm):
                continue
            sub = validate(norm[i:j])
            if sub["is_valid_format"] and (best is None or j - i > len(best[0]["normalized_text"])):
                best = (sub, i, j)
    if best is None:
        return v
    sub, i, j = best
    sub["raw_text"] = raw
    sub["extracted_from"] = norm
    if not (norm[:i] in ("", "IND") and j == len(norm)):
        sub["warnings"].append("EXTRACTED_FROM_LONGER_TEXT")
    return sub


def looks_like_plate_text(norm):
    """Plate-like but not valid (partial / damaged): has a run of 3+ digits and is 4-12 chars.
    Arbitrary words (CHEVROLETELIBERTY, VEHICLE) do not."""
    return 4 <= len(norm) <= 12 and re.search(r"\d{3,}", norm) is not None


def candidates(chars, max_out=5):
    """Position-aware alternatives that use the model's own probabilities.

    chars: list of {"char": str, "p": float, "alts": [(char, p), ...]} (alts sorted, include
    the chosen char). A character is only swapped for an alternative the model itself gave
    probability >= ALT_MIN_PROB (>= ALT_MIN_PROB_NON_CONFUSABLE unless the pair is a known
    look-alike), at most MAX_SUBSTITUTIONS swaps. Characters are never added or removed.
    Returns valid candidates (best first): {text, score, substitutions, validation}."""
    options = []
    for c in chars:
        opts = [(c["char"], c["p"], False)]
        for a, p in c.get("alts", []):
            if a == c["char"]:
                continue
            confusable = frozenset((a, c["char"])) in cfg.CONFUSABLE
            if p >= (cfg.ALT_MIN_PROB if confusable else cfg.ALT_MIN_PROB_NON_CONFUSABLE):
                opts.append((a, p, True))
        options.append(opts)

    beams = [("", 0.0, [])]
    for pos, opts in enumerate(options):
        nxt = []
        for text, lp, subs in beams:
            for ch, p, is_sub in opts:
                if is_sub and len(subs) >= cfg.MAX_SUBSTITUTIONS:
                    continue
                s = subs + [{"pos": pos, "from": chars[pos]["char"], "to": ch,
                             "p_from": chars[pos]["p"], "p_to": p}] if is_sub else subs
                nxt.append((text + ch, lp + math.log(max(p, 1e-9)), s))
        beams = sorted(nxt, key=lambda b: -b[1])[:256]

    out, seen = [], set()
    n = max(1, len(chars))
    for text, lp, subs in beams:
        v = validate(text)
        if v["is_valid_format"] and text not in seen:
            seen.add(text)
            out.append({"text": text, "score": math.exp(lp / n), "substitutions": subs,
                        "validation": v})
        if len(out) >= max_out:
            break
    return out
