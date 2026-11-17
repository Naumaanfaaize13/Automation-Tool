"""Combine all OCR attempts for one plate into one decision: SUCCESS / REVIEW / REJECT.

Evidence used: per-character probabilities, format validity, agreement between attempts,
detection confidence, plate geometry and crop quality. A reading produced by swapping
characters (position-aware candidates) is never SUCCESS on its own: it is shown as a
suggestion for manual review."""
import anpr_config as cfg
import indian_plate_validator as ipv

SUCCESS, REVIEW, REJECT = "SUCCESS", "REVIEW", "REJECT"
MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
# detailed OCR status, stored next to the (unchanged) SUCCESS / REVIEW / REJECT / NO PLATE status
LOW_CONFIDENCE, INVALID_FORMAT = "LOW_CONFIDENCE", "INVALID_FORMAT"
DETECTION_FAILED, OCR_FAILED = "DETECTION_FAILED", "OCR_FAILED"
OCR_STATUSES = (SUCCESS, LOW_CONFIDENCE, INVALID_FORMAT, MANUAL_REVIEW_REQUIRED,
                DETECTION_FAILED, OCR_FAILED)


def detailed_status(plate):
    """Map a plate dict (after the ML verifier) to one detailed status. Keeps OCR confidence
    (LOW_CONFIDENCE) apart from registration format (INVALID_FORMAT).
      OCR_FAILED      the model raised an error or returned no text at all
      INVALID_FORMAT  text read, but not shaped like an Indian registration (REJECT, or REVIEW
                      of partial text)
      LOW_CONFIDENCE  valid format; the only objections are confidence ones
      MANUAL_REVIEW_REQUIRED  any other objection (conflicts, character swaps, warnings, ...)
    DETECTION_FAILED is used for images where YOLO found no plate (no plate dict exists)."""
    d = plate["decision"]
    if not ipv.normalize(plate.get("raw") or "") and not d.get("text"):
        return OCR_FAILED
    if d["status"] == SUCCESS:
        return SUCCESS
    if d["status"] == REJECT or not (d.get("validation") or {}).get("is_valid_format"):
        return INVALID_FORMAT
    reasons = set(d.get("reasons") or [])
    if reasons and reasons <= cfg.LOW_CONFIDENCE_REASONS:
        return LOW_CONFIDENCE
    return MANUAL_REVIEW_REQUIRED


def decide(attempts, det_conf=None, quality=None, box_aspect=None, base_attempts=None,
           accept_avg=cfg.ACCEPT_AVG_CONF, accept_weakest=cfg.ACCEPT_WEAKEST_CHAR):
    """attempts: list of {"variant", "raw", "chars", "avg", "weakest"} (one per PARSeq read).
    base_attempts: how many of them were un-enhanced crops (for retry statistics).
    Returns a dict describing the decision; 'final_text' is only set for SUCCESS."""
    quality = quality or {"flags": []}
    by_text = {}
    best_raw = max(attempts, key=lambda a: a["avg"]) if attempts else None
    for a in attempts:
        v = ipv.find_in_text(a["raw"])
        a["validation"] = v
        a["normalized"] = ipv.normalize(a["raw"])
        if v["is_valid_format"]:
            e = by_text.setdefault(v["normalized_text"], {"direct": [], "corrected": []})
            e["direct"].append(a)
    # position-aware alternatives, only from attempts whose own reading is not valid or is weak
    for a in attempts:
        if a["validation"]["is_valid_format"] and a["weakest"] >= accept_weakest:
            continue
        for c in ipv.candidates(a["chars"], max_out=3):
            if not c["substitutions"]:
                continue
            e = by_text.setdefault(c["text"], {"direct": [], "corrected": []})
            e["corrected"].append({"attempt": a, "cand": c})

    reasons, result = [], {"attempts": len(attempts), "quality_flags": list(quality["flags"])}
    ranked = sorted(by_text.items(), key=lambda kv: (
        len(kv[1]["direct"]), max([d["avg"] for d in kv[1]["direct"]] or [0]),
        len(kv[1]["corrected"]), max([c["cand"]["score"] for c in kv[1]["corrected"]] or [0])),
        reverse=True)
    result["candidates"] = [
        {"text": t, "direct_support": len(e["direct"]), "corrected_support": len(e["corrected"]),
         "best_avg": round(max([d["avg"] for d in e["direct"]] or
                               [c["cand"]["score"] for c in e["corrected"]]), 4)}
        for t, e in ranked[:5]]

    if not ranked:
        # nothing shaped like a registration number
        a = best_raw
        norm = ipv.normalize(a["raw"]) if a else ""
        result.update(status=REJECT, text=norm, final_text="", chosen=a,
                      validation=ipv.validate(a["raw"] if a else ""), corrected=False,
                      avg=a["avg"] if a else 0.0, weakest=a["weakest"] if a else 0.0,
                      substitutions=[])
        if a is None or "UNUSABLE" in quality["flags"]:
            reasons.append("UNUSABLE_CROP")
        elif ipv.looks_like_plate_text(norm):
            # partial / damaged plate: a person may still read it
            result["status"] = REVIEW
            reasons.append("PARTIAL_OR_INVALID_PLATE_TEXT")
        else:
            reasons.append("NOT_A_REGISTRATION_NUMBER")
        result["reasons"] = reasons
        result["manual_review_status"] = MANUAL_REVIEW_REQUIRED if result["status"] == REVIEW else ""
        return result

    text, e = ranked[0]
    if e["direct"]:
        chosen = max(e["direct"], key=lambda d: (not d["validation"]["warnings"],
                                                  d["avg"], d["weakest"]))
        avg, weakest, corrected, subs = chosen["avg"], chosen["weakest"], False, []
        v = chosen["validation"]
    else:
        best = max(e["corrected"], key=lambda c: c["cand"]["score"])
        chosen, cand = best["attempt"], best["cand"]
        avg = cand["score"]
        weakest = min([s["p_to"] for s in cand["substitutions"]] +
                      [c["p"] for i, c in enumerate(chosen["chars"])
                       if i not in {s["pos"] for s in cand["substitutions"]}])
        corrected, subs, v = True, cand["substitutions"], cand["validation"]
        reasons.append("CHARACTERS_SUBSTITUTED_FROM_MODEL_ALTERNATIVES")

    # ---- checks that keep a valid-format reading out of SUCCESS ----
    if avg < accept_avg:
        reasons.append("LOW_AVERAGE_CONFIDENCE")
    if weakest < accept_weakest:
        reasons.append("UNCERTAIN_CHARACTER")
    reasons += v["warnings"]
    for t2, e2 in (ranked[1:] if cfg.ATTEMPT_CONFLICT_CHECK else []):
        if e2["direct"] and max(d["avg"] for d in e2["direct"]) >= cfg.CONFLICT_MIN_CONF:
            reasons.append("CONFLICTING_READINGS")
            break
    if det_conf is not None and det_conf < cfg.DET_CONF_MIN_SUCCESS:
        reasons.append("LOW_DETECTION_CONFIDENCE")
    if box_aspect is not None and not (cfg.PLATE_ASPECT_RANGE[0] <= box_aspect <= cfg.PLATE_ASPECT_RANGE[1]):
        reasons.append("UNUSUAL_PLATE_SHAPE")
    if "UNUSABLE" in quality["flags"]:
        reasons.append("POOR_IMAGE_QUALITY")

    status = SUCCESS if not reasons else REVIEW
    result.update(status=status, text=text, final_text=text if status == SUCCESS else "",
                  chosen=chosen, avg=avg, weakest=weakest, corrected=corrected,
                  substitutions=subs, validation=v, reasons=reasons,
                  manual_review_status="" if status == SUCCESS else MANUAL_REVIEW_REQUIRED)
    return result
