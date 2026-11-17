"""Stores every read plate (text, confidence, status, images) under results/, keeps the manual
correction audit trail, and exports to Excel. No torch needed here.

results/log.csv       one row per plate (or per image with no plate). The first 9 columns are the
                      original layout; new columns were appended (older logs are migrated in place).
results/corrections.csv  append-only audit of every manual correction / confirmation.
The OCR output columns are never changed after a row is written; corrections live in their own
columns (manually_corrected_text, ...)."""
import csv
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2

RESULTS = Path(__file__).resolve().parents[1] / "results"
CROPS = RESULTS / "crops"
THUMBS = RESULTS / "thumbs"
ANNOT = RESULTS / "annotated"
ENHANCED = RESULTS / "enhanced"
LOG = RESULTS / "log.csv"
AUDIT = RESULTS / "corrections.csv"
TRAIN_CROPS = RESULTS / "train_crops"   # un-enhanced crop each reading came from (fine-tuning data)

FIELDS_V1 = ["time", "batch", "source", "plate_text", "ocr_conf", "weakest",
             "status", "crop_file", "thumb_file"]
NEW_FIELDS = ["row_id", "det_conf", "raw_ocr_text", "normalized_ocr_text",
              "final_registration_number", "format_validation_status", "state_code", "rto_code",
              "registration_series", "registration_digits", "char_confidences", "substitutions",
              "candidates", "retry_count", "ocr_attempts", "preprocessing_method", "quality_flags",
              "review_reasons", "manual_review_status", "manually_corrected_text",
              "correction_reason", "reviewer", "correction_timestamp", "enhanced_file",
              "duplicate_of", "ml_probability", "rule_status",
              "ocr_engine", "ocr_status", "ocr_time_ms", "processing_seconds", "image_quality",
              "train_crop_file", "resolver_probability", "resolved_by",
              "calibrated_probability", "hybrid_reasons", "tta_result", "ocr_model"]
FIELDS = FIELDS_V1 + NEW_FIELDS
AUDIT_FIELDS = ["time", "row_id", "batch", "source", "ocr_text", "previous_corrected_text",
                "corrected_text", "format_validation", "action", "reason", "reviewer"]

MIN_CHAR_CONF = 0.30   # used only by the legacy decide_status below
SUCCESS, REVIEW, REJECT, NO_PLATE = "SUCCESS", "REVIEW", "REJECT", "NO PLATE"
STATUSES = (SUCCESS, REVIEW, REJECT, NO_PLATE)
_LOCK = threading.RLock()   # the web server reads the log while a job appends to it


def decide_status(is_valid_plate, avg_conf, threshold, weakest=1.0):
    """LEGACY rule (v1), kept for evaluate.py's before/after comparison."""
    ok = is_valid_plate and avg_conf >= threshold and weakest >= MIN_CHAR_CONF
    return SUCCESS if ok else REVIEW


def _new_id():
    return uuid.uuid4().hex[:16]


def _write_all(rows):
    tmp = LOG.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, LOG)


def _migrate_v1():
    """Add the new columns to a v1 log, keeping every row."""
    from indian_plate_validator import validate
    with open(LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["row_id"] = _new_id()
        if r["status"] != NO_PLATE:
            v = validate(r["plate_text"])
            r["format_validation_status"] = "VALID" if v["is_valid_format"] else v["reason"]
    LOG.rename(RESULTS / f"log_v1_backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
    _write_all(rows)


def _upgrade_columns():
    """A newer version added columns: rewrite the log with them (empty for old rows)."""
    with open(LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    LOG.rename(RESULTS / f"log_backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
    _write_all(rows)


def _ensure_log():
    for d in (RESULTS, CROPS, THUMBS, ANNOT, ENHANCED, TRAIN_CROPS):
        d.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if LOG.exists():
            with open(LOG, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header == FIELDS_V1:
                _migrate_v1()
            elif header != FIELDS and set(header) <= set(FIELDS) and "row_id" in header:
                _upgrade_columns()
            elif header != FIELDS:   # unknown layout: keep it aside, start fresh
                LOG.rename(RESULTS / f"log_old_{datetime.now():%Y%m%d_%H%M%S}.csv")
        if not LOG.exists():
            with open(LOG, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def new_batch_id():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _img_name():
    return f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"


def save_thumb(frame_bgr, height=64):
    """Small copy of the whole vehicle image for the dashboard. Returns the file name."""
    _ensure_log()
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)
    name = _img_name()
    cv2.imwrite(str(THUMBS / name), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return name


def save_result(batch, source, thumb_file, text="", ocr_conf=0.0, weakest=0.0,
                status=NO_PLATE, crop_bgr=None, enhanced_bgr=None, train_bgr=None, **extra):
    """Append one row. extra: any of NEW_FIELDS. Returns the row id.
    train_bgr: the un-enhanced crop the reading came from (used when the row is verified)."""
    _ensure_log()
    crop_file = enh_file = train_file = ""
    if train_bgr is not None and train_bgr.size:
        train_file = _img_name()
        cv2.imwrite(str(TRAIN_CROPS / train_file), train_bgr)
    if crop_bgr is not None and crop_bgr.size:
        crop_file = _img_name()
        cv2.imwrite(str(CROPS / crop_file), crop_bgr)
    if enhanced_bgr is not None and enhanced_bgr.size:
        enh_file = _img_name()
        cv2.imwrite(str(ENHANCED / enh_file), enhanced_bgr)
    row = {k: "" for k in FIELDS}
    row.update({k: ("" if v is None else v) for k, v in extra.items() if k in NEW_FIELDS})
    if isinstance(row.get("det_conf"), float):
        row["det_conf"] = f"{row['det_conf']:.4f}"
    row.update({
        "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "batch": batch, "source": source,
        "plate_text": text, "ocr_conf": f"{ocr_conf:.4f}", "weakest": f"{weakest:.4f}",
        "status": status, "crop_file": crop_file, "thumb_file": thumb_file,
        "row_id": _new_id(), "enhanced_file": enh_file, "train_crop_file": train_file})
    with _LOCK, open(LOG, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    return row["row_id"]


def _read_all():
    if not LOG.exists():
        return []
    _ensure_log()
    with _LOCK, open(LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_verified(r):
    """A person confirmed or corrected this plate in manual review."""
    return bool(r.get("manually_corrected_text"))


def effective_text(r):
    """What the plate is taken to be: a manual correction wins, else the pipeline's final
    registration (SUCCESS only), else ''."""
    return r.get("manually_corrected_text") or r.get("final_registration_number") or ""


def load_rows(query="", limit=None, batch=None, status=None):
    """Newest first. query matches texts, file name, status or time; status filters exactly
    ('CORRECTED' = rows with a manual correction)."""
    rows = _read_all()[::-1]
    if batch:
        rows = [r for r in rows if r["batch"] == batch]
    if status == "CORRECTED":
        rows = [r for r in rows if r.get("manually_corrected_text")]
    elif status:
        rows = [r for r in rows if r["status"] == status]
    q = (query or "").strip().lower()
    if q:
        rows = [r for r in rows if q in " ".join(
            (r["plate_text"], r["source"], r["status"], r["time"], r.get("raw_ocr_text", ""),
             r.get("manually_corrected_text", ""), r.get("manual_review_status", ""))).lower()]
    return rows[:limit] if limit else rows


def get_row(row_id):
    for r in _read_all():
        if r.get("row_id") == row_id:
            return r
    return None


def summarize(rows):
    """Counts and the overall confidence for a set of rows.
    plates = SUCCESS + REVIEW rows (registration candidates). REJECT rows are detections whose
    text is not a registration number; they are counted separately.
    Overall confidence = mean of the average-character confidence over the plates."""
    found = [r for r in rows if r["status"] in (SUCCESS, REVIEW)]
    confs = [float(r["ocr_conf"]) for r in found]
    success = sum(r["status"] == SUCCESS for r in rows)
    return {
        "images": len({(r["batch"], r["thumb_file"]) for r in rows}),
        "plates": len(found),
        "success": success,
        "review": sum(r["status"] == REVIEW for r in rows),
        "reject": sum(r["status"] == REJECT for r in rows),
        "no_plate": sum(r["status"] == NO_PLATE for r in rows),
        "manual_review_pending": sum(r.get("manual_review_status") == "MANUAL_REVIEW_REQUIRED"
                                     for r in rows),
        "corrected": sum(bool(r.get("manually_corrected_text")) for r in rows),
        "overall": (sum(confs) / len(confs)) if confs else None,
        "success_rate": (success / len(found)) if found else None,
        # automatic SUCCESS + plates a person confirmed or corrected in manual review
        "verified": success + sum(1 for r in found if r["status"] != SUCCESS
                                  and r.get("manually_corrected_text")),
        "verified_rate": ((success + sum(1 for r in found if r["status"] != SUCCESS
                                         and r.get("manually_corrected_text"))) / len(found))
                         if found else None,
    }


def correct(row_id, corrected_text, reason="", reviewer="", allow_invalid=False):
    """Record a reviewer's correction (or confirmation, when the text equals the OCR text).
    The OCR columns are untouched. Every call is appended to corrections.csv.
    Returns (row, validation). Raises KeyError (no row) or ValueError (invalid format)."""
    from indian_plate_validator import normalize, validate
    text = normalize(corrected_text)
    v = validate(text)
    if not v["is_valid_format"] and not allow_invalid:
        raise ValueError(f"'{text or corrected_text}' is not a valid Indian registration format "
                         f"({v['reason']}).")
    now = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    with _LOCK:
        rows = _read_all()
        row = next((r for r in rows if r.get("row_id") == row_id), None)
        if row is None:
            raise KeyError(row_id)
        previous = row.get("manually_corrected_text", "")
        action = "CONFIRMED" if text == row["plate_text"] and row["status"] != NO_PLATE else "CORRECTED"
        row.update(manually_corrected_text=text, correction_reason=reason, reviewer=reviewer,
                   correction_timestamp=now, manual_review_status=action)
        _write_all(rows)
        new = not AUDIT.exists()
        with open(AUDIT, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"time": now, "row_id": row_id, "batch": row["batch"], "source": row["source"],
                        "ocr_text": row["plate_text"], "previous_corrected_text": previous,
                        "corrected_text": text,
                        "format_validation": "VALID" if v["is_valid_format"] else v["reason"],
                        "action": action, "reason": reason, "reviewer": reviewer})
    return row, v


def audit_history(row_id):
    if not AUDIT.exists():
        return []
    with _LOCK, open(AUDIT, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["row_id"] == row_id][::-1]


def clear():
    with _LOCK:
        for p in (LOG, AUDIT):
            if p.exists():
                p.unlink()
        for d in (CROPS, THUMBS, ANNOT, ENHANCED, RESULTS / "uploads"):
            if d.exists():
                for p in d.iterdir():
                    if p.is_file():
                        p.unlink()


# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------
def export_excel(rows=None, out=None):
    """Write all rows (or the given rows) to an .xlsx file (default results/export_<time>.xlsx).
    Returns the path."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from PIL import Image as PILImage

    _ensure_log()
    rows = load_rows() if rows is None else rows
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    cols = [("Date/Time", 20), ("Vehicle image", 16), ("Plate image", 26), ("Extracted text", 20),
            ("Confidence (%)", 15), ("Weakest char (%)", 16), ("Status", 14), ("File", 28),
            ("Final registration", 20), ("Raw OCR", 22), ("Format check", 24), ("State", 8),
            ("RTO", 7), ("Series", 8), ("Number", 9), ("Review reasons", 40), ("OCR attempts", 9),
            ("Retries", 8), ("Preprocessing", 16), ("Manual review", 22), ("Corrected text", 18),
            ("Correction reason", 26), ("Reviewer", 14), ("Corrected at", 20),
            ("ML P(correct) %", 12), ("Rule-only status", 12), ("OCR engine", 18),
            ("OCR status", 22), ("OCR time (ms)", 11), ("Processing time (s)", 12),
            ("Image quality", 22), ("Calibrated P(correct) %", 13), ("TTA check", 18), ("OCR model", 18)]
    ws.append([c for c, _ in cols])
    head_fill = PatternFill("solid", fgColor="1E2840")
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = head_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, (_, w) in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    fills = {SUCCESS: "C6EFCE", REVIEW: "FFEB9C", REJECT: "F4C7C3", NO_PLATE: "E0E0E0"}
    for i, r in enumerate(rows, start=2):
        found = r["status"] != NO_PLATE
        g = r.get
        ws.append([r["time"], "", "", r["plate_text"] if found else "",
                   round(float(r["ocr_conf"]) * 100, 1) if found else "",
                   round(float(r["weakest"]) * 100, 1) if found else "",
                   ("VERIFIED" if is_verified(r) else r["status"]), r["source"],
                   effective_text(r),
                   g("raw_ocr_text", ""), g("format_validation_status", ""), g("state_code", ""),
                   g("rto_code", ""), g("registration_series", ""), g("registration_digits", ""),
                   (g("review_reasons") or "").replace("|", ", "), g("ocr_attempts", ""),
                   g("retry_count", ""), g("preprocessing_method", ""),
                   g("manual_review_status", ""), g("manually_corrected_text", ""),
                   g("correction_reason", ""), g("reviewer", ""), g("correction_timestamp", ""),
                   round(float(g("ml_probability")) * 100, 1) if g("ml_probability") else "",
                   g("rule_status", ""), g("ocr_engine", ""), g("ocr_status", ""),
                   g("ocr_time_ms", ""), g("processing_seconds", ""), g("image_quality", ""),
                   round(float(g("calibrated_probability")) * 100, 1) if g("calibrated_probability") else "",
                   g("tta_result", ""), g("ocr_model", "")])
        ws.row_dimensions[i].height = 52
        for c in ws[i]:
            c.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
        ws.cell(i, 7).fill = PatternFill("solid", fgColor=fills.get(SUCCESS if is_verified(r) else r["status"], "FFFFFF"))
        ws.cell(i, 4).font = Font(name="Consolas", bold=True, size=12)
        for col, folder, fname, max_h, max_w in (("B", THUMBS, r["thumb_file"], 64, 110),
                                                 ("C", CROPS, r["crop_file"], 44, 170)):
            p = folder / fname if fname else None
            if p and p.exists():
                w0, h0 = PILImage.open(p).size
                scale = min(max_h / h0, max_w / w0)
                img = XLImage(str(p))
                img.width, img.height = int(w0 * scale), int(h0 * scale)
                ws.add_image(img, f"{col}{i}")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{max(2, len(rows) + 1)}"

    s = summarize(rows)
    ws2 = wb.create_sheet("Summary")
    for k, v in (("Images uploaded", s["images"]), ("Plates found", s["plates"]),
                 ("Success", s["success"]), ("Needs review", s["review"]),
                 ("Rejected (not a registration)", s["reject"]),
                 ("No plate detected", s["no_plate"]),
                 ("Manually corrected / confirmed", s["corrected"]),
                 ("Overall confidence (%)", round(s["overall"] * 100, 1) if s["overall"] is not None else "-"),
                 ("Success rate (%)", round(100 * s["success_rate"], 1) if s["success_rate"] is not None else "-")):
        ws2.append([k, v])
    ws2.column_dimensions["A"].width = 32
    for c in ws2["A"]:
        c.font = Font(bold=True)

    out = out or RESULTS / f"export_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    wb.save(out)
    return str(out)
