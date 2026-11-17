"""Unit tests for format validation, position-aware candidates, the decision engine and the
results store (migration + audited corrections). No models needed.

    python -m pytest tests/test_anpr_rules.py -q
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard_log as dl  # noqa: E402
import indian_plate_validator as ipv  # noqa: E402
import plate_decision as pdec  # noqa: E402


# ----------------------------------------------------------------------------- validator
@pytest.mark.parametrize("text,state,rto,series,num", [
    ("TN01AW5679", "TN", "01", "AW", "5679"), ("TN01AB1234", "TN", "01", "AB", "1234"),
    ("MH12DE1433", "MH", "12", "DE", "1433"), ("KA03MN1234", "KA", "03", "MN", "1234"),
    ("PY01AA1234", "PY", "01", "AA", "1234"), ("DL8CAF1234", "DL", "8", "CAF", "1234"),
    ("AP39CD1234", "AP", "39", "CD", "1234"), ("KL07AB1234", "KL", "07", "AB", "1234"),
    ("TN91B3184", "TN", "91", "B", "3184"), ("DL10CA1234", "DL", "10", "CA", "1234"),
])
def test_valid_formats(text, state, rto, series, num):
    v = ipv.validate(text)
    assert v["is_valid_format"], v
    assert (v["state_code"], v["rto_code"], v["series"], v["registration_number"]) == (state, rto, series, num)
    assert v["warnings"] == []


def test_bh_series():
    v = ipv.validate("22BH1234AA")
    assert v["is_valid_format"] and v["format"] == "BH" and v["registration_number"] == "1234"
    assert ipv.validate("23BH4850C")["is_valid_format"]
    assert ipv.validate("19BH1234AA")["reason"] == "INVALID_BH_YEAR"     # BH started in 2021


def test_raw_text_is_normalised_not_changed():
    v = ipv.validate("tn 01 aw 5679")
    assert v["raw_text"] == "tn 01 aw 5679"
    assert v["normalized_text"] == "TN01AW5679" and v["is_valid_format"]
    assert ipv.normalize("TN-01.AW 5679") == "TN01AW5679"
    assert ipv.normalize("TNO1AW5679") == "TNO1AW5679"          # O is NOT turned into 0


@pytest.mark.parametrize("text,reason", [
    ("CHEVROLETELIBERTY", "TOO_LONG"), ("STREETSTATES", "TOO_LONG"),
    ("RANDOMTEXT", "INVALID_REGISTRATION_FORMAT"), ("VEHICLE", "INVALID_REGISTRATION_FORMAT"),
    ("UNKNOWN", "INVALID_REGISTRATION_FORMAT"), ("XX12AB1234", "UNKNOWN_STATE_CODE"),
    ("MH00B1770", "INVALID_RTO_CODE"), ("TN1AB1234", "INVALID_RTO_CODE"),
    ("TN01AB0000", "INVALID_NUMBER"), ("14CV0002", "INVALID_REGISTRATION_FORMAT"),
    ("TN01ABCD1234", "TOO_LONG"), ("TN01ABC12345", "TOO_LONG"), ("TN01A1B234", "INVALID_REGISTRATION_FORMAT"), ("", "EMPTY_TEXT"), ("TN01", "TOO_SHORT"),
])
def test_invalid(text, reason):
    v = ipv.validate(text)
    assert not v["is_valid_format"] and v["reason"] == reason


def test_warnings_keep_valid_reads_out_of_success():
    assert "SERIES_HAS_I_OR_O" in ipv.validate("TN01OB1234")["warnings"]
    assert "NO_SERIES_LETTERS" in ipv.validate("MH031225")["warnings"]
    assert "LEGACY_STATE_CODE" in ipv.validate("OR02AB1234")["warnings"]


def test_find_in_text():
    v = ipv.find_in_text("INDRJ14CV0002")
    assert v["normalized_text"] == "RJ14CV0002" and v["warnings"] == []
    v = ipv.find_in_text("XQRJ14CV0002ZZ")
    assert v["normalized_text"] == "RJ14CV0002" and "EXTRACTED_FROM_LONGER_TEXT" in v["warnings"]
    assert not ipv.find_in_text("COMPAGNERPHOTOGRAPHY")["is_valid_format"]


# ----------------------------------------------------------------------------- helpers
def chars(text, p=0.99, low=None):
    """low: {pos: (p_of_char, [(alt, p_alt), ...])}"""
    out = []
    for i, c in enumerate(text):
        if low and i in low:
            pc, alts = low[i]
            out.append({"char": c, "p": pc, "alts": [(c, pc)] + alts})
        else:
            out.append({"char": c, "p": p, "alts": [(c, p)]})
    return out


def attempt(text, p=0.99, low=None, variant="original"):
    cs = chars(text, p, low)
    import math
    ps = [c["p"] for c in cs]
    return {"variant": variant, "raw": text, "chars": cs, "img": None,
            "avg": math.exp(sum(math.log(x) for x in ps) / len(ps)), "weakest": min(ps)}


# ----------------------------------------------------------------------------- confusions
@pytest.mark.parametrize("read,pos,wrong,right,expect", [
    ("TNO1AW5679", 2, "O", "0", "TN01AW5679"),     # O vs 0 in the RTO digits
    ("TN0IAW5679", 3, "I", "1", "TN01AW5679"),     # I vs 1
    ("MH12DE14B3", 8, "B", "8", "MH12DE1483"),     # B vs 8 in the number
    ("KA03MN12S4", 8, "S", "5", "KA03MN1254"),     # S vs 5
    ("PY01AAZ234", 6, "Z", "2", "PY01AA2234"),     # Z vs 2
    ("TN01AW5G79", 7, "G", "6", "TN01AW5679"),     # G vs 6
])
def test_confusions_become_review_suggestions_never_success(read, pos, wrong, right, expect):
    a = attempt(read, low={pos: (0.55, [(right, 0.40)])})
    c = ipv.candidates(a["chars"])
    assert c and c[0]["text"] == expect and c[0]["substitutions"][0]["to"] == right
    d = pdec.decide([a])
    assert d["status"] == pdec.REVIEW
    assert d["text"] == expect and d["final_text"] == ""            # suggestion, not a final answer
    assert d["manual_review_status"] == pdec.MANUAL_REVIEW_REQUIRED
    assert "CHARACTERS_SUBSTITUTED_FROM_MODEL_ALTERNATIVES" in d["reasons"]


def test_no_substitution_without_model_evidence():
    # the model gave '6' almost no probability: the G must not be "fixed" into a 6
    a = attempt("TN01AW5G79", low={7: (0.97, [("6", 0.001)])})
    assert ipv.candidates(a["chars"]) == []
    d = pdec.decide([a])
    assert d["status"] != pdec.SUCCESS and d["final_text"] == ""


def test_valid_but_uncertain_letter_is_review():
    a = attempt("TN01AB1234", low={5: (0.55, [("8", 0.44)])})
    d = pdec.decide([a])
    assert d["status"] == pdec.REVIEW and "UNCERTAIN_CHARACTER" in d["reasons"]


# ----------------------------------------------------------------------------- decision
def test_confident_valid_is_success():
    d = pdec.decide([attempt("MH12DE1433")], det_conf=0.8, box_aspect=4.0)
    assert d["status"] == pdec.SUCCESS and d["final_text"] == "MH12DE1433" and d["reasons"] == []


@pytest.mark.parametrize("junk", ["CHEVROLETELIBERTY", "STREETSTATES", "RANDOMTEXT", "VEHICLE", "UNKNOWN"])
def test_confident_words_are_rejected(junk):
    d = pdec.decide([attempt(junk, p=0.999)], det_conf=0.9, box_aspect=4.0)
    assert d["status"] == pdec.REJECT and d["final_text"] == ""
    assert "NOT_A_REGISTRATION_NUMBER" in d["reasons"]


def test_partial_plate_is_review_not_guessed():
    d = pdec.decide([attempt("2DE1433", p=0.999)])       # first letters cut off
    assert d["status"] == pdec.REVIEW and d["final_text"] == ""
    assert "PARTIAL_OR_INVALID_PLATE_TEXT" in d["reasons"]


def test_conflicting_confident_readings_are_review():
    d = pdec.decide([attempt("MH12DE1433"), attempt("MH12DE1438", variant="clahe")])
    assert d["status"] == pdec.REVIEW and "CONFLICTING_READINGS" in d["reasons"]


def test_agreement_picks_majority():
    d = pdec.decide([attempt("MH12DE1433", 0.97), attempt("MH12DE1433", 0.96),
                     attempt("MH12DE1438", 0.70)])
    assert d["text"] == "MH12DE1433"


def test_detection_and_geometry_block_success():
    a = [attempt("MH12DE1433")]
    assert "LOW_DETECTION_CONFIDENCE" in pdec.decide(a, det_conf=0.1)["reasons"]
    assert "UNUSUAL_PLATE_SHAPE" in pdec.decide(a, box_aspect=12.0)["reasons"]
    assert "POOR_IMAGE_QUALITY" in pdec.decide(a, quality={"flags": ["UNUSABLE"]})["reasons"]


def test_thresholds_are_configurable():
    a = [attempt("MH12DE1433", low={9: (0.75, [("8", 0.2)])})]
    assert pdec.decide(a)["status"] == pdec.REVIEW
    assert pdec.decide(a, accept_avg=0.9, accept_weakest=0.7)["status"] == pdec.SUCCESS


# ----------------------------------------------------------------------------- results store
@pytest.fixture
def store(tmp_path, monkeypatch):
    r = tmp_path / "results"
    for k, v in dict(RESULTS=r, CROPS=r / "crops", THUMBS=r / "thumbs", ANNOT=r / "annotated",
                     ENHANCED=r / "enhanced", LOG=r / "log.csv", AUDIT=r / "corrections.csv").items():
        monkeypatch.setattr(dl, k, v)
    return r


def test_v1_log_is_migrated_without_losing_rows(store):
    store.mkdir()
    with open(dl.LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dl.FIELDS_V1)
        w.writeheader()
        w.writerow({"time": "2026-11-01 10:00:00", "batch": "b1", "source": "a.jpg", "plate_text": "PY01DF8868",
                    "ocr_conf": "0.97", "weakest": "0.9", "status": "SUCCESS", "crop_file": "c.jpg", "thumb_file": "t.jpg"})
        w.writerow({"time": "2026-11-01 10:00:01", "batch": "b1", "source": "b.jpg", "plate_text": "",
                    "ocr_conf": "0", "weakest": "0", "status": "NO PLATE", "crop_file": "", "thumb_file": "t2.jpg"})
    rows = dl.load_rows()
    assert len(rows) == 2 and all(r["row_id"] for r in rows)
    assert rows[1]["plate_text"] == "PY01DF8868" and rows[1]["format_validation_status"] == "VALID"
    assert list(store.glob("log_v1_backup_*.csv"))


def test_correction_keeps_original_and_is_audited(store):
    rid = dl.save_result("b", "car.jpg", "t.jpg", "MH12DE1438", 0.7, 0.4, dl.REVIEW,
                         raw_ocr_text="MH12DE1438", manual_review_status="MANUAL_REVIEW_REQUIRED")
    with pytest.raises(ValueError):
        dl.correct(rid, "VEHICLE")
    row, v = dl.correct(rid, "mh 12 de 1433", reason="8 was 3", reviewer="QA")
    assert v["is_valid_format"]
    row = dl.get_row(rid)
    assert row["plate_text"] == "MH12DE1438" and row["raw_ocr_text"] == "MH12DE1438"   # untouched
    assert row["manually_corrected_text"] == "MH12DE1433" and row["manual_review_status"] == "CORRECTED"
    dl.correct(rid, "MH12DE1438", reason="actually right", reviewer="QA2")
    hist = dl.audit_history(rid)
    assert [h["corrected_text"] for h in hist] == ["MH12DE1438", "MH12DE1433"]
    assert hist[0]["previous_corrected_text"] == "MH12DE1433" and hist[0]["action"] == "CONFIRMED"
    assert dl.summarize(dl.load_rows())["corrected"] == 1
    assert len(dl.load_rows(status="CORRECTED")) == 1
    with pytest.raises(KeyError):
        dl.correct("nope", "MH12DE1433")
