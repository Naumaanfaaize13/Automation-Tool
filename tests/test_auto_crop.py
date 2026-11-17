"""Automatic crop mode: plate-crop detection and the choice between YOLO and whole-image reads."""
import pipeline as P
import plate_decision as pdec


def _plate(status, text, avg, valid=True):
    return {"status": status, "text": text, "avg": avg, "final_text": text if status == pdec.SUCCESS else "",
            "decision": {"status": status, "reasons": [], "final_text": text if status == pdec.SUCCESS else "",
                         "validation": {"is_valid_format": valid}, "manual_review_status": ""},
            "raw": text, "chars": [{"char": c, "p": avg} for c in text]}


def test_plate_shaped_images_are_crops():
    assert P.looks_like_plate_crop(640, 186, [])                       # no box, plate-shaped
    assert P.looks_like_plate_crop(640, 186, [[4, 5, 354, 133, 0.5]])  # box covers ~37 %
    assert P.looks_like_plate_crop(640, 581, [[161, 73, 640, 513, 0.8]])


def test_full_scenes_are_not_crops():
    assert not P.looks_like_plate_crop(1920, 1080, [])                  # too large
    assert not P.looks_like_plate_crop(800, 600, [[350, 400, 450, 430, 0.9]])  # plate ~0.6 %
    assert not P.looks_like_plate_crop(200, 900, [])                    # portrait, not a plate


def test_whole_image_reading_wins_when_better():
    yolo = _plate(pdec.REVIEW, "TN07B92", 0.83, valid=False)
    whole = _plate(pdec.SUCCESS, "TN13AF4447", 0.99)
    best = P.Pipeline._pick_crop_reading([yolo], whole)
    assert best is whole and best["status"] == pdec.SUCCESS and best["auto_crop"]


def test_confident_disagreement_goes_to_review(monkeypatch):
    monkeypatch.setattr(P.cfg, "CROP_CONFLICT_MODE", "any")
    yolo = _plate(pdec.REVIEW, "MH39BH5441", 0.91)
    whole = _plate(pdec.SUCCESS, "TN19BH5441", 0.99)
    best = P.Pipeline._pick_crop_reading([yolo], whole)
    assert best["status"] == pdec.REVIEW and best["final_text"] == ""
    assert "CONFLICTING_READINGS" in best["decision"]["reasons"]


def test_success_only_mode_ignores_weaker_review_reading(monkeypatch):
    monkeypatch.setattr(P.cfg, "CROP_CONFLICT_MODE", "success_only")
    best = P.Pipeline._pick_crop_reading([_plate(pdec.REVIEW, "MH39BH5441", 0.91)],
                                         _plate(pdec.SUCCESS, "TN19BH5441", 0.99))
    assert best["status"] == pdec.SUCCESS


def test_conflict_check_off_keeps_best_reading(monkeypatch):
    monkeypatch.setattr(P.cfg, "CROP_CONFLICT_MODE", "off")
    best = P.Pipeline._pick_crop_reading([_plate(pdec.SUCCESS, "MH02EU3764", 0.98)],
                                         _plate(pdec.SUCCESS, "MH02FU3764", 0.99))
    assert best["status"] == pdec.SUCCESS and best["text"] == "MH02FU3764"


def test_agreeing_readings_stay_success():
    yolo = _plate(pdec.SUCCESS, "TN09BR6934", 0.99)
    whole = _plate(pdec.REVIEW, "TN09BR6934", 0.95)
    best = P.Pipeline._pick_crop_reading([yolo], whole)
    assert best is yolo and best["status"] == pdec.SUCCESS


def test_reading_without_warnings_preferred():
    yolo = _plate(pdec.REVIEW, "TN09BR6934", 0.997)
    whole = _plate(pdec.REVIEW, "DN09BR6934", 0.999)
    whole["decision"]["validation"]["warnings"] = ["LEGACY_STATE_CODE"]
    assert P.Pipeline._pick_crop_reading([yolo], whole) is yolo
