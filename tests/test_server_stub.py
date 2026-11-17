"""End-to-end test of server.py + static/index.html with a STUB pipeline (no YOLO/PARSeq).

Runs a real Flask server on a free port, drives it with headless Chromium (Playwright),
and checks the API directly. Uses a temporary results folder; your real results/ is untouched.

    pip install playwright pytest && playwright install chromium
    python -m pytest tests/test_server_stub.py -q
"""
import io
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dashboard_log as dl  # noqa: E402
import server  # noqa: E402


# ----------------------------------------------------------------------------- setup
def _point_results_to(tmp):
    dl.RESULTS = tmp
    dl.CROPS, dl.THUMBS, dl.ANNOT, dl.LOG = tmp / "crops", tmp / "thumbs", tmp / "annotated", tmp / "log.csv"
    dl.ENHANCED, dl.AUDIT = tmp / "enhanced", tmp / "corrections.csv"
    server.UPLOADS = tmp / "uploads"
    server.FILE_DIRS.update(thumbs=dl.THUMBS, crops=dl.CROPS, annotated=dl.ANNOT, enhanced=dl.ENHANCED)


PLATES = ["PY01DF8868", "RJ14CV0002", "MH12DE1433", "DL1CAB1234", "22BH1234AA", "TN20DC4647"]


class StubPipeline:
    """Fakes the model: by image index -> SUCCESS, REVIEW, or NO PLATE."""
    delay = 0.15

    def process(self, path, batch, det_conf, accept_thr, display_name):
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"{display_name}: could not open the image")
        time.sleep(self.delay)
        idx = int(Path(display_name).stem.split("_")[-1])
        thumb = dl.save_thumb(frame)
        cv2.imwrite(str(dl.ANNOT / thumb), frame)
        crop = frame[10:50, 10:170]
        if idx % 4 == 3:
            dl.save_result(batch, display_name, thumb)
            return []
        text = PLATES[idx % len(PLATES)]
        conf, status = (0.95, dl.SUCCESS) if idx % 4 != 2 else (0.62, dl.REVIEW)
        dl.save_result(batch, display_name, thumb, text, conf, 0.8, status, crop, enhanced_bgr=crop,
                       raw_ocr_text=text, char_confidences="|".join(f"{c}:0.9" for c in text),
                       final_registration_number=text if status == dl.SUCCESS else "",
                       manual_review_status="" if status == dl.SUCCESS else "MANUAL_REVIEW_REQUIRED")
        return [{"text": text, "conf": conf, "status": status}]


def _img_bytes(i):
    im = np.full((240, 320, 3), (40 + i * 7) % 255, np.uint8)
    cv2.rectangle(im, (10, 10), (170, 50), (255, 255, 255), -1)
    cv2.putText(im, f"IMG {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return cv2.imencode(".jpg", im)[1].tobytes()


@pytest.fixture(scope="module")
def env():
    tmp = Path(tempfile.mkdtemp(prefix="pv_test_"))
    _point_results_to(tmp / "results")
    imgdir = tmp / "imgs"
    imgdir.mkdir()
    for i in range(12):
        (imgdir / f"car_{i}.jpg").write_bytes(_img_bytes(i))
    app = server.create_app(StubPipeline(), "cpu")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{port}", "app": app, "imgs": imgdir, "tmp": tmp}
    srv.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


def _wait_idle(client, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if not client.get("/api/config").json["busy"]:
            return
        time.sleep(0.1)
    raise AssertionError("job did not finish")


# ----------------------------------------------------------------------------- browser
def test_browser_flow(env, tmp_path):
    from playwright.sync_api import sync_playwright, expect
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(accept_downloads=True, viewport={"width": 1280, "height": 900})
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(env["url"])
        expect(page.locator("#device")).to_have_text("device: cpu")

        # upload 12 images
        page.set_input_files("#pick", sorted(str(f) for f in env["imgs"].iterdir()))
        expect(page.locator("#sel")).to_contain_text("12 images selected")
        page.click("#go")
        # progress shows while running, rows appear before it finishes
        expect(page.locator("#ptext")).to_contain_text("Reading", timeout=10000)
        expect(page.locator("#rows tr:not(.empty-row)").first).to_be_visible(timeout=10000)
        mid = page.locator("#ptext").inner_text()
        expect(page.locator("#ptext")).to_contain_text("Finished 12 of 12", timeout=30000)
        page.screenshot(path=str(tmp_path / "done.png"))

        # summary for this upload: 12 images, 9 plates, 6 success, 3 review, 3 no plate
        expect(page.locator("#tab-batch")).to_have_attribute("aria-selected", "true")
        for k, v in (("images", "12"), ("plates", "9"), ("success", "6"), ("review", "3"), ("noplate", "3")):
            expect(page.locator(f"#s-{k}")).to_have_text(v)
        expect(page.locator("#s-overall")).to_have_text("66.7%")   # success rate: 6 SUCCESS of 9 plates
        expect(page.locator("#s-rate")).to_contain_text("mean OCR confidence 84.0%")   # (6*.95 + 3*.62)/9

        # pagination with 10 entries
        page.select_option("#limit", "10")
        expect(page.locator("#showing")).to_have_text("Showing 1 to 10 of 12 entries")
        page.click("#pages >> text=2")
        expect(page.locator("#showing")).to_have_text("Showing 11 to 12 of 12 entries")
        page.select_option("#limit", "25")

        # status filter
        page.click(".pill[data-s='REVIEW']")
        expect(page.locator("#showing")).to_have_text("Showing 1 to 3 of 3 entries")
        page.click(".pill[data-s='']")
        # search
        page.fill("#q", "PY01")
        expect(page.locator("#showing")).to_have_text("Showing 1 to 2 of 2 entries")
        expect(page.locator("#rows .plate").first).to_have_text("PY01DF8868")

        # modal
        page.locator("#rows tr td.time").first.click()
        expect(page.locator("#modal")).to_be_visible()
        expect(page.locator("#m-ocr-dl")).to_contain_text("PY01DF8868")
        assert page.evaluate("document.getElementById('m-img').naturalWidth") > 0
        page.keyboard.press("Escape")
        expect(page.locator("#modal")).to_be_hidden()

        # manual correction of the REVIEW row, with audit history; original text kept
        page.locator("#rows tr:has(.badge.b-REVIEW) td.time").first.click()
        expect(page.locator("#m-chars span")).to_have_count(10)
        assert page.evaluate("document.getElementById('m-enh').naturalWidth") > 0
        page.fill("#rv-text", "VEHICLE")
        page.click("#rv-save")
        expect(page.locator(".toast.err")).to_contain_text("not a valid Indian registration")
        page.fill("#rv-text", "py01 df 8860")
        page.fill("#rv-reason", "last digit misread")
        page.fill("#rv-reviewer", "QA tester")
        page.click("#rv-save")
        expect(page.locator("#m-audit li")).to_have_count(1)
        expect(page.locator("#m-audit li")).to_contain_text("PY01DF8868 → PY01DF8860")
        expect(page.locator("#m-ocr-dl")).to_contain_text("PY01DF8860")
        page.keyboard.press("Escape")
        expect(page.locator("#rows .tag.edit")).to_have_count(1)
        expect(page.locator("#s-corrected")).to_have_text("1")

        # Excel export respects the search
        with page.expect_download() as dl_info:
            page.click("#export")
        xlsx = tmp_path / "out.xlsx"
        dl_info.value.save_as(xlsx)
        from openpyxl import load_workbook
        wb = load_workbook(xlsx)
        ws = wb["Results"]
        assert ws.max_row == 3                                  # header + 2 rows
        assert {ws.cell(r, 4).value for r in (2, 3)} == {"PY01DF8868"}
        assert "PY01DF8860" in {ws.cell(r, 21).value for r in (2, 3)}        # corrected text column
        assert len(ws._images) == 4                             # vehicle + plate per row
        assert wb["Summary"]["B2"].value == 2

        # light theme toggle persists
        page.click("#theme")
        page.reload()
        assert page.evaluate("document.documentElement.dataset.theme") == "light"

        # phone width: no horizontal page scroll
        page.set_viewport_size({"width": 375, "height": 800})
        page.wait_for_timeout(200)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

        # clear history
        page.set_viewport_size({"width": 1280, "height": 900})
        page.on("dialog", lambda d: d.accept())
        page.click("#clear")
        expect(page.locator("#showing")).to_have_text("Showing 0 to 0 of 0 entries")
        expect(page.locator("#s-images")).to_have_text("0")
        browser.close()
    assert "Reading" in mid
    expected = [e for e in errors if "status of 400" in e]     # the deliberately invalid correction
    assert len(expected) <= 1
    assert not [e for e in errors if e not in expected], errors


# ----------------------------------------------------------------------------- API
def test_api_conflict_cap_and_export(env):
    c = env["app"].test_client()
    files = lambda n: [(io.BytesIO(_img_bytes(i)), f"car_{i}.jpg") for i in range(n)]

    # 409 on a second job while one runs; 409 on clear while running
    StubPipeline.delay = 0.3
    r1 = c.post("/api/jobs", data={"files": files(5)}, content_type="multipart/form-data")
    assert r1.status_code == 202
    r2 = c.post("/api/jobs", data={"files": files(1)}, content_type="multipart/form-data")
    assert r2.status_code == 409
    assert c.delete("/api/results").status_code == 409
    assert c.get("/api/results").headers["Cache-Control"] == "no-store"
    _wait_idle(c)

    # upload cap (server.MAX_IMAGES): extra files are skipped, not processed
    StubPipeline.delay = 0.0
    cap = server.MAX_IMAGES
    r = c.post("/api/jobs", data={"files": files(cap + 5), "conf": "0.3", "accept": "0.9"},
               content_type="multipart/form-data")
    assert r.status_code == 202 and r.json["total"] == cap and r.json["skipped"] == 5
    _wait_idle(c)
    j = c.get(f"/api/jobs/{r.json['id']}").json
    assert j["status"] == "done" and j["done"] == cap and j["summary"]["images"] == cap

    # cancel
    StubPipeline.delay = 0.2
    r = c.post("/api/jobs", data={"files": files(20)}, content_type="multipart/form-data")
    c.post(f"/api/jobs/{r.json['id']}/cancel")
    _wait_idle(c)
    j = c.get(f"/api/jobs/{r.json['id']}").json
    assert j["status"] == "cancelled" and j["done"] < 20

    # a broken image does not stop the job
    StubPipeline.delay = 0.0
    bad = [(io.BytesIO(b"not an image"), "car_0.jpg"), (io.BytesIO(_img_bytes(1)), "car_1.jpg")]
    r = c.post("/api/jobs", data={"files": bad}, content_type="multipart/form-data")
    _wait_idle(c)
    j = c.get(f"/api/jobs/{r.json['id']}").json
    assert j["done"] == 2 and len(j["errors"]) == 1 and j["summary"]["images"] == 1

    # uploads are deleted, files endpoint is whitelisted
    assert not any(server.UPLOADS.iterdir())
    assert c.get("/files/../log.csv").status_code == 404
    assert c.get("/files/uploads/x.jpg").status_code == 404

    # review endpoints
    rid = c.get("/api/results?limit=1").json["rows"][0]["row_id"]
    assert c.post(f"/api/results/{rid}/review", json={"corrected_text": "RANDOMTEXT"}).status_code == 400
    r = c.post(f"/api/results/{rid}/review", json={"corrected_text": "TN01AW5679", "reviewer": "x"})
    assert r.status_code == 200 and r.json["validation"]["state_code"] == "TN"
    assert len(c.get(f"/api/results/{rid}/audit").json["history"]) == 1
    assert c.post("/api/results/nope/review", json={"corrected_text": "TN01AW5679"}).status_code == 404
    assert c.get("/api/results?status=REJECT").status_code == 200

    # export filtered by status
    x = c.get("/api/export?status=NO%20PLATE")
    from openpyxl import load_workbook
    ws = load_workbook(io.BytesIO(x.data))["Results"]
    assert all(ws.cell(r, 7).value == "NO PLATE" for r in range(2, ws.max_row + 1))


def test_plate_rules():
    from plate_reader import extract_plate
    assert extract_plate("INDRJ14CV0002") == "RJ14CV0002"
    assert extract_plate("PY01DF8868") == "PY01DF8868"
    assert extract_plate("22BH1234AA") == "22BH1234AA"
    assert extract_plate("DL1CAB1234") == "DL1CAB1234"
    assert extract_plate("14CV0002") == ""                    # cut-off read is not a plate
    assert extract_plate("XX12AB1234") == ""                  # not a state code
    assert dl.decide_status(False, 0.99, 0.85, 0.99) == dl.REVIEW
    assert dl.decide_status(True, 0.99, 0.85, 0.20) == dl.REVIEW
    assert dl.decide_status(True, 0.90, 0.85, 0.50) == dl.SUCCESS


def test_merge_boxes():
    from pipeline import merge_boxes
    boxes = [[100, 50, 160, 80, .8], [150, 52, 230, 82, .7], [400, 300, 480, 330, .9]]
    merged = merge_boxes(boxes)
    assert len(merged) == 2
    assert merged[0][0][:4] == [100, 50, 230, 82]
