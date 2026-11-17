"""PlateVision web server.

Run:   python server.py            then open http://127.0.0.1:5000
Flags: --weights PATH  --device cpu|cuda  --host 127.0.0.1  --port 5000
"""
import argparse
import sys
import inspect
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # core modules live in src/

import anpr_config as cfg  # noqa: E402
import dashboard_log as dl
import indian_plate_validator as ipv

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
UPLOADS = dl.RESULTS / "uploads"
MAX_IMAGES = 500         # images per upload (was 150)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILE_DIRS = {"thumbs": dl.THUMBS, "crops": dl.CROPS, "annotated": dl.ANNOT, "enhanced": dl.ENHANCED,
             "train_crops": dl.TRAIN_CROPS}
MIN_TUNE_SAMPLES = 150    # verified plates needed before the dashboard offers fine-tuning
DEFAULTS = {"conf": 0.25, "accept": cfg.ACCEPT_AVG_CONF, "weakest": cfg.ACCEPT_WEAKEST_CHAR}


def default_weights():
    p = ROOT / "runs" / "detect" / "runs" / "vehicle_plate" / "weights" / "best.pt"
    return p if p.exists() else ROOT / "best.pt"


def _clamp(v, lo, hi, default):
    try:
        return min(hi, max(lo, float(v)))
    except (TypeError, ValueError):
        return default


def create_app(pipeline, device="cpu"):
    """pipeline: any object with .process(path, batch, det_conf, accept_thr, display_name)."""
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024   # 1 GB per upload
    jobs, state = {}, {"running": None}
    jobs_lock, model_lock = threading.Lock(), threading.Lock()
    try:   # newer pipelines take extra options; a minimal stub may not
        params = inspect.signature(pipeline.process).parameters
        extra_ok = "accept_weakest" in params and "crop_mode" in params
    except (TypeError, ValueError):
        extra_ok = False

    @app.after_request
    def no_cache(resp):
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(413)
    def too_big(_e):
        return jsonify(error="The upload is too large (over 1 GB)."), 413

    def busy():
        jid = state["running"]
        return jid is not None and jobs[jid]["status"] == "running"

    # ---------------- pages / config ----------------
    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/config")
    def config():
        ver = getattr(pipeline, "verifier", None)
        reader = getattr(pipeline, "reader", None)
        return jsonify(max_images=MAX_IMAGES, device=device, defaults=DEFAULTS,
                       ml_verifier=ver.describe() if ver is not None else None,
                       ocr=reader.describe() if hasattr(reader, "describe") else None,
                       busy=busy(), running=state["running"] if busy() else None)

    # ---------------- jobs ----------------
    def run_job(job, items):
        with model_lock:
            t0 = time.time()
            for i, (path, name) in enumerate(items):
                if job["cancel"]:
                    job["status"] = "cancelled"
                    break
                job["current"] = name
                try:
                    extra = ({"accept_weakest": job["weakest"], "crop_mode": job["crop_mode"]}
                             if extra_ok else {})
                    pipeline.process(str(path), job["batch"], job["conf"], job["accept"], name,
                                     **extra)
                except Exception as e:                       # one bad image must not stop the job
                    msg = str(e) if isinstance(e, ValueError) else f"{name}: could not be processed"
                    job["errors"].append(msg)
                    traceback.print_exc()
                finally:
                    Path(path).unlink(missing_ok=True)
                job["done"] = i + 1
                job["elapsed"] = time.time() - t0
            else:
                job["status"] = "done"
            for path, _ in items:                            # leftovers after cancel
                Path(path).unlink(missing_ok=True)
            job["current"] = ""
            job["elapsed"] = time.time() - t0
            print(f"[job {job['id']}] {job['status']}: {job['done']}/{job['total']} images "
                  f"in {job['elapsed']:.1f} s")

    @app.post("/api/jobs")
    def create_job():
        with jobs_lock:
            if busy():
                return jsonify(error="Another upload is still being processed."), 409
            files = [f for f in request.files.getlist("files")
                     if f and f.filename and Path(f.filename).suffix.lower() in IMG_EXT]
            if not files:
                return jsonify(error="No image files received (jpg, jpeg, png, bmp, webp)."), 400
            skipped = max(0, len(files) - MAX_IMAGES)
            files = files[:MAX_IMAGES]
            batch = datetime.now().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]
            UPLOADS.mkdir(parents=True, exist_ok=True)
            items = []
            for k, f in enumerate(files):
                display = Path(f.filename.replace("\\", "/")).name
                safe = secure_filename(display) or "image.jpg"
                dest = UPLOADS / f"{batch}_{k:03d}_{safe}"
                f.save(dest)
                items.append((dest, display))
            jid = uuid.uuid4().hex[:12]
            job = {"id": jid, "batch": batch, "status": "running", "done": 0,
                   "total": len(items), "current": "", "errors": [], "cancel": False,
                   "elapsed": 0.0, "skipped": skipped,
                   "conf": _clamp(request.form.get("conf"), 0.05, 0.9, DEFAULTS["conf"]),
                   "accept": _clamp(request.form.get("accept"), 0.5, 0.99, DEFAULTS["accept"]),
                   "weakest": _clamp(request.form.get("weakest"), 0.3, 0.99, DEFAULTS["weakest"]),
                   "crop_mode": request.form.get("crop_mode") in ("1", "true", "on")}
            jobs[jid] = job
            state["running"] = jid
            threading.Thread(target=run_job, args=(job, items), daemon=True).start()
        return jsonify(id=jid, batch=batch, total=len(items), skipped=skipped), 202

    def job_view(job):
        return {k: job[k] for k in ("id", "batch", "status", "done", "total", "current",
                                    "errors", "elapsed", "skipped")} | {
            "summary": dl.summarize(dl.load_rows(batch=job["batch"]))}

    @app.get("/api/jobs/<jid>")
    def get_job(jid):
        job = jobs.get(jid) or abort(404)
        return jsonify(job_view(job))

    @app.post("/api/jobs/<jid>/cancel")
    def cancel_job(jid):
        job = jobs.get(jid) or abort(404)
        job["cancel"] = True
        return jsonify(ok=True)

    # ---------------- results ----------------
    def filtered():
        status = request.args.get("status") or None
        if status not in (None, "CORRECTED") + dl.STATUSES:
            status = None
        return dl.load_rows(request.args.get("q", ""), batch=request.args.get("batch") or None,
                            status=status)

    @app.get("/api/results")
    def results():
        rows = filtered()
        limit = int(_clamp(request.args.get("limit"), 1, 1000, 25))
        offset = int(_clamp(request.args.get("offset"), 0, 10 ** 9, 0))
        scope = dl.load_rows(batch=request.args.get("batch") or None)
        page = [dict(r, state_name=ipv.STATE_CODES.get(r.get("state_code") or "")
                     or ipv.LEGACY_STATE_CODES.get(r.get("state_code") or "", ""))
                for r in rows[offset:offset + limit]]
        return jsonify(rows=page, total=len(rows),
                       summary=dl.summarize(scope))

    @app.get("/api/export")
    def export():
        rows = filtered()
        tmp = Path(tempfile.mkstemp(suffix=".xlsx")[1])
        dl.export_excel(rows, out=tmp)
        name = f"platevision_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        resp = send_file(tmp, as_attachment=True, download_name=name,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        resp.call_on_close(lambda: tmp.unlink(missing_ok=True))
        return resp

    @app.post("/api/results/<row_id>/review")
    def review(row_id):
        body = request.get_json(silent=True) or {}
        text = str(body.get("corrected_text", "")).strip()
        if not text:
            return jsonify(error="Enter the correct registration number."), 400
        try:
            row, v = dl.correct(row_id, text, str(body.get("reason", ""))[:300],
                                str(body.get("reviewer", ""))[:80],
                                allow_invalid=bool(body.get("allow_invalid")))
        except KeyError:
            return jsonify(error="That result no longer exists."), 404
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(row=row, validation={k: v[k] for k in (
            "is_valid_format", "reason", "state_code", "rto_code", "series", "registration_number")})

    @app.get("/api/results/<row_id>/audit")
    def audit(row_id):
        return jsonify(history=dl.audit_history(row_id))

    @app.delete("/api/results")
    def clear():
        if busy():
            return jsonify(error="Wait for the current upload to finish first."), 409
        dl.clear()
        return jsonify(ok=True)

    # ---------------- fine-tuning the OCR model on verified plates ----------------
    tune = {"proc": None, "started": None, "finished": None, "result": ""}
    tune_log = dl.RESULTS / "finetune.log"
    ft_dir = Path(cfg.AWIROS_FINETUNED_WEIGHTS).parent

    def tune_running():
        return tune["proc"] is not None and tune["proc"].poll() is None

    def verified_samples():
        import finetune_awiros as ft
        ns = argparse.Namespace(verified=str(ft.ROOT / "training" / "verified_labels.csv"),
                                dashboard=True, labels=None)
        return len(ft.load_samples(ns))

    def watch_tuning(proc):
        proc.wait()
        tune["finished"] = time.time()
        rep_file = ft_dir / "report.json"
        try:
            import json
            rep = json.loads(rep_file.read_text())
        except Exception:
            tune["result"] = f"Training stopped (exit code {proc.returncode}); see {tune_log}."
            return
        if not rep.get("deployed"):
            o, b = rep.get("original", {}).get("exact"), (rep.get("best") or {}).get("val_exact")
            tune["result"] = (f"Not installed: the new model was not better on the held-out plates"
                              + (f" ({o:.1%} -> {b:.1%})" if o is not None and b is not None else "") + ".")
            return
        try:                                     # hot-swap the OCR model; jobs wait for the lock
            from awiros_ocr import AwirosReader
            new = AwirosReader(getattr(pipeline.reader, "device", device), weights=cfg.AWIROS_FINETUNED_WEIGHTS)
            with model_lock:
                pipeline.reader = new
                if hasattr(pipeline, "_image_cache"):
                    pipeline._image_cache.clear()
            o, b = rep["original"]["exact"], rep["best"]["val_exact"]
            tune["result"] = f"Installed: held-out exact match {o:.1%} -> {b:.1%}. New uploads use it."
        except Exception as e:
            tune["result"] = f"Trained, but loading the new model failed: {e}"

    @app.get("/api/finetune")
    def finetune_status():
        import json
        rep = None
        if (ft_dir / "report.json").is_file():
            try:
                rep = json.loads((ft_dir / "report.json").read_text())
                rep = {k: rep.get(k) for k in ("train", "val", "original", "best_epoch", "deployed")} | {
                    "best": {k: v for k, v in (rep.get("best") or {}).items() if k != "wrong"},
                    "epochs": rep.get("epochs", [])[-5:]}
            except Exception:
                rep = None
        tail = tune_log.read_text(errors="replace").splitlines()[-12:] if tune_log.is_file() else []
        reader = getattr(pipeline, "reader", None)
        return jsonify(running=tune_running(), started=tune["started"], finished=tune["finished"],
                       result=tune["result"], samples=verified_samples(),
                       min_samples=MIN_TUNE_SAMPLES, device=device, log=tail, report=rep,
                       weights=reader.describe().get("weights") if hasattr(reader, "describe") else None)

    @app.post("/api/finetune")
    def finetune_start():
        import subprocess
        import sys
        body = request.get_json(silent=True) or {}
        if tune_running():
            return jsonify(error="Fine-tuning is already running."), 409
        if busy():
            return jsonify(error="Wait for the current upload to finish first."), 409
        n = verified_samples()
        if n < MIN_TUNE_SAMPLES:
            return jsonify(error=f"Only {n} verified plates. Verify at least {MIN_TUNE_SAMPLES} "
                                 f"(dashboard corrections or label_tool.py) before fine-tuning."), 400
        if device == "cpu" and not body.get("force_cpu"):
            return jsonify(error="No GPU on this server: fine-tuning on CPU takes many hours. Run the "
                                 "server with --device cuda on the GPU PC, or confirm CPU training.",
                           needs_confirm=True), 400
        epochs = int(_clamp(body.get("epochs"), 1, 100, 30))
        cmd = [sys.executable, str(ROOT / "src" / "finetune_awiros.py"), "--device", device,
               "--epochs", str(epochs), "--deploy", "--min-samples", str(MIN_TUNE_SAMPLES)]
        f = open(tune_log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        tune.update(proc=proc, started=time.time(), finished=None, result="")
        threading.Thread(target=watch_tuning, args=(proc,), daemon=True).start()
        return jsonify(ok=True, samples=n, epochs=epochs)

    @app.get("/files/<kind>/<name>")
    def files(kind, name):
        if kind not in FILE_DIRS:
            abort(404)
        return send_from_directory(FILE_DIRS[kind], secure_filename(name))

    return app


def main():
    ap = argparse.ArgumentParser(description="PlateVision web server")
    ap.add_argument("--weights", default=str(default_weights()))
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-ml", action="store_true",
                    help="do not use the HistGradientBoosting OCR verifier (rules only)")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}\n"
                         f"Give the path: python server.py --weights path/to/best.pt")
    if args.device in ("cuda", "0"):
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available here; falling back to CPU.")
            args.device = "cpu"
    from pipeline import Pipeline
    from awiros_ocr import AwirosUnavailable
    print("Loading YOLO11 and Awiros ANPR OCR...")
    try:
        pipe = Pipeline(args.weights, args.device, use_ml=not args.no_ml)
    except AwirosUnavailable as e:
        raise SystemExit(f"Awiros ANPR OCR could not be loaded: {e}")
    o = pipe.reader.describe()
    print(f"OCR: {o['engine']} on {o['device']} ({o['weights']}), "
          f"warm-up {pipe.reader.warmup_ms:.0f} ms")
    if pipe.verifier is not None:
        print(f"ML verifier: HistGradientBoostingClassifier, threshold {pipe.verifier.threshold:.3f}")
    else:
        print("ML verifier: off (rules only). Train it with train_verifier.py.")
    app = create_app(pipe, args.device)
    print(f"PlateVision ready: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
