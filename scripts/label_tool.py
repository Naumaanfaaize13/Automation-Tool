"""Verify plate labels by hand before fine-tuning the OCR model.

The manifest labels are unaudited (most came from the old PARSeq system, an LLM or a
"high confidence" filter). Training on wrong labels teaches the model wrong answers, so
finetune_awiros.py only uses labels confirmed here (plus dashboard corrections).

    python scripts/label_tool.py --manifest ~/Desktop/plates/car/manifest.csv \
        --manifest ~/Desktop/plates/bike/manifest.csv --manifest ~/Desktop/plates/others/manifest.csv \
        --folder ~/Desktop/correct
    open http://127.0.0.1:5001

Plates where the current model disagrees with the dataset label are shown first: those are
the ones the model gets wrong, i.e. the useful training examples.
Keys: Enter = save, Alt+U = unreadable, Alt+S = skip, Alt+B = back, Alt+1 / Alt+2 = use the
dataset label / the model read.
Results: training/verified_labels.csv (path,label,status,reviewer,time; the last row per path wins).
"""
import argparse
import csv
import json
import threading
import time
from pathlib import Path
import sys  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]  # core modules live in src/

import cv2
from flask import Flask, jsonify, request, send_file

import indian_plate_validator as ipv

OUT = ROOT / "training" / "verified_labels.csv"
PRED_CACHE = ROOT / "training" / "model_reads.json"
FIELDS = ["path", "label", "status", "reviewer", "time"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_items(manifests, folders):
    items, seen = [], set()
    for m in manifests or []:
        m = Path(m).expanduser()
        for r in csv.DictReader(open(m, newline="", encoding="utf-8")):
            p = str((m.parent / r["path"]).resolve())
            if p in seen or not Path(p).is_file():
                continue
            seen.add(p)
            items.append({"path": p, "dataset_label": ipv.normalize(r.get("plate_number", "")),
                          "source": r.get("plate_source", ""), "vehicle": r.get("vehicle_type", "")})
    for f in folders or []:
        for p in sorted(Path(f).expanduser().iterdir()):
            if p.suffix.lower() in IMG_EXT and str(p.resolve()) not in seen:
                seen.add(str(p.resolve()))
                items.append({"path": str(p.resolve()), "dataset_label": "", "source": "folder",
                              "vehicle": ""})
    return items


def load_verified():
    done = {}
    if OUT.is_file():
        for r in csv.DictReader(open(OUT, newline="", encoding="utf-8")):
            done[r["path"]] = r
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append")
    ap.add_argument("--folder", action="append")
    ap.add_argument("--reviewer", default="reviewer")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    items = load_items(args.manifest, args.folder)
    if not items:
        raise SystemExit("No images. Give --manifest and/or --folder.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    preds = json.loads(PRED_CACHE.read_text()) if PRED_CACHE.is_file() else {}
    state = {"reading": True}
    lock = threading.Lock()

    def read_all():                      # model reads, used to put disagreements first
        from awiros_ocr import AwirosReader
        reader = AwirosReader("cpu")
        todo = [it for it in items if it["path"] not in preds]
        for s in range(0, len(todo), 16):
            chunk = todo[s:s + 16]
            imgs = [cv2.imread(it["path"]) for it in chunk]
            for it, r in zip(chunk, reader.read_chars(imgs)):      # unreadable file -> empty read
                with lock:
                    preds[it["path"]] = {"text": ipv.normalize(r["raw"]), "avg": round(r["avg"], 4),
                                         "weakest": round(r["weakest"], 4)}
            PRED_CACHE.write_text(json.dumps(preds))
        state["reading"] = False

    threading.Thread(target=read_all, daemon=True).start()
    app = Flask(__name__)

    def queue():
        done = load_verified()
        with lock:
            p = dict(preds)

        def key(it):
            pr = p.get(it["path"])
            if pr is None:
                return (2, 0)
            disagree = pr["text"] != it["dataset_label"]
            return (0 if disagree else 1, pr["avg"])     # disagreements, least confident first
        return [it for it in sorted(items, key=key) if it["path"] not in done], done, p

    @app.get("/api/next")
    def nxt():
        skip = set(request.args.get("skip", "").split("|")) - {""}
        q, done, p = queue()
        q = [it for it in q if it["path"] not in skip]
        stats = {"total": len(items), "verified": len(done), "remaining": len(q),
                 "model_reading": state["reading"], "model_read": len(p)}
        if not q:
            return jsonify(item=None, stats=stats)
        it = dict(q[0], model=p.get(q[0]["path"]))
        return jsonify(item=it, stats=stats)

    @app.get("/api/item")
    def item():
        path = request.args["path"]
        it = next(i for i in items if i["path"] == path)
        with lock:
            return jsonify(item=dict(it, model=preds.get(path), saved=load_verified().get(path)))

    @app.get("/img")
    def img():
        path = request.args["path"]
        if path not in {i["path"] for i in items}:
            return "not found", 404
        return send_file(path)

    @app.post("/api/save")
    def save():
        d = request.get_json(force=True)
        label = ipv.normalize(d.get("label", "")) if d["status"] == "ok" else ""
        if d["status"] == "ok" and not label:
            return jsonify(error="empty label"), 400
        new = not OUT.is_file()
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow({"path": d["path"], "label": label, "status": d["status"],
                        "reviewer": args.reviewer, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        v = ipv.validate(label) if label else {"is_valid_format": False}
        return jsonify(ok=True, valid=v["is_valid_format"])

    @app.get("/")
    def index():
        return PAGE

    print(f"{len(items)} images. Open http://127.0.0.1:{args.port}  (results: {OUT})")
    app.run(host="127.0.0.1", port=args.port, threaded=True)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Plate label check</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#141a26;--card:#1d2535;--fg:#e6ebf5;--mut:#8f9bb3;--acc:#f5a524;--ok:#3ccf7a;--bad:#ef5b5b}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
main{max-width:900px;margin:0 auto;padding:16px}
.card{background:var(--card);border-radius:10px;padding:16px;margin-bottom:12px}
img{max-width:100%;max-height:340px;display:block;margin:0 auto;border-radius:6px;image-rendering:auto}
input{font:600 30px ui-monospace,monospace;letter-spacing:3px;width:100%;box-sizing:border-box;
 padding:10px;border-radius:8px;border:2px solid #3a4560;background:#0f1420;color:var(--fg);text-transform:uppercase}
input.valid{border-color:var(--ok)} input.invalid{border-color:var(--bad)}
button{font:600 15px system-ui;padding:10px 16px;border-radius:8px;border:0;cursor:pointer;margin:6px 6px 0 0}
.save{background:var(--ok);color:#06210f}.unr{background:var(--bad);color:#fff}.sk{background:#3a4560;color:var(--fg)}
.chip{display:inline-block;font:600 18px ui-monospace,monospace;padding:6px 10px;border-radius:6px;background:#0f1420;
 border:1px solid #3a4560;cursor:pointer;margin:4px 8px 4px 0}
.mut{color:var(--mut);font-size:13px}.row{display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.bar{height:6px;background:#0f1420;border-radius:3px;overflow:hidden}.bar>div{height:100%;background:var(--acc)}
</style></head><body><main>
<div class="card"><div class="row" style="justify-content:space-between"><b>Plate label check</b>
<span class="mut" id="stats"></span></div><div class="bar" style="margin-top:8px"><div id="prog"></div></div></div>
<div class="card" id="box"><p class="mut">Loading...</p></div>
<div class="card mut">Enter = save &middot; Alt+U = unreadable (excluded from training) &middot; Alt+S = skip
&middot; Alt+B = back &middot; Alt+1 = dataset label &middot; Alt+2 = model read. Type exactly what is printed on the plate.</div>
</main><script>
let cur=null, skipped=[], history=[];
const $=id=>document.getElementById(id);
const VALID=/^([A-Z]{2}\\d{1,2}[A-Z]{0,3}\\d{4}|\\d{2}BH\\d{4}[A-Z]{1,2})$/;
function check(){const i=$('lab'); if(!i) return; const v=i.value.toUpperCase().replace(/[^A-Z0-9]/g,'');
 i.className=v?(VALID.test(v)?'valid':'invalid'):'';}
function show(it,stats){
 if(stats){$('stats').textContent=`${stats.verified} verified / ${stats.total}  ·  ${stats.remaining} left`+
  (stats.model_reading?`  ·  model reading ${stats.model_read}/${stats.total} (disagreements first once done)`:'');
  $('prog').style.width=(100*stats.verified/stats.total)+'%';}
 cur=it; if(!it){$('box').innerHTML='<p>All images checked. Run <code>python src/finetune_awiros.py</code>.</p>';return;}
 const m=it.model||{}; const pre=(it.saved&&it.saved.label)||m.text||it.dataset_label||'';
 $('box').innerHTML=`<img src="/img?path=${encodeURIComponent(it.path)}">
 <p class="mut" style="word-break:break-all">${it.path.split('/').slice(-2).join('/')} · source: ${it.source||'-'} ${it.vehicle||''}</p>
 <div>${it.dataset_label?`<span class="chip" id="c1" title="1">1 · dataset: ${it.dataset_label}</span>`:''}
 ${m.text?`<span class="chip" id="c2" title="2">2 · model: ${m.text} <span class="mut">${Math.round(100*m.avg)}% / weakest ${Math.round(100*m.weakest)}%</span></span>`:''}</div>
 <input id="lab" value="${pre}" autocomplete="off" spellcheck="false">
 <div><button class="save" id="bs">Save (Enter)</button><button class="unr" id="bu">Unreadable (Alt+U)</button>
 <button class="sk" id="bk">Skip (Alt+S)</button><button class="sk" id="bb">Back (Alt+B)</button></div>`;
 const i=$('lab'); i.focus(); i.select(); i.oninput=check; check();
 if($('c1')) $('c1').onclick=()=>{i.value=it.dataset_label;check();i.focus()};
 if($('c2')) $('c2').onclick=()=>{i.value=m.text;check();i.focus()};
 $('bs').onclick=()=>save('ok'); $('bu').onclick=()=>save('unreadable'); $('bk').onclick=skip; $('bb').onclick=back;}
async function next(){const r=await fetch('/api/next?skip='+encodeURIComponent(skipped.join('|')));const j=await r.json();show(j.item,j.stats);}
async function save(status){if(!cur)return; const label=$('lab').value;
 const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({path:cur.path,label,status})}); if(!r.ok){alert((await r.json()).error);return;}
 history.push(cur.path); next();}
function skip(){if(cur){skipped.push(cur.path);history.push(cur.path);} next();}
async function back(){const p=history.pop(); if(!p)return; skipped=skipped.filter(x=>x!==p);
 const j=await (await fetch('/api/item?path='+encodeURIComponent(p))).json(); show(j.item);}
document.addEventListener('keydown',e=>{const inInput=e.target.id==='lab';
 if(e.key==='Enter'){e.preventDefault();save('ok');return;}
 if(inInput && !e.altKey) { if(e.key==='Escape') e.target.blur(); return; }
 const k=e.key.toLowerCase(); if(['u','s','b','1','2'].includes(k)) e.preventDefault();
 if(k==='u')save('unreadable'); else if(k==='s')skip(); else if(k==='b')back();
 else if(k==='1'&&$('c1'))$('c1').click(); else if(k==='2'&&$('c2'))$('c2').click();});
next(); setInterval(()=>{if(!cur)next()},5000);
</script></body></html>"""

if __name__ == "__main__":
    main()
