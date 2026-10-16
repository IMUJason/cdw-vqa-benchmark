"""Local labeling web app (stdlib only): pipes material adjudication.
Run:  python scripts/label_server.py   then open http://127.0.0.1:8099
Each item: L0 highlighted + L1 + zoom views; metal/PVC/other buttons (keys 1/2/3);
auto-save + auto-advance; resumable (saved labels are reloaded)."""
import csv, json, time, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.parse import unquote

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
JV = Path(r"datasets/plan3_arms/judge_view")
OUT_CSV = DATA / "pipes_human_labels.csv"

todo = [int(r["diag_id"]) for r in csv.DictReader(open(DATA / "pipes_to_judge.csv", encoding="utf-8-sig"))]
audit_rows = {int(r["diag_id"]): r["auto_label"] for r in
              csv.DictReader(open(DATA / "pipes_auto_accept.csv", encoding="utf-8-sig"))
              if r.get("audit_flag") == "AUDIT"}
R2 = DATA / "label_items_round2.json"
if R2.exists():
    ITEMS = json.load(open(R2))  # round-2 takes precedence when present
else:
    ITEMS = [{"id": i, "audit": False} for i in todo] + \
            [{"id": i, "audit": True, "auto": audit_rows[i]} for i in sorted(audit_rows)]

labels = {}
if OUT_CSV.exists():
    for r in csv.DictReader(open(OUT_CSV, encoding="utf-8-sig")):
        labels[int(r["diag_id"])] = r["label"]


def save_labels():
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["diag_id", "label", "ts"])
        for did, lab in sorted(labels.items()):
            w.writerow([did, lab, int(time.time())])


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>pipes 材质标注</title><style>
body{font-family:Segoe UI,Arial;margin:16px;background:#f5f5f5;text-align:center}
#bar{font-size:18px;margin:8px}#imgs{display:flex;gap:8px;justify-content:center}
#imgs img{max-width:32%;max-height:55vh;border:2px solid #999;object-fit:contain;background:#888}
.btn{font-size:22px;padding:14px 34px;margin:10px;border-radius:8px;border:0;cursor:pointer;color:#fff}
.metal{background:#455a64}.pvc{background:#1565c0}.other{background:#757575}
.nav{background:#9e9e9e}.audit{color:#b71c1c;font-weight:bold}
</style></head><body>
<div id="bar"></div>
<div id="imgs">
<img id="im0" title="L0 原图（红框=目标）"><img id="im1" title="L1 背景抹除"><img id="im2" title="放大视图">
</div>
<div>
<button class="btn metal" onclick="pick('metal')">metal 金属 (按1)</button>
<button class="btn pvc" onclick="pick('PVC')">PVC 塑料 (按2)</button>
<button class="btn other" onclick="pick('other')">other 拿不准 (按3)</button>
</div>
<div>
<button class="btn nav" onclick="go(-1)">← 上一个</button>
<button class="btn nav" onclick="go(1)">下一个 →</button>
<span id="saved" style="margin-left:20px;color:#2e7d32"></span>
</div>
<script>
let items=[], cur=0;
async function load(){let r=await fetch('/state'); let d=await r.json();
 items=d.items; labels=d.labels; let first=d.first; cur=first; show();}
function show(){let it=items[cur];
 document.getElementById('im0').src='/img/'+it.id+'_L0h.jpg?'+Date.now();
 document.getElementById('im1').src='/img/'+it.id+'_L1.jpg?'+Date.now();
 document.getElementById('im2').src='/img/'+it.id+'_Z.jpg?'+Date.now();
 let lab=labels[it.id];
 document.getElementById('bar').innerHTML='#'+it.id+'（第 '+(cur+1)+'/'+items.length+' 个）'
  +(it.audit?' <span class=audit>[审计样本 自动判:'+it.auto+']</span>':'')
  +'　已标: <b>'+(lab||'未标')+'</b>　总进度 '+Object.keys(labels).length+'/'+items.length;}
async function pick(v){let it=items[cur]; labels[it.id]=v;
 await fetch('/save?id='+it.id+'&label='+v);
 document.getElementById('saved').textContent='已保存 ✓';
 if(cur<items.length-1){cur++;} show();}
function go(d){cur=Math.min(Math.max(cur+d,0),items.length-1); show();}
document.addEventListener('keydown',e=>{if(e.key=='1')pick('metal');
 if(e.key=='2')pick('PVC'); if(e.key=='3')pick('other');
 if(e.key=='ArrowLeft')go(-1); if(e.key=='ArrowRight')go(1);});
load();</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            b = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(b)
        elif u.path == "/state":
            first = next((i for i, it in enumerate(ITEMS) if it["id"] not in labels), len(ITEMS) - 1)
            b = json.dumps({"items": ITEMS, "labels": labels, "first": first}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b)
        elif u.path.startswith("/img/"):
            name = unquote(u.path[5:])
            p = JV / name
            if not p.exists():
                self.send_error(404); return
            b = p.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers()
            self.wfile.write(b)
        elif u.path == "/save":
            qs = parse_qs(u.query)
            labels[int(qs["id"][0])] = qs["label"][0]
            save_labels()
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_error(404)


if __name__ == "__main__":
    print("open http://127.0.0.1:8099  (items:", len(ITEMS), ")")
    ThreadingHTTPServer(("127.0.0.1", 8099), H).serve_forever()
