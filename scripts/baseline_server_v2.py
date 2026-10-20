"""Human baseline v2 answer app: balanced pipes, forced three-way choice.
Run: python scripts/baseline_server_v2.py  then open http://127.0.0.1:8099
Answers saved incrementally to data/human_baseline_v2.csv (resumable).
Keyboard: 1 = PVC/塑料, 2 = 金属, 3 = 其他; ←/→ navigate."""
import csv, json, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
ARMS = Path(r"datasets/plan3_arms")
OUT_CSV = DATA / "human_baseline_v2.csv"

ITEMS = json.load(open(DATA / "human_baseline_v2_items.json", encoding="utf-8"))
ZH_ARM = {"L0": "原位（整图）", "L1": "背景抹除", "L2": "背景模糊", "L3": "裁剪孤立"}

for it in ITEMS:
    it["_img"] = str(ARMS / it["image"])
    it["question_zh"] = ("这个碎片主要由什么材料构成？（注意：请判断材质，"
                         "不是物体名称）" + f"　[视图: {ZH_ARM[it['arm']]}]")

answers = {}
if OUT_CSV.exists():
    for r in csv.DictReader(open(OUT_CSV, encoding="utf-8-sig")):
        answers[r["uid"]] = r["answer"]


def save():
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["uid", "answer", "ts"])
        for uid, a in sorted(answers.items()):
            w.writerow([uid, a, int(time.time())])


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>人类基线 v2（平衡细粒度）</title><style>
body{font-family:Segoe UI,Arial;margin:16px;background:#f5f5f5;text-align:center}
#bar{font-size:17px;margin:6px}
#main{display:flex;gap:14px;justify-content:center;align-items:flex-start}
#main img{max-width:52%;max-height:60vh;border:2px solid #999;object-fit:contain;background:#888}
#q{font-size:19px;text-align:left;max-width:44%;line-height:1.5}
.btn{display:block;width:100%;font-size:19px;padding:12px;margin:8px 0;border-radius:8px;border:1px solid #666;background:#fff;cursor:pointer;text-align:left}
.btn:hover{background:#e3f2fd}
.nav{background:#eee;font-size:15px;padding:8px 22px;display:inline-block;width:auto}
</style></head><body>
<div id="bar"></div>
<div id="main"><img id="im"><div id="q"></div></div>
<div style="margin-top:10px">
<button class="btn nav" onclick="go(-1)">← 上一个</button>
<button class="btn nav" onclick="go(1)">下一个 →</button>
<span id="saved" style="margin-left:20px;color:#2e7d32"></span>
</div>
<script>
let items=[],cur=0,answers={};
async function load(){let r=await fetch('/state');let d=await r.json();
 items=d.items;answers=d.answers;cur=d.first;show();}
function show(){let it=items[cur];
 document.getElementById('im').src='/img?i='+cur+'&t='+Date.now();
 let html='<b>题：</b> '+it.question_zh;
 html+='<button class="btn" onclick="pick(\\'PVC\\')">1. PVC / 塑料</button>';
 html+='<button class="btn" onclick="pick(\\'metal\\')">2. 金属</button>';
 html+='<button class="btn" onclick="pick(\\'other\\')">3. 其他</button>';
 html+='<div style="color:#888;font-size:14px;margin-top:8px">已答: '+(answers[it.uid]||'—')+'</div>';
 document.getElementById('q').innerHTML=html;
 document.getElementById('bar').innerHTML='第 '+(cur+1)+' / '+items.length+' 题　总进度 '+Object.keys(answers).length+'/'+items.length;}
async function pick(v){let it=items[cur];answers[it.uid]=v;
 await fetch('/save?uid='+encodeURIComponent(it.uid)+'&a='+encodeURIComponent(v));
 document.getElementById('saved').textContent='已保存 ✓';
 if(cur<items.length-1){cur++;}show();}
function go(d){cur=Math.min(Math.max(cur+d,0),items.length-1);show();}
document.addEventListener('keydown',e=>{
 if(e.key==='1')pick('PVC');if(e.key==='2')pick('metal');if(e.key==='3')pick('other');
 if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1);});
load();</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            b = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(b)
        elif u.path == "/state":
            first = next((i for i, it in enumerate(ITEMS) if it["uid"] not in answers), len(ITEMS) - 1)
            b = json.dumps({"items": ITEMS, "answers": answers, "first": first}, ensure_ascii=False).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers()
            self.wfile.write(b)
        elif u.path == "/img":
            i = int(parse_qs(u.query)["i"][0])
            p = Path(ITEMS[i]["_img"])
            if not p.exists():
                self.send_error(404); return
            b = p.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers()
            self.wfile.write(b)
        elif u.path == "/save":
            qs = parse_qs(u.query)
            answers[qs["uid"][0]] = qs["a"][0]
            save()
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_error(404)


if __name__ == "__main__":
    print("open http://127.0.0.1:8099  (items:", len(ITEMS), ")")
    ThreadingHTTPServer(("127.0.0.1", 8099), H).serve_forever()
