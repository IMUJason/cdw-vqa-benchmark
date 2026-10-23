"""Remote multi-annotator labeling server (stdlib only; Linux deploy OK).

Deploys the balanced fine-grained pipes baseline (200 items, 3-way choice)
so that multiple annotators can label independently from browsers.
Each annotator logs in with a name + access code; answers are saved to
answers/<name>.csv (resume-safe). Item order is shuffled per annotator
(deterministic by name) to avoid copying bias; the underlying question set
is identical for everyone. All data stays on your server.

Usage:  python3 baseline_server_remote.py [port]
Then open http://<server-ip>:<port>/  and log in.
"""
import csv, json, random, re, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE = Path(__file__).resolve().parent
ITEMS = json.load(open(BASE / "items" / "items.json", encoding="utf-8"))
ARMS = BASE / "items"          # images live beside items.json as <image>.jpg
OUTDIR = BASE / "answers"
OUTDIR.mkdir(exist_ok=True)

ACCESS_CODE = "cdw2026"        # change before deploying; give it only to your annotators
ZH_ARM = {"L0": "原位（整图）", "L1": "背景抹除", "L2": "背景模糊", "L3": "裁剪孤立"}

SAFE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff_-]+")


def safe_name(name):
    n = SAFE.sub("", str(name).strip())[:24]
    return n if len(n) >= 2 else None


def order_for(name):
    import hashlib
    seed = int(hashlib.md5(("cdwvqa|" + name).encode("utf-8")).hexdigest()[:8], 16)
    idx = list(range(len(ITEMS)))
    random.Random(seed).shuffle(idx)
    return idx


def answers_path(name):
    return OUTDIR / f"answers_{safe_name(name)}.csv"


def load_answers(name):
    p = answers_path(name)
    if not p.exists():
        return {}
    return {r["uid"]: r["answer"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))}


def save_answer(name, uid, ans):
    p = answers_path(name)
    answers = load_answers(name)
    # keep the ORIGINAL timestamp of the first answer to this uid
    prev = {}
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            prev[r["uid"]] = r.get("ts", "")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["uid", "answer", "ts"])
        for u, a in sorted({**answers, uid: ans}.items()):
            w.writerow([u, a, prev.get(u) or int(time.time())])


def log_login(name, action="login"):
    with open(OUTDIR / "logins.csv", "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([name, action, int(time.time())])


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>材质判别标注</title><style>
body{font-family:system-ui,Segoe UI,Arial;margin:14px;background:#f5f5f5;text-align:center}
#bar{font-size:16px;margin:4px}
#main{display:flex;gap:12px;justify-content:center;align-items:flex-start;flex-wrap:wrap}
#main img{max-width:94%;max-width:min(94vw,720px);max-height:56vh;border:2px solid #999;object-fit:contain;background:#888}
#q{font-size:18px;text-align:left;max-width:46%;min-width:290px;line-height:1.5}
.btn{display:block;width:100%;font-size:18px;padding:11px;margin:7px 0;border-radius:8px;border:1px solid #666;background:#fff;cursor:pointer;text-align:left}
.btn:hover{background:#e3f2fd}
.nav{background:#eee;font-size:14px;padding:7px 18px;display:inline-block;width:auto}
#login{max-width:420px;margin:60px auto;background:#fff;padding:26px;border-radius:12px;border:1px solid #ccc;text-align:left}
#login input{width:100%;font-size:17px;padding:9px;margin:6px 0;box-sizing:border-box}
.note{color:#666;font-size:14px;line-height:1.5}
h2{margin:0 0 8px}
</style></head><body>
<div id="login"><h2>材质判别标注</h2>
<div class="note">判断碎片<b>材质</b>（不是物体名称）。参考：白色/绿色/红色管段通常为 PVC；黑色管段有概率是金属；金属与 PVC 接头形状一般不同。凭视觉判断即可，不确定时选你认为更可能的。</div>
<input id="nm" placeholder="你的姓名（用于分开保存，可中文）">
<input id="cd" placeholder="访问码" type="password">
<button class="btn" style="text-align:center" onclick="login()">进入标注</button>
<div id="lerr" style="color:#c00;font-size:14px"></div>
</div>
<div id="app" style="display:none">
<div id="bar"></div>
<div id="main"><img id="im"><div id="q"></div></div>
<div style="margin-top:8px">
<button class="btn nav" onclick="go(-1)">← 上一题</button>
<button class="btn nav" onclick="go(1)">下一题 →</button>
<span id="saved" style="margin-left:14px;color:#2e7d32"></span></div>
</div>
<script>
let items=[],cur=0,ans={},name='';
function login(){
 name=document.getElementById('nm').value.trim();
 const code=document.getElementById('cd').value;
 if(name.length<2){document.getElementById('lerr').textContent='请输入姓名（至少 2 个字）';return;}
 fetch('/state?name='+encodeURIComponent(name)+'&code='+encodeURIComponent(code))
 .then(r=>r.json()).then(d=>{
  if(d.err){document.getElementById('lerr').textContent=d.err;return;}
  localStorage.setItem('cdwname',name);items=d.items;ans=d.answers;cur=d.first;
  document.getElementById('login').style.display='none';
  document.getElementById('app').style.display='block';show();})
 .catch(()=>document.getElementById('lerr').textContent='网络错误');}
function show(){let it=items[cur];
 document.getElementById('im').src='/img?i='+cur+'&name='+encodeURIComponent(name)+'&t='+Date.now();
 let h='<b>题：</b>这个碎片主要由什么材料构成？<span style="color:#888">（判断材质，不是物体名）</span>　['+it.arm_zh+']';
 h+='<button class="btn" onclick="pick(\\'PVC\\')">1. PVC / 塑料</button>';
 h+='<button class="btn" onclick="pick(\\'metal\\')">2. 金属</button>';
 h+='<button class="btn" onclick="pick(\\'other\\')">3. 其他</button>';
 h+='<div style="color:#888;font-size:13px;margin-top:6px">已答: '+(ans[it.uid]||'—')+'</div>';
 document.getElementById('q').innerHTML=html=h;
 document.getElementById('bar').innerHTML='第 '+(cur+1)+' / '+items.length+' 题　总进度 '+Object.keys(ans).length+'/'+items.length;}
async function pick(v){let it=items[cur];ans[it.uid]=v;
 await fetch('/save?name='+encodeURIComponent(name)+'&uid='+encodeURIComponent(it.uid)+'&a='+encodeURIComponent(v));
 document.getElementById('saved').textContent='已保存 ✓';
 if(cur<items.length-1){cur++;}show();}
function go(d){cur=Math.min(Math.max(cur+d,0),items.length-1);show();}
document.addEventListener('keydown',e=>{
 if(e.key==='1')pick('PVC');if(e.key==='2')pick('metal');if(e.key==='3')pick('other');
 if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1);});
const savedName=localStorage.getItem('cdwname');
if(savedName){document.getElementById('nm').value=savedName;}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/state":
            name = safe_name(q.get("name", [""])[0])
            if not name:
                return self._json({"err": "姓名无效"}, 400)
            if q.get("code", [""])[0] != ACCESS_CODE:
                return self._json({"err": "访问码不正确"}, 403)
            items = [dict(ITEMS[i], arm_zh=ZH_ARM[ITEMS[i]["arm"]]) for i in order_for(name)]
            log_login(name)
            answers = load_answers(name)
            first = next((k for k, it in enumerate(items) if it["uid"] not in answers),
                         len(items) - 1)
            self._json({"items": items, "answers": answers, "first": first})
        elif u.path == "/img":
            i = int(q["i"][0])
            name = safe_name(q.get("name", [""])[0])
            if not name:
                self.send_error(400, "name required")
                return
            idx = order_for(name)
            if i < 0 or i >= len(idx):
                self.send_error(400, "bad index")
                return
            it = ITEMS[idx[i]]
            p = ARMS / it["image"]
            if not p.exists():
                self.send_error(404)
                return
            b = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/save":
            name = safe_name(q.get("name", [""])[0])
            uid, a = q.get("uid", [""])[0], q.get("a", [""])[0]
            if not name or not uid or a not in ("PVC", "metal", "other"):
                return self._json({"err": "bad"}, 400)
            save_answer(name, uid, a)
            self._json({"ok": True})
        else:
            self.send_error(404)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"serving on 0.0.0.0:{port}  (items: {len(ITEMS)}, access code set)")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
