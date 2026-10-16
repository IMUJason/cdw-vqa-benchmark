"""Human baseline answer web app: image + question + choice buttons.
Run: python scripts/baseline_server.py  then open http://127.0.0.1:8098
Answers saved incrementally to data/human_baseline.csv (resumable)."""
import csv, json, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
ARMS = Path(r"datasets/plan3_arms")
SCENE_CACHE = Path(r"datasets/plan3_scene_cache")
OUT_CSV = DATA / "human_baseline.csv"

ITEMS = json.load(open(DATA / "human_baseline_items.json", encoding="utf-8"))

ZH_CLASS = {"brick": "砖块", "concrete": "混凝土", "tile": "瓷砖", "stone": "石块",
            "wood": "木材", "gypsum_board": "石膏板", "pipes": "管件", "plastic": "塑料",
            "foam": "泡沫", "general_w": "一般混杂废料", "cardboard": "纸板",
            "fill_dirt": "填土", "hard_plastic": "硬质塑料", "soft_plastic": "软质塑料",
            "steel": "钢材", "fabric": "织物", "plasterboard": "石膏板"}
ZH_LOC = {"top-left": "左上", "top-center": "中上", "top-right": "右上",
          "middle-left": "中左", "center": "中央", "middle-center": "中部",
          "middle-right": "中右", "bottom-left": "左下", "bottom-center": "中下",
          "bottom-right": "右下"}


def zh_of(it):
    import re
    q = it["question"]
    m = re.search(r"Is there any ([a-z_ ]+) visible", q)
    if m:
        cls_key = m.group(1).strip().replace(" ", "_")
        return f"图中这堆废料里能看到「{ZH_CLASS.get(cls_key, m.group(1))}」吗？"
    if "How many distinct" in q:
        return "图中可以看到几类不同的废料材质？"
    if "Which material dominates" in q:
        return "这堆废料中，可见面积占比最大的材质是哪种？"
    if "non-mineral" in q:
        return "这堆废料中是否含有非矿物类材质（如木材、塑料、织物、纸板）？"
    m = re.search(r"the fragment at ([a-z-]+)", q)
    if m:
        return f"位于{ZH_LOC.get(m.group(1), m.group(1))}的那块碎片，主要由什么材质制成？"
    if "isolated fragment" in q:
        return "图中这块孤立的碎片，主要由什么材质制成？"
    if "What type of waste object" in q:
        m = re.search(r"at ([a-z-]+)", q)
        if m:
            return f"位于{ZH_LOC.get(m.group(1), m.group(1))}的那块物体，属于哪类废料？"
        return "图中这块孤立的物体，属于哪类废料？"
    return q


def img_path(it):
    if "diag_id" not in it:  # scene item
        p = SCENE_CACHE / (Path(it["image_path"]).stem + ".jpg")
        return p if p.exists() else Path(it["image_path"])
    return ARMS / it["image"]


for it in ITEMS:
    it["_img"] = str(img_path(it))
    it["question_zh"] = zh_of(it)
    if it.get("qtype") == "mcq" and "choices" in it:
        it["choices_zh"] = {k: ZH_CLASS.get(str(v).replace(" ", "_"), v)
                            for k, v in it["choices"].items()}

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
<title>人类基线作答</title><style>
body{font-family:Segoe UI,Arial;margin:16px;background:#f5f5f5;text-align:center}
#bar{font-size:17px;margin:6px}
#main{display:flex;gap:14px;justify-content:center;align-items:flex-start}
#main img{max-width:46%;max-height:58vh;border:2px solid #999;object-fit:contain;background:#888}
#q{font-size:19px;text-align:left;max-width:46%;line-height:1.5}
.btn{display:block;width:100%;font-size:19px;padding:12px;margin:8px 0;border-radius:8px;border:1px solid #666;background:#fff;cursor:pointer;text-align:left}
.btn:hover{background:#e3f2fd}
.nav{background:#eee;font-size:15px;padding:8px 22px;display:inline-block;width:auto}
#txtin{font-size:18px;padding:8px;width:90%}
</style></head><body>
<div id="bar"></div>
<div id="main"><img id="im"><div id="q"></div></div>
<div style="margin-top:10px">
<button class="btn nav" onclick="go(-1)">← 上一个</button>
<button class="btn nav" onclick="go(1)">下一个 →</button>
<span id="saved" style="margin-left:20px;color:#2e7d32"></span>
</div>
<script>
let items=[], cur=0, answers={};
async function load(){let r=await fetch('/state'); let d=await r.json();
 items=d.items; answers=d.answers; cur=d.first; show();}
function show(){let it=items[cur];
 document.getElementById('im').src='/img?i='+cur+'&t='+Date.now();
 let q=it.question_zh||it.question;
 let html='<b>题：</b> '+q;
 let qt=it.qtype;
 if(qt==='yesno'){html+='<button class="btn" onclick="pick(\\'Yes\\')">是（Yes）</button><button class="btn" onclick="pick(\\'No\\')">否（No）</button>';}
 else if(qt==='mcq'){let ch=it.choices_zh||it.choices;for(const[k,v]of Object.entries(ch)){html+='<button class="btn" onclick="pick(\\''+k+'\\')">'+k+'. '+v+'</button>';}}
 else{html+='<br><input id="txtin" placeholder="输入简短答案后回车" onkeydown="if(event.key==\\'Enter\\'&&this.value.trim())pick(this.value.trim())">';}
 html+='<div style="color:#888;font-size:14px;margin-top:8px">已答: '+(answers[it.uid]||'—')+'</div>';
 document.getElementById('q').innerHTML=html;
 document.getElementById('bar').innerHTML='第 '+(cur+1)+' / '+items.length+' 题　总进度 '+Object.keys(answers).length+'/'+items.length;}
async function pick(v){let it=items[cur]; answers[it.uid]=v;
 await fetch('/save?uid='+encodeURIComponent(it.uid)+'&a='+encodeURIComponent(v));
 document.getElementById('saved').textContent='已保存 ✓';
 if(cur<items.length-1){cur++;} show();}
function go(d){cur=Math.min(Math.max(cur+d,0),items.length-1); show();}
document.addEventListener('keydown',e=>{let t=e.target;if(t&&t.id==='txtin')return;
 let it=items[cur];if(!it)return;
 if(it.qtype==='yesno'){if(e.key==='1')pick('Yes');if(e.key==='2')pick('No');}
 if(it.qtype==='mcq'){let k=e.key.toUpperCase();if(['A','B','C','D'].includes(k))pick(k);}
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
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
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
    print("open http://127.0.0.1:8098  (items:", len(ITEMS), ")")
    ThreadingHTTPServer(("127.0.0.1", 8098), H).serve_forever()
