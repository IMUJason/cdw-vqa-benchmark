"""Run the GLM QLoRA training as a detached process with checkpoints.

Launches train_qlora_glm.py via PowerShell Start-Process so the run is not a
child of this shell and survives session interruptions; the trainer saves a
LoRA checkpoint every 100 optimizer steps for the same reason.
"""
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PY = r"~/anaconda3/envs/env310/python.exe"
LOG = BASE / "results" / "logs" / "glm_v2_train.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

ps = f"""
$env:OUT_DIR = 'models/adapters/glm41v-cdw-lora-r16-v2'
$env:PYTHONUNBUFFERED = '1'
Start-Process -FilePath '{PY}' `
  -ArgumentList 'scripts/train_qlora_glm.py' `
  -WorkingDirectory '.' `
  -RedirectStandardOutput '{LOG.as_posix()}' `
  -RedirectStandardError '{LOG.as_posix()}.err' `
  -WindowStyle Hidden -Priority BelowNormal
"""
r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                   capture_output=True, text=True)
print("launch rc:", r.returncode, r.stdout[-200:], r.stderr[-300:])
print("log:", LOG)
