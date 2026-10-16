"""Unified vision-chat clients for the evaluation arm.
Local HF models + API models (Kimi OpenAI-style, GLM/DeepSeek Anthropic-style,
GPT relay). Secrets read from the out-of-sync secrets.env."""
from __future__ import annotations
import base64, json
from pathlib import Path

SECRETS = Path(r"secrets.env")


def load_secrets() -> dict:
    return dict(l.strip().split("=", 1)
                for l in SECRETS.read_text().splitlines()
                if "=" in l and not l.startswith("#"))


def img_data_url(path: str | Path, max_side: int = 1568) -> str:
    """Read image, downscale, return base64 data URL (jpeg)."""
    import cv2
    img = cv2.imdecode(__import__("numpy").fromfile(str(path), dtype=__import__("numpy").uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class KimiVision:
    name = "kimi_k3"

    def __init__(self, model: str = "k3"):
        from openai import OpenAI
        kv = load_secrets()
        self.model = model
        self.client = OpenAI(api_key=kv["KIMI_API_KEY"],
                             base_url=kv["KIMI_BASE_URL"] + "/v1")

    def ask(self, question: str, image: str | Path | None = None,
            max_tokens: int = 2048) -> str:
        content = ([{"type": "image_url", "image_url": {"url": img_data_url(image)}},
                    {"type": "text", "text": question}] if image
                   else [{"type": "text", "text": question}])
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}])
        return (r.choices[0].message.content or "").strip()


class AnthropicVision:
    """GLM / DeepSeek Anthropic-compatible endpoints."""

    def __init__(self, provider: str = "glm"):
        import anthropic  # noqa: F401 -- sdk optional; fall back to raw http
        kv = load_secrets()
        self.kv = kv
        self.provider = provider
        if provider == "glm":
            self.base, self.key = kv["GLM_BASE_URL"], kv["GLM_API_KEY"]
            self.model = kv.get("GLM_MODEL", "glm-5.3")
        else:
            self.base, self.key = kv["DEEPSEEK_BASE_URL"], kv["DEEPSEEK_API_KEY"]
            self.model = kv.get("DEEPSEEK_MODEL", "deepseek-flash")
        self.name = f"{provider}_{self.model}"

    def ask(self, question: str, image: str | Path | None = None,
            max_tokens: int = 2048) -> str:
        import requests
        blocks = []
        if image:
            url = img_data_url(image)
            b64 = url.split(",", 1)[1]
            blocks.append({"type": "image",
                           "source": {"type": "base64",
                                      "media_type": "image/jpeg", "data": b64}})
        blocks.append({"type": "text", "text": question})
        r = requests.post(self.base + "/v1/messages",
                          headers={"x-api-key": self.key,
                                   "anthropic-version": "2023-06-01"},
                          json={"model": self.model, "max_tokens": max_tokens,
                                "messages": [{"role": "user", "content": blocks}]},
                          timeout=300)
        r.raise_for_status()
        data = r.json()
        return "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text").strip()


class LocalQwenVL:
    name = "qwen2.5-vl-7b-local"

    def __init__(self, path: str = r"models/qwen2.5-vl-7b-instruct",
                 load_in_4bit: bool = False, trust_remote_code: bool = False):
        import torch
        from transformers import (AutoModelForImageTextToText,
                                  AutoProcessor)
        kw = dict(dtype=torch.bfloat16, device_map="cuda",
                  trust_remote_code=trust_remote_code)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw = dict(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
                device_map="cuda", trust_remote_code=trust_remote_code)
        if trust_remote_code:
            from transformers import AutoModelForCausalLM as _Auto
        else:
            _Auto = AutoModelForImageTextToText
        self.model = _Auto.from_pretrained(path, **kw)
        self.processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=trust_remote_code)
        self.model.eval()

    def ask(self, question: str, image: str | Path | None = None,
            max_tokens: int = 512) -> str:
        import torch
        content = []
        if image:
            content.append({"type": "image", "image": str(image)})
        content.append({"type": "text", "text": question})
        msgs = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        imgs, vids = process_vision_info(msgs)
        inputs = self.processor(text=[text], images=imgs, videos=vids,
                                padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
