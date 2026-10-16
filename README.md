# CDW-VQA

Benchmark construction, evaluation, and adaptation code for
*When Objects Overshadow Materials: Benchmarking and Adapting Multimodal Large Language Models for Construction and Demolition Waste Auditing*.

## What is here
- `src/cdw_vqa/` — dataset parsers (CODD XML polygons; CDW-Seg COCO), four-arm image synthesis, unified clients for local HF models and API models.
- `scripts/` — the full pipeline, numbered in execution order:
  1. `freeze_instances.py`, `build_diagnostic_v2.py` — instance manifest and the four-arm suite (485 instances x 4 arms, paired object/material questions)
  2. `build_benchmark_v1.py` — scene suite (4,072 QA over 930 images)
  3. three-judge labelling: `prefill_pipes.py`, `prefill_pipes_glm.py`, `prefill_pipes_kimi.py`, `merge_judges.py`, `finalize_labels.py`, `round2_setup.py`, `finalize_pipes_labels.py`
  4. human labelling tools: `label_server.py`, `build_judge_views.py`, `make_judge_sheets.py`, `baseline_server.py`, `build_human_baseline.py`
  5. evaluation: `run_w6_gate.py`, `w6_stats.py`, `w6_typology.py`, `run_scene_eval.py`, `scene_stats.py`, `run_api_eval.py`
  6. adaptation: `build_sft_data.py`, `train_qlora.py`, `train_qlora_glm.py`, `eval_adapted.py`, `adapted_compare.py`
- `data/` — benchmark manifests and QA files (question lists, manifests, labelling CSVs).
- `results/` — raw model responses and aggregated statistics for every number reported in the paper; the analysis scripts read only these files, so all reported statistics can be regenerated without a GPU.

## Data sources (public)
- CODD: Demetriou et al., Waste Management 178 (2024), Mendeley Data (CC BY)
- CDW-Seg: Sirimewan et al., Scientific Data (2025), figshare (CC0)

Expected local layout (configure paths at the top of each script or via the defaults):
```
datasets/codd/...        # CODD images + XML
datasets/cdw_seg/...     # CDW-Seg COCO format
datasets/plan3_arms/     # generated four-arm images
models/                  # local HF checkpoints
```

## API keys
Set as environment variables when running `run_api_eval.py` / `prefill_pipes_kimi.py`:
`KIMI_API_KEY`, `KIMI_BASE_URL`, `GLM_API_KEY`, `GLM_BASE_URL`, `GLM_MODEL`,
`DEEPSEEK_API_KEY`, `GPT_API_KEY`, `GPT_BASE_URL`, `GPT_MODEL`.
Never commit keys.

## Environment
Python 3.10; see `requirements.txt` (PyTorch 2.x + CUDA, transformers 5.x, bitsandbytes, peft, accelerate, trl, opencv, scipy, matplotlib, openai, anthropic, qwen-vl-utils, python-docx). All local training and inference ran on a single 32 GB GPU.

## Reproducing
Run scripts in numbered order. Every evaluation/training script is resume-safe (skips already-answered items). Statistical analyses (`w6_stats.py`, `scene_stats.py`, `adapted_compare.py`) depend only on `results/`.
