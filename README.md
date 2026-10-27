# CDW-VQA

Benchmark construction, evaluation, and adaptation code for
*When Objects Overshadow Materials: Benchmarking and Adapting Multimodal Large Language Models for Construction and Demolition Waste Auditing*.

Jin Xin Cao (Inner Mongolia Academy of Science and Technology; Inner Mongolia University) and Shi Yang Li (Inner Mongolia University). Manuscript under review, 2026. This release accompanies the submission and is intended to make every reported number reproducible; the benchmark construction, human-reference protocol and scoring code are released together so the results can be checked independently.

## What is here
- `src/cdw_vqa/` — dataset parsers (CODD XML polygons; CDW-Seg COCO), four-arm image synthesis, unified clients for local HF models and API models.
- `scripts/` — the full pipeline, numbered in execution order:
  1. `freeze_instances.py`, `build_diagnostic_v2.py` — instance manifest and the four-arm suite (485 instances x 4 arms, paired object/material questions)
  2. `build_benchmark_v1.py` — scene suite (4,072 QA over 930 images)
  3. three-judge labelling: `prefill_pipes.py`, `prefill_pipes_glm.py`, `prefill_pipes_kimi.py`, `merge_judges.py`, `finalize_labels.py`, `round2_setup.py`, `finalize_pipes_labels.py`
  4. human labelling tools: `label_server.py`, `build_judge_views.py`, `make_judge_sheets.py`, `baseline_server.py`, `build_human_baseline.py`
  5. evaluation: `run_w6_gate.py`, `w6_stats.py`, `w6_typology.py`, `run_scene_eval.py`, `scene_stats.py`, `run_api_eval.py`, `api_policy.py` (unified delivery policy: success / end-to-end / conditional + delivery mask)
  6. adaptation: `build_sft_data.py`, `train_qlora.py`, `train_qlora_glm.py`, `eval_adapted.py`, `adapted_compare.py`; ablation `build_sft_data_l0.py`, `eval_l0_ablation.py`; oracle control `run_oracle.py`; supervised baseline `train_seg_baseline.py`, `eval_seg_baseline.py`
  7. two-level scoring and corrected gate: `fine_metrics.py` (coarse vs fine material, symmetric object aliases, corrected interaction with cluster bootstrap, McNemar)
  8. human baseline v2 (balanced, fine-grained, blind): `build_human_baseline_v2.py`, `baseline_server_v2.py`, `human_baseline_v2_score.py`
  9. scene human reference (ten annotators, 120 stratified items): `build_scene_survey.py` builds the item set; `scene_human_reference.py` reproduces the per-cell table from `data/annotators_scene/`
 10. unified scene scoring: `rescore_scene.py` (boxed answer if present, else first token; writes `results/scene_eval/scene_scores_unified.json`)
- `data/` — benchmark manifests and QA files (question lists, manifests, labelling CSVs).
- `results/` — raw model responses and aggregated statistics for every number reported in the paper; the analysis scripts read only these files, so all reported statistics can be regenerated without a GPU.

Note on file versions: files ending in `_v1biased` (and the unsuffixed `sft_train.jsonl` counterpart `sft_train_v1biased.jsonl`) are snapshots from before the answer-key rebuild; they are kept for provenance only. Every number in the paper is computed from the unsuffixed and `_v2` files.

Note on the L1 arms: the released images are authoritative. For diagnostic ids 000-071 the L1 image comes from a first-generation full-image inpainting renderer; `src/cdw_vqa/arms.py:arm_l1_full_inpaint` is a re-implementation kept for inspection and it matches the released background to roughly 9/255 on id 000 rather than exactly. The sensitivity split reported with the L1 results is computed from the released images.

Note on GLM-4.1V scene answers: `results/scene_eval/scene_glm-4.1v-9b.jsonl` was collected before the answer-key rebuild, and its option sets differ from the released corpus. Both task families have absolute answers (a count, a material name), so `scripts/rescore_scene.py` re-scores those responses against the released truth by mapping the model's boxed choice to its content; `results/scene_eval/scene_scores_unified.json` holds the resulting cells for all five systems. All other systems answer in their first token and are unaffected by the rule.

## Human reference

Two independent cohorts, both collected blind from single views with a forced choice and per-annotator randomised order:

- **Scene suite** — ten annotators (one co-author, nine graduate students) answered 120 stratified items (F1--F4 x CODD/CDW-Seg x 15). Per-annotator accuracy 0.49--0.76, Fleiss' kappa 0.29--0.56 by family. Responses are in `data/annotators_scene/`; `python scripts/scene_human_reference.py` reproduces the per-cell means, 95% CIs, majority vote and kappa.
- **Instance suite** — fifteen annotators (two authors, thirteen graduate students) answered 200 balanced PVC/metal items; responses are in `data/annotators/`, and `results/annotators/human_multi_annotator_stats.txt` holds the aggregate statistics.

## Reproducing the reported numbers

| Quantity | Script | Input |
|---|---|---|
| Scene cells, all five systems | `scripts/rescore_scene.py` | `results/scene_eval/*.jsonl`, `results/api_eval/api_*_v2.jsonl` |
| Two-level material scoring, corrected gate | `scripts/fine_metrics.py` | `results/w6/`, `results/adapted/*_v2.jsonl`, `results/oracle/*_v2.jsonl` |
| Balanced accuracy and per-class recall | `scripts/balanced_metrics.py` | same |
| API delivery policy and mask | `scripts/api_policy.py` | `results/api_eval/api_*_v2.jsonl` |
| Scene human reference | `scripts/scene_human_reference.py` | `data/benchmark_v1_scene_qa.jsonl`, `data/annotators_scene/` |

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
