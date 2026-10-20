# PRE_REGISTRATION — W6 evidence gate (analysis plan)

Verbatim extract from the project analysis plan (`00_选题决策报告.md` §9.3, "W6 证据闸门 v2"),
authored 2026-10-09. Instance-suite data was collected 2026-10-10/11 and the gate was
evaluated 2026-10-11 (`results/w6/w6_gate_stats.txt`). The original scorer (OBJ_ALIAS /
MAT_ALIAS tables, exact matching rule) is `scripts/run_w6_gate.py`, unchanged since the
gate run; the corrected post-hoc scorer is `scripts/fine_metrics.py` (symmetric object
aliases, two-level material scoring).

---

## W6 证据闸门 v2（红队三致命攻击修订后）

- **统计功效**（治 A4）：TOST 等效检验 ±5pp 需 343–514 对/臂 → 闸门扩至 **500 实例×4 臂**
  （power 0.89–0.97）；**交互项升主判据**（≥10pp，power≈0.96）；正交档 ±8pp 形式 TOST；
  10–15pp 灰区预承诺扩样重测；
- **词汇蕴含分层**（治 A5）：按 H(material|object) 条件熵三档分层，**headline 数字只许
  正交档（配额≥45%）**；改名对照（pseudoword 双格臂+蕴含中和选项臂各 150 题）作诊断
  （非闸门判据）；人类基线 200 题加"答题依据"标签防反噬。

（02_方案.md §W6 与本节共同构成预注册文本；四臂定义 L0 原位 / L1 同画布背景抹除
（保目标像素/尺度/位置）/ L2 背景模糊 / L3 分辨率匹配裁剪。）

---

## Outcome (recorded 2026-10-11, before any scorer revision)

- Interaction (Δ_L0 − Δ_L1) pooled = **−0.162 ± 0.050** → pre-registered criterion (≥ +0.10) **FAILED**.
- Per-model interactions: −0.138 / −0.089 / −0.259. TOST equivalence on L1 holds only for Qwen3.
- Verdict recorded verbatim in `results/w6/w6_gate_stats.txt` ("W6 闸门判决: FAIL").

## Post-hoc correction (2026-10-19, disclosed in the paper §4.6)

Two scorer artefacts documented: (i) material alias supersets of object aliases for
entangled classes; (ii) coarse-level material scoring under an 88% PVC base rate.
Corrected, symmetric, audit-grade scorer (`fine_metrics.py`) on the same responses:
interaction +0.31 / +0.24 / +0.14 per model, pooled **+0.23**, cluster-bootstrap 95% CI
[0.19, 0.27]. Labeled post hoc in the paper; recommended for confirmation on
independently collected data.
