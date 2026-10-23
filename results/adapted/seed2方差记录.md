

## seed2 方差实验记录（2026-10-23）
- 第二训练种子（SEED=20261023，同配方同数据 4,277），adapter qwen3vl-cdw-lora-r16-seed2，评测 results/adapted/adapted_qwen3vl_seed2.jsonl（2,407 条）。
- 关键对照（seed1 vs seed2）：fine 材质 L1 .766/.761、L0 .378/.372（±.005）；物体 L0 瓶颈 .487/.487（完全一致）；L1 物体 .862/.910；F3 cdw_seg .897/.966（n=29 小样本波动）；coarse L0 .729/.628（粗粒度受基率敏感，佐证以 fine 为准）。
- 结论：审计级数字 run-to-run 稳定（≤±.005），单 seed 质疑排除 → §5.2 已加一句。
