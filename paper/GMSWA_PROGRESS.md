# GMSWA · 状态追踪重构进展

把 GMSWA 从"又一个高效长上下文架构"重构为 **retrieval vs state-tracking** 的机制故事。基于 Mozer et al. *The Topological Trouble With Transformers*（arXiv 2604.17121）与 Grazzi et al. 负特征值（2411.12537）。

- ✅ 两个新方法旋钮（multi-scale memory / β-range）已实现并验证
- ✅ v10ms 340M 预训练完成 + eval
- ✅ 合成状态追踪因果链闭环（parity / S_3）
- ✅ Transformer 在 S_3/latch 的对照已补齐
- ✅ v11gb scale-check 预训练 + NIAH/recall eval 完成
- ✅ submission-oriented paper draft 已生成（Markdown + AAAI LaTeX draft）
- 🟡 待做：多 seed、gate routing 可视化、自然语言化 state-tracking 任务、最终 LaTeX 编译环境

---

## 核心发现（TL;DR）

朴素 hybrid 的 mix-gate 会在硬状态追踪任务上**塌缩到 SWA 捷径**，压制 memory 分支。门偏置初始化（`mix_gate_logit_bias` 向 memory）是干净解法。最新 parity bias-only 对照显示：**课程不再是必要条件**，核心就是 gate-bias。因果链 9 个对照点，零反例。

```
问题  前馈窗外崩          (Mozer 深度限制)      ← SWA / Transformer / latch>4W
元件  β∈(0,2) 负特征值     (Grazzi)             ← memonly β1 崩, β2 满分
陷阱  朴素 hybrid 塌缩     (新发现)              ← gmswa_new on parity/S_3
诊断  强制 α=0 恢复满分                          ← force-alpha=0
解法  门偏置初始化 → 学习的门也追踪              ← gmswa_bias_only
```

---

## 一、代码改动（已合入）

| 改动 | 作用 | 默认 | 状态 |
| --- | --- | --- | --- |
| `mem_beta_range` | β∈(0,1)=sigmoid 特征值[0,1]；β∈(0,2)=2·sigmoid 特征值[-1,1]，解锁状态追踪 | 1.0 | ✅ forward/bwd/gen 验证 |
| `mem_num_scales` | 多尺度 memory（slow+fast decay，学习门组合） | 1 | ✅ DCP→HF round-trip 验证 |

> 两者与旧 checkpoint 不兼容（A_log shape `(H,)`→`(S,H)`），需 from-scratch 训练。

---

## 二、340M 预训练（检索轴，10k steps / FineWeb）

v10ms = multi-scale S=2 + β=2；v11gb = v10ms + gate-bias (`mix_gate_logit_bias=-4`)。同 harness。

| Model | NIAH@512 | @1024 | @2048 | @4096 | @8192 | recall |
| --- | --- | --- | --- | --- | --- | --- |
| v5conv (baseline) | 0.777 | 0.475 | 0.242 | 0.143 | 0.070 | 0.155 |
| **v10ms** | 0.730 | 0.489 | 0.231 | 0.156 | 0.070 | 0.160 |
| **v11gb** (gate-bias) | 0.730 | 0.424 | 0.208 | 0.101 | 0.056 | 0.134 |

**结论：**v10ms 检索轴基本打平；v11gb 训练稳定（final loss 2.5415）但 NIAH/recall 有轻微回退。paper 里应诚实表述为：gate-bias unlocks synthetic state-tracking, but introduces a retrieval/state-tracking calibration tradeoff at 340M.

---

## 三、合成状态追踪实验（state-tracking 轴）

小模型从头训（depth=6, window=64, seq_len=1024）。按 op-distance 分桶报准确率。**无泄漏设计**（content-free QUERY，状态永不进输入流）。

### Parity（1-bit 累积，干净判别器）

| variant | ≤W/2 | W/2..W | W..2W | 2W..4W | >4W |
| --- | --- | --- | --- | --- | --- |
| SWA | 1.00 | 0.77 | 0.50 | 0.50 | 0.50 |
| Transformer | 0.98 | 0.61 | 0.50 | 0.50 | 0.50 |
| memonly β=1 | 0.87 | 0.51 | 0.50 | 0.50 | 0.50 |
| **memonly β=2** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| gmswa_new (朴素 β2 hybrid) | 0.94 | 0.52 | 0.50 | 0.50 | 0.50 |
| **gmswa_new, 强制 α=0** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **gmswa_curr_bias** (课程+门偏置) | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **gmswa_bias_only** (仅门偏置，新补) | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

### S_3（6-class 群合成，更难）

| variant | ≤W/2 | W/2..W | W..2W | 2W..4W | >4W |
| --- | --- | --- | --- | --- | --- |
| SWA | 0.54 | 0.20 | 0.167 | 0.165 | 0.168 |
| Transformer (新补) | 0.474 | 0.168 | 0.164 | 0.167 | 0.167 |
| memonly β=2 | 1.00 | 1.00 | 1.00 | 0.998 | 0.972 |
| gmswa_new (朴素) | 0.50 | 0.17 | 0.16 | 0.167 | 0.167 |
| **gmswa_bias_only** (仅门偏置) | **1.00** | **1.00** | 0.995 | 0.967 | **0.887** |
| gmswa_curr_bias (课程+偏置) | 0.43 | 0.33 | 0.33 | 0.33 | 0.33 |

### Latch（携带 1-bit，易任务）

| variant | ≤W/2 | W/2..W | W..2W | 2W..4W | >4W |
| --- | --- | --- | --- | --- | --- |
| SWA | 1.0 | 1.0 | 1.0 | 0.73 | 0.49 |
| Transformer (新补) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| memonly β=2 / gmswa_new / bias_only | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**诚实细节**：门塌缩是 task-dependent——只在"必须积分全历史"的硬任务（parity/S_3）出现；latch 太易，朴素 hybrid 就能用 memory。但 latch 仍干净展示了 SWA 多层接力→深度耗尽崩溃（>4W=0.49），是 Mozer 论点的额外证据。

---

## 四、任务有效性地图

| 任务 | 有效? | 备注 |
| --- | --- | --- |
| parity | ✅ 干净断崖 | 主判别器 |
| S_3 | ✅ 更难 | 泛化；门偏置单独即够 |
| latch | ✅ 但易 | 展示 SWA 深度接力 |
| flip-flop (p=0.15) | ❌ set 太密 | 最近 set 总在窗内 |
| S_5 (120-class) | ❌ 不可学 | 此规模窗内都随机 |

---

## 五、方法论提炼

- **核心解法 = 门偏置向 memory 初始化**（`mix_gate_logit_bias` 取负）。让门默认信任 memory，不塌缩到窗内 SWA 捷径。
- 新补 parity bias-only = 1.0 全距离；S_3 bias-only 也成立。**因此核心配方简化为 gate-bias alone**。
- SWA-drop 课程是 task-dependent 的辅助，甚至可能有害（S_3 上 curr_bias 崩到 0.33）。降级为可选。
- 朴素 hybrid 失败 ≠ memory 无能：强制 α=0 / memonly β=2 都满分。

---

## 六、下一步

| 编号 | 内容 | 状态 | 目的 |
| --- | --- | --- | --- |
| N-1 | v11gb scale-check 预训练（门偏置配方，340M） | 🟡 已启动 | 确认真实 LM 下 gate-bias 不伤训练，且 NIAH/recall 不退化 |
| N-2 | Transformer 在 S_3/latch 的对照 | ✅ 已完成 | S_3 仍随机；latch 满分，补全上界 |
| N-3 | v11gb eval（NIAH/recall） | 待训练完成 | 和 v10ms/v5conv 对齐比较 |
| N-4 | 多 seed（≥3）误差棒 | 暂跳过 | reviewer 门槛 |
| N-5 | racing-thoughts 复现（Patchscopes，bank 歧义） | 待做 | 机制可解释性，额外 figure |

**v11gb 运行位置**：`GMSWA/flash-linear-attention/flame/saves/GMSWA-340M-v11gb-10k/`。配方：v10ms + `mix_gate_logit_bias=-4.0`，无 SWA-drop 课程。启动检查健康：8×H100 active，config 记录正确，step 30 loss=9.1903。

---

## 附：文件位置（node41 `/data/Minko`）

- 实验记录：`GMSWA/paper/STATE_TRACKING_EXPERIMENTS.md`
- 合成 harness：`st_gen_tasks.py` / `st_train_eval.py`
- 结果 JSON：`st_{parity_d6,parity_memonly,parity_force0,parity_curr,parity_bias_only,s3,s3_transformer,latch,latch_transformer,s5,flipflop}.json`
- v10ms checkpoint：`GMSWA/flash-linear-attention/flame/saves/GMSWA-340M-v10ms-10k/`
- v11gb running checkpoint：`GMSWA/flash-linear-attention/flame/saves/GMSWA-340M-v11gb-10k/`
