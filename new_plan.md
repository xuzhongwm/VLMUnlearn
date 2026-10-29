---
title: "vlm-cross-modal-gate"
type: plan
created: 2026-10-29
updated: 2026-10-29
status: draft   # 用户确认 Hypothesis + Falsification Criteria 后改为 pre-registered
author: mixed
design: "[[vlm-cross-modal-gate-design]]"
methodology: quantitative
---

# VLM 跨模态遗忘泄漏的机制解释 — Research Plan

> 执行方：Claude Code（写代码）+ Google Colab（跑实验）。每个 Task 的产物路径都相对于实验 repo 根目录。

**Research Question:** 文本 unlearning 为什么不能迁移到图像查询？泄漏发生在模型内部的哪个位置、哪个机制？

**Hypothesis (pre-registered, immutable):**
H1：在两条路线都学过的 VLM 中，文本 unlearning 不删除事实，而是在晚层（≥ L_conv）建立一个由输入形式触发的门控；图像查询携带不同的 key，不触发门控，因此泄漏。

**竞争假设：** H0（遗忘改的是汇合前通路，图像绕过）。H2（不对称随汇合深度变化）作为探索性分析，不预注册。

**Pre-registered predictions:**
- **P1 存在性（go/no-go）：**在配对题 + 双形式 vanilla 上，A = ND(T→T) − ND(T→I) > 0，95% bootstrap CI（按人物重采样）不含 0，在 ≥2/3 种方法、两个 seed 上成立。ND = (vanilla − unlearned)/(vanilla − 0.25)，在匹配遗忘强度（in-modality ND = 0.8±0.1，retain 下降 ≤10 分）的 checkpoint 上计算。
- **P2 晚层掩码（区分 H1/H0 的核心实验）：**只在 [L_conv, N) 层放 LoRA 做文本 unlearning，不对称仍然存在（A > 0，CI 不含 0）。
- **P3 跨路线 patching：**在 ≥ L_conv 的某一层，把 unlearned 模型文本查询的 last-token residual patch 进图像查询，图像题准确率下降 ≥15 分；反向 patch（图像 → 文本）使文本题准确率恢复 ≥15 分。
- **P4 存储完好：**在 unlearned 模型上，图像查询的逐层 logit lens 正确答案排名与 vanilla 无显著差异（直到最后 3 层）。

**Falsification Criteria:**
- P1 失败 → H1 不可检验；停止机制实验，论文改写为"公平设置下不对称消失 / 方向翻转"。
- P2 失败（晚层掩码下遗忘能完全迁移，A 的 CI 包含 0 且点估计 < 0.05）→ 支持 H0，拒绝 H1。
- P3 在所有 ≥ L_conv 的层上两个方向的效应都 < 5 分 → 拒绝"key 不匹配"机制。
- 只有 P2 和 P3 同时成立才算支持 H1；只成立一个记为 "partial / inconclusive"。

**Stop Criterion:** 2 个模型（LLaVA-1.5-7B；Qwen2.5-VL-7B）× 3 种方法（GA、GA_Diff、NPO）× 2 个 seed，forget_10。2026-12-20 之后不再新增实验。

**Method:** LoRA unlearning；Cohen 式 cross-patching 测 L_conv；activation patching；逐层 logit lens；LoRA 分层掩码；4-bit 量化 / 少量重学压力测试。

**Output target:** article (ARR 2027-01-04 → ACL 2027 main，保底 Findings)
**Language:** en

---

## Data Sources

- MLLMU-Bench（HF `MLLMMU/MLLMU-Bench`，github.com/franciscoliu/MLLMU-Bench）— 未下载
- 次要：MLLMU-Bench 中 153 位真实名人（预训练知识条件）
- 参照：UNLINK-VL（arXiv 2608.03791）的配对设计；Cohen et al.（2412.14133）的 cross-patching 协议；Hong et al.（2410.06606）的组件恢复协议

## Tasks

### Task 0: 混杂检查（11/1 前）
**Files:** `analysis/confound_check.md`
- [ ] 下载 MLLMU-Bench 训练 JSON，确认 vanilla 训练只有照片形式
- [ ] 统计每人 7 道图片题和 7 道文字题的属性重叠
- [ ] 结论写入 `analysis/confound_check.md`

### Task 1: 配对题与双形式 vanilla（11/1–11/8）
**Files:** `data/build_pairs.py`, `data/pairs_forget10.jsonl`, `train/finetune_vanilla.py`, `results/vanilla_acc.csv`
- [ ] 从 9 个传记字段生成配对题，评测措辞与训练措辞不同
- [ ] 用照片形式 + 名字替换的文本形式同时微调 vanilla（先 Qwen2-VL-2B 跑通，再 LLaVA-1.5-7B）
- [ ] 两种模态的 forget 准确率都 ≥80%，否则调整训练再继续
- [ ] 只保留两种形式都答对的事实

### Task 2: 存在性检验 P1（11/8–11/15，go/no-go）
**Files:** `train/unlearn.py`, `eval/mc_logit.py`, `results/transfer_table.csv`, `results/p1_verdict.md`
- [ ] 最小验证：GA_Diff、一个 lr、只做文本遗忘，报告 4 个数
- [ ] 完整网格：3 种方法 × 2 种遗忘模态 × lr {2e-5, 5e-5, 1e-4} × 2 seeds
- [ ] 按匹配遗忘强度规则选 checkpoint，计算 A 和 bootstrap CI
- [ ] 写 GO / NO-GO / UNCLEAR 判定

### Task 3: 测汇合层 L_conv（11/15–11/22）
**Files:** `mech/cross_patch.py`, `results/lconv.json`
- [ ] 在 vanilla 上做 Cohen 式实体 cross-patching，逐层找身份确定层
- [ ] 两个模型各报告 L_conv

### Task 4: 晚层掩码 P2（11/22–11/29）
**Files:** `train/unlearn.py --layers`, `results/p2_mask.csv`
- [ ] 文本遗忘只用 [L_conv, N) 层 LoRA；对照只用 [0, L_conv) 层
- [ ] 两个模型 × 3 种方法 × 2 seeds，计算 A 和 CI

### Task 5: 跨路线 patching P3 与 logit lens P4（11/29–12/13）
**Files:** `mech/route_patch.py`, `mech/logit_lens.py`, `results/p3_patch.csv`, `results/p4_lens.csv`
- [ ] 逐层（≥ L_conv）做文本 → 图像、图像 → 文本两个方向的 last-token residual patching
- [ ] 逐层 logit lens：比较 vanilla 和 unlearned 模型在两种查询上的正确答案排名

### Task 6: 压力测试与探索性分析（12/13–12/20）
**Files:** `results/stress.csv`, `results/exploratory.md`
- [ ] 4-bit 量化、少量重学，比较两条路线的恢复率
- [ ] "先说名字" CoT 诊断
- [ ] H2：按每个实体的身份确定层做分层回归（探索性，不预注册）
- [ ] 真实名人条件（探索性）

### Task 7: 写作（12/20–1/3）
**Files:** `paper/main.tex`
- [ ] 8 页 long paper，ARR 模板
- [ ] 用 requesting-peer-review 做两轮模拟审稿
- [ ] 复查 ICLR 2027 在审稿件中有无并发工作
- [ ] 所有引用数字对照原论文 PDF 复核

## Verification

完成条件：
- [ ] `results/p1_verdict.md` 给出明确判定
- [ ] P2、P3、P4 各自明确标注 supported / refuted / inconclusive
- [ ] 所有 run 的 config 和日志都在 Drive 上，可复现
- [ ] 论文在 2027-01-04 AoE 前提交到 ARR

## Deviations Log

在预注册之后，范围、方法或数据的任何改动都记录在 `DEVIATIONS.md`，注明日期和原因；受影响的结果标为 exploratory。