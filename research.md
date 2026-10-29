# 跨模态遗忘失败尚无因果机制解释

**结论先行：** 截至 2026-10-29，没有任何 VLM unlearning 论文用 activation patching、causal tracing、attention knockout 或逐层 logit lens，因果地解释为何"纯文本遗忘的实体事实仍可经图像+文本查询答出"。现象本身已被行为层面确证：UNLINK-VL 在 Qwen3-VL-8B 上测得 T→M 惩罚平均 21.2 分，而 M→T 仅 12.2 分，8 组方法×更新方式中全部是 M→T 更小 ([UNLINK-VL](https://arxiv.org/html/2608.03791))。最接近机制分析的是 CrossInf 的逐层 LoRA 范数与 CKA，但它只是相关性分析，且用的是安全/越狱数据 ([CrossInf](https://arxiv.org/pdf/2607.16442))。真正的机制证据分散在两类非遗忘文献里。一类是 VLM 实体召回路径：LLaVA 的图像路线跳过早层 entity-token enrichment，身份约在 L17–24 才确定 ([Cohen et al.](https://arxiv.org/html/2412.14133v3); [Venhoff et al.](https://arxiv.org/html/2512.03276v1))。另一类是 LLM 遗忘的作用位置：GA/GradDiff/NPO 主要改动末几层的 MLP coefficients 和 attention，基本不改存储 ([Hong et al.](https://arxiv.org/html/2410.06606v2))。两者合起来，原假设"文本遗忘改动了图像路线绕过的、汇合之前的 name→entity 通路"只得到部分支持。证据更支持一个精炼版本：遗忘是一个**晚层的、由输入形式决定触发的门控**（key mismatch），它只认文本路线产生的表示。这一说法可以用 LoRA 分层掩码和跨路线 patching 直接证伪。对 ARR 2027 年 1 月投稿而言，"因果机制解释"这个贡献位仍然空着，但 UNLINK-VL 与 CrossInf 已把行为现象和相关性分析占住，论文必须拿出干预性证据。

**核验说明：** 本文所有数字均来自经摘要模型转述的 WebFetch 抓取（arXiv HTML/PDF），未逐一对照原文表格。下文标 ⚠abs 的条目只核验了摘要、标题或检索摘要。投稿前须对照 PDF 复核所有引用数字。

## 十余篇跨模态遗忘论文中，无一做因果定位

下表逐篇列出是否包含机制或定位分析。"方法内定位"指定位只用于决定改哪些参数，并非用来解释遗忘为何失败。

| 论文 | 设定 | 跨模态发现 | 机制/定位分析 |
|---|---|---|---|
| UNLINK-VL 2608.03791 | 250 个真实实体，Wikidata 事实，文本/图像问题严格配对；Qwen3-VL-8B/32B | T→M 惩罚 15.9–27.4（均值 21.2），M→T 6.4–19.2（均值 12.2）；遗忘后 multi-hop 和 rewrite 准确率仍高 | **无**，只有行为准确率和案例 |
| SMU 2607.18615 | Hateful/COVID Memes；Qwen2.5-VL、InternVL3 | LM-only 文本遗忘在多模态查询上失效（InternVL NPO 39.41→73.30）；未测 M→T | **无**；⚠ 表 2 部分 GradDiff 数字与表 1 重复 |
| CrossInf 2607.16442 | 安全数据（PKU-SafeRLHF、JailbreakV）；LLaVA-1.5、InstructBLIP、IDEFICS | 文本→视觉迁移强（FigStep TS 85.6→0.9）；LLaVA 的视觉→文本有约 40 分缺口 | **有，但为相关性**：两种遗忘的逐层 LoRA 范数相关 r=0.55–0.71，L24–31 更新最大；CKA 早中层≈1、晚层下降，6 例中 5 例 CKA 越低迁移越强；无 patching 或 logit lens |
| MMUnlearner 2502.11051 | MLLMU-Bench、CLEAR；LLaVA-1.5、Qwen2-VL | 方向相反：基线主要删掉文本知识，视觉知识难删 | 方法内 Fisher 掩码，外加 top 偏移参数热图（GA/NPO 集中在中层 MLP/attn 和浅层 attn）；**未分析文本与图像显著参数的重叠** |
| MIP-Editor 2511.06793 | MLLMU-Bench、CLEAR；Qwen2.5-VL-3B、LLaVA | 联合遗忘有效（Qwen FVQA 39.2→4.8） | 方法内 neuron path（LLM FFN 与视觉 FFN）；**未分析两条路径的重叠或层分布**；⚠ 部分数字不一致 |
| CAGUL 2510.07567 | FIUBench；基座冻结，替换视觉 token | **未测**跨模态迁移 | 方法内 attention 排序，无逐层分析 |
| 鲁棒性 2605.26992 | VGGFace2，Qwen2.5-VL-3B，13 种方法 | 未直接研究跨模态；1% 重学即可让 SLUG 掉 80% 以上 | **无**；推测"hide rather than remove" |
| MLLMU-Bench 2410.22108 | 500 个虚构档案加 153 个名人；LLaVA、Idefics2 | 多模态 GA 在图像分类上更好、文本分类上更差 | **无**；作者自认"without uncovering its root cause" |
| CLEAR 2410.18057 | TOFU 作者配生成图像 | "仅文本遗忘即足够"（⚠ 表 3 支撑存疑），与 UNLINK-VL 方向相反 | **无**，只用 LoRA 范数作漂移代理 |
| VKD 2512.11325 | MLLMU-Bench、CLEAR；冻结 LLM，只改视觉端 | FVQA 45.8→29.8，FQA 基本不变 | **无自身分析**，定位论断引自他人 |
| UnLOK-VQA 2505.01456 | LLaVA-1.5，在 L7 MLP 做 rank-1 LoRA | 多模态改写攻击 45.5%，高于文本 39% 和图像 32% | **唯一在遗忘后的 MLLM 上用 logit lens 的工作**：中间层仍残留知识；编辑 LLM MLP 远优于编辑 projector（15.7% vs 57.3%）；**未比较两条路线或指出泄漏层** |
| MANU 2502.15910 | LLaVA、Idefics2 | 多模态遗忘不保证单模态删除 | 方法内神经元剪枝；"不同模态激活不同神经元"依据的是逐层保留率热图，非直接重叠分析 |
| CMJU (*Visual Intelligence* 2026) | UMU-Bench、CLEAR；LoRA 仅加在 LLM | 模态完备的监督不自动带来平衡遗忘 | 方法内 Fisher 分组（文本偏好/多模态偏好/共享）；无层或表示分析 |
| POPS 2607.06649 | 遗忘后的知识恢复攻击 | 纯文本 POPS 反而降低 6.1 分，多模态 POPS 提升 3.7 分 | **无** |
| Chakraborty et al. EMNLP'24 Findings | 安全越狱 | 文本遗忘迁移到视觉越狱（ASR 75.69→7.52） | **无**，只有"fused into language space"的概念论证 |

来源：[UNLINK-VL](https://arxiv.org/html/2608.03791)、[SMU](https://arxiv.org/pdf/2607.18615)、[CrossInf](https://arxiv.org/pdf/2607.16442)、[MMUnlearner](https://arxiv.org/pdf/2502.11051)、[MIP-Editor](https://arxiv.org/html/2511.06793v1)、[CAGUL](https://arxiv.org/html/2510.07567v1)、[Robustness](https://arxiv.org/pdf/2605.26992)、[MLLMU-Bench](https://ar5iv.labs.arxiv.org/html/2410.22108)、[CLEAR](https://arxiv.org/html/2410.18057v1)、[VKD](https://arxiv.org/html/2512.11325v2)、[UnLOK-VQA](https://www.arxiv.org/pdf/2505.01456)、[MANU](https://arxiv.org/pdf/2502.15910)、[CMJU](https://link.springer.com/article/10.1007/s44267-026-00130-4)、[POPS](https://arxiv.org/pdf/2607.06649)、[Chakraborty](https://aclanthology.org/2024.findings-emnlp.574.pdf)。

各作者给出的解释可归为五类，没有一类经过干预检验：LLM 是瓶颈、一切汇入语言空间；跨模态纠缠或走不同通路；身份存储在视觉编码器或 projector 中；架构耦合程度不同（MLP projector 与 Q-Former 和 cross-attention 的差别）；遗忘只是抑制输出 token 或拒答。CrossInf 对 LLaVA 的判断"transfer is a boundary effect, not a representational one"，以及鲁棒性论文所说的"decoder-focused methods mainly suppress token probabilities"，最接近下文的门控假设 ([CrossInf](https://arxiv.org/pdf/2607.16442); [Robustness](https://arxiv.org/pdf/2605.26992))。

文献状态需注意几点。⚠abs：VGID 2606.00105、Knowledge Holes 2608.01849、ICCV 2025 TMKE（文本编辑向多模态的传递性，用 Knowledge Neurons 分析）和 ICML 2026 UMM 跨模态编辑论文，都只读了摘要或标题，不能排除其中含机制内容；Knowledge Vector Weakening 2601.21794 因限流未读。UNLINK-VL 的作者在两份笔记中不一致，一份记为 Chunlin Liu…Yuntao Du，另一份的摘要转述页记为 "Chen et al."，需核对。MLLMU-Bench 的 NAACL 2025 与 CAGUL 的 NeurIPS 2025 track 均未核实。OpenReview 上 ICLR 2027 的在审稿件未检索。

## 图像路线在 LLaVA 中约 L20 才完成实体识别

VLM 召回机制的证据相当一致。Venhoff et al. 用 attribution patching 找到两个召回站点：早层 entity-token 处的 MLP+attn，以及中晚层 last-token 处的 MLP+attn。在 LLaVA-1.5-7B/13B 和 LLaVA-MORE 上，图像实体**只激活第二个站点**；Gemma-3-12B 与 Qwen2.5-VL-7B 两个站点都用。把 LLM 骨干的早层 entity-token MLP 输出注入 VLM，可恢复约 35% 的差距；"先说出实体名"的 CoT 对 Pixtral 能完全弥合差距，对 LLaVA-1.5-7B 无效 ([Venhoff et al.](https://arxiv.org/html/2512.03276v1)；venue 可能是 workshop，未核实)。Cohen et al. 的 cross-patching 显示，LLaVA-1.5-7B 的图像身份在 **L20** 才确定（SigLIP 为 L17，CLIP 为 L24），图像 token 的 attention 在 L15–20 最关键。文本与图像的准确率差距在 LLaVA-1.5-7B 上为 0.177，在 **Qwen2-VL 上仅 0.043**。早识别实体（<L5）的差距依然存在，说明瓶颈在于身份是否以"名字兼容"的格式出现在 subject 位置，而不只在于何时出现 ([Cohen et al.](https://arxiv.org/html/2412.14133v3))。Nikankin et al. 在 Qwen2-VL、Pixtral 和 Gemma-3 上发现，视觉与文本电路只共享约 18% 的组件，data position 的重叠接近 0，而 query 和 generation 子电路可以跨模态互换且保持 faithfulness ([Nikankin et al.](https://arxiv.org/html/2506.09047v3))。Neo et al. 的 logit lens 显示，LLaVA-1.5 的物体信息在约 L25.7/33 才可解码，last token 主要在 L15–24 从物体 patch 读取信息 ([Neo et al.](https://arxiv.org/html/2410.07149v2))。

综合起来：**下游的 query/answer 提取电路是共享的，分歧在上游的 data-token 处理**，汇合层随模型而变。Basu et al. 报告 LLaVA 早层 MLP（约 L1–4）有因果作用，并且用图像输入在 L2 做的编辑可以泛化，但其中被破坏的"视觉约束"多为文本短语，与 Venhoff 的设定不可直接比较 ([Basu et al.](https://arxiv.org/html/2406.04236v1))。⚠ LatentLens 声称 logit lens 严重低估了视觉 token 的可解释性，这一点只读了摘要 ([LatentLens](https://arxiv.org/abs/2602.00462v1))。另一缺口是：Venhoff 关于"早层复用"的证据针对 Qwen2.5-VL，**而不是用户计划使用的 Qwen2-VL**；Qwen2-VL 只有 Cohen 的 0.043 小差距和 Nikankin 的电路证据作间接支持。

## LLM 遗忘改的是晚层提取，不是存储

文本 LLM 方面的证据也很一致：GA、GradDiff、NPO 和 DPO 都在压制知识，而非删除。Hong et al. 显示 MLP value vectors 几乎不变，只恢复它们时 KRS<0.001；**只恢复最后 2–3 层的 MLP coefficients，KRS≥0.8** ([Hong et al. 2024](https://arxiv.org/html/2410.06606v2))。ConceptVectors 中各方法把 target QA 压到 0.25 以下，但概念向量的 Jaccard 仍>0.96 ([Hong et al. EMNLP'25](https://arxiv.org/html/2406.11614v2))。Guo et al. 指出非定位微调"mainly change the extraction mechanisms"；早层 fact-lookup 定位（Gemma/Llama 的 L2–8）必须人为施加，换来的 MCQ 召回是 0.11，而 causal tracing 定位为 0.85 ([Guo et al.](https://arxiv.org/html/2410.12949v2))。可恢复性证据同样充分：10 条无关样本即可让 NPO 恢复到 52.7%，RMU 在消融拒答方向后恢复到 64.7% ([Łucki et al.](https://arxiv.org/pdf/2409.18025))；4-bit 量化让遗忘知识的留存率从 21% 升到 83% ([Zhang et al.](https://arxiv.org/html/2410.16454v3))；RTT 下预训练知识恢复>88%，但**微调注入的知识恢复<35%** ([Deeb & Roger](https://arxiv.org/html/2410.08827v3))。

与跨模态最接近的类比是多语言遗忘：遗忘后，跨语言问题表示在早中层仍高度相似，只在晚层解码处分叉；要求模型用微调语言作答，召回最多恢复 61 分 ([Xiang et al.](https://arxiv.org/pdf/2606.03291))。在 RWKU 上，multi-hop 查询残留更多（1/2/3 跳分别约 20%/27%/34%），alias 只多恢复 2.4 分，说明输入一旦映射到实体表示，名字形式就不再重要 ([Shah et al.](https://arxiv.org/html/2603.11266v1))。反方证据是：随机选 10% 的 value vectors 做遗忘，效果不差于精心定位 ([Lee et al.](https://arxiv.org/html/2505.16252v1))。⚠abs：Ssiuu 的"spurious unlearning neurons"只读了摘要 ([Yang et al.](https://arxiv.org/abs/2509.22263v1))。注意以上全部是纯文本 LLM 上的结论，没有在 VLM 上复现过。

## 原假设需改为晚层的、按输入形式触发的门控

**原假设 H0**（文本遗忘改动汇合前的 name→entity 通路，图像路线绕过它）有两条支持：LLaVA 图像路线确实绕过早层 entity-token 站点 (Venhoff)，且 data-position 电路几乎不重叠 (Nikankin)。它与两条证据冲突。第一，默认的 GA/NPO **并不主要修改早层查找**，改的是晚层提取 (Hong; Guo)。第二，UNLINK-VL 在 **Qwen3-VL** 上同样测到 21 分的 T→M 惩罚，而 Venhoff 认为 Qwen 系列会复用早层站点；按 H0，此时应当迁移良好（推断：Qwen3-VL 本身未被机制研究过）。这就出现一个悖论：如果遗忘作用在晚层，而晚层提取电路跨模态共享，遗忘本应能够迁移。

解释这个悖论的最佳候选是 **H1，按输入形式触发的晚层门控（key mismatch）**。遗忘改变的是晚层 MLP 的 *coefficients*，即激活，而激活取决于输入的 key。GA 学到的抑制方向由文本路线在晚层 last-token 处特有的特征触发，例如经 attention 从名字 token 搬运来的、带有名字形式的成分。图像路线到达同一个共享提取电路时携带的是不同的 key，所以门控不触发，存储完好的事实照常被读出。支持 H1 的证据有：Basu 用*图像输入*计算的编辑能够泛化，说明问题出在 key 而非存储位置；CrossInf 观察到 L24–31 更新最大、CKA 晚层下降；以及多语言遗忘中晚层分叉的类比。M→T 惩罚较小也能自然解释：多模态遗忘数据里含文本问句，其门控 key 天然覆盖更多形式（推断）。**H2，架构调制的汇合深度**：LLaVA 在约 L20 才解析图像实体，Qwen2.5-VL 和 Gemma-3 则早期复用，因此不对称性应随"汇合层减去遗忘作用层"的差值增大。H2 与 H1 相容，可作为 H1 的调节变量，但单独无法解释 Qwen3-VL 的结果。**H3，视觉端存储**（SMU、VKD 的说法）：身份信息存放在视觉编码器或 projector，LLM-only 遗忘碰不到。H3 与 UnLOK-VQA 的结果相矛盾：编辑 LLM MLP 远比编辑 projector 有效。

| 假设 | 可证伪预测 | 判别实验 |
|---|---|---|
| H0 汇合前 bypass | 文本遗忘的 Δ 集中在早层 subject token；在 L<L_conv 处，两条路线的实体表示相似度低 | **LoRA 分层掩码**：只在 [0, L_conv) 做文本遗忘，图像路线应完全不受影响；只在 [L_conv, N) 做，应能迁移 |
| H1 晚层输入形式门控 | Hong 式恢复定位到晚层 coefficients；在该层，两条路线的实体内容可线性互译（probe 跨路线迁移），但 ΔMLP 输出在图像查询上远小于文本查询；**把文本 prompt 中名字路线的晚层 residual 补丁进图像查询会重新触发遗忘** | 逐层计算 cos(Δh_text, Δh_image) 与 ‖Δh‖ 之比；跨路线 activation patching；"先说名字"CoT 应恢复遗忘效果 |
| H2 架构调制 | 不对称性 LLaVA > Qwen2-VL；对每个实体，Cohen 式切换层越晚，迁移越差 | 两个模型对比，并按实体的切换层分层回归 |
| H3 视觉端存储 | 冻结 LLM、只改视觉端即可让图像路线遗忘，且不影响文本路线；把原始模型的视觉端输出补丁进去不会恢复图像召回 | 把原始模型的 projector 输出 patch 进遗忘模型；对比 VKD 式的视觉端遗忘 |

几个 logit lens 层面的辅助预测：在 H1 下，文本遗忘模型处理图像查询时，答案在晚层的排名应与原始模型相同，而处理文本查询时只在最后几层被压低。在 H0 下，文本查询的答案应从早中层起就无法解码。用 4-bit 量化或少量重学作为压力测试可以区分"门控"与"删除"：两条路线的恢复率应当相同，只有门控失效的那条路线在遗忘前就已"泄漏"。

## 不对称方向随设定翻转，且 MLLMU-Bench 存在未被提及的混杂

在真实预训练实体上（UNLINK-VL、SMU），T→M 失败比 M→T 严重。在微调注入的虚构档案上（MMUnlearner、MANU、MLLMU-Bench、CLEAR），结果相反或混杂：MMUnlearner 的基线先删掉的是文本知识 ([MMUnlearner](https://arxiv.org/pdf/2502.11051))，CLEAR 声称纯文本遗忘就够 ([CLEAR](https://arxiv.org/html/2410.18057v1))。在安全数据上，文本→视觉迁移近乎完全 ([CrossInf](https://arxiv.org/pdf/2607.16442); [Chakraborty](https://aclanthology.org/2024.findings-emnlp.574.pdf))。一个统一的解释是：拒答类的安全目标对应粗粒度、跨模态共享的"有害性"特征，门控 key 天然不区分模态；实体事实则需要先解析身份，key 带有路线特异性。对虚构档案，Deeb & Roger 显示微调注入的知识本身更易被删 ([Deeb & Roger](https://arxiv.org/html/2410.08827v3))，所以泛化出来的路线最先垮掉（以上均为推断）。

**MLLMU-Bench 混杂没有任何论文明确指出。** 它的 vanilla 微调数据只描述为 ⟨image, question, answer⟩ 三元组，没有提到纯文本训练；每个档案有 7 道图像题和 7 道文本题，但没有说明两组题是否针对同一属性；正文写 3 个 epoch，表 3 写 4 个 ([MLLMU-Bench PDF v2](https://arxiv.org/pdf/2410.22108v2))。如果属实，文本路线召回只是对照片绑定学习的泛化，两组题测的也不是同一批事实，任何"跨模态不对称"都会混入"知识强度不同"和"题目不同"两个因素。CLEAR 也只是部分配对（3,770 道图像题对 4,000 道文本题）。UNLINK-VL 批评了合成档案，并用 "differ only in how the subject entity is accessed" 的配对设计隐式规避了这个问题，但没有点名该混杂 ([UNLINK-VL](https://arxiv.org/html/2608.03791))；CrossInf 只说各基准"not contradictory" ([CrossInf](https://arxiv.org/pdf/2607.16442))。这一推断尚未通过检查发布的训练 JSON 来证实。

## 对计划实验的设计含义

**基准层面**，先下载 MLLMU-Bench 的训练 JSON，确认只有照片形式，并统计 7 道图像题与 7 道文本题的属性重叠。然后自建配对题，让同一 (E, r, o) 各有文本和图像两种问法，只保留 vanilla 模型在两种形式下都答对的事实（沿用 UNLINK-VL 的联合可识别过滤）。另外加一个预训练知识条件，可以用 MLLMU-Bench 的 153 位名人子集或 UNLINK-VL 的数据。否则审稿人会把不对称归因于混杂，而且"方向随设定翻转"本身就可以写成一个发现。**模型层面**，LLaVA-1.5-7B 的身份切换层已知（约 L20），适合检验 H0/H2。Qwen2-VL 的早层复用尚无直接证据，需要先自己做 Cohen 式 cross-patching 和逐层 probe 测出 L_conv，才能确定它是不是 H2 的对照组；建议加入 Qwen2.5-VL-7B，以便与 Venhoff、CAGUL 和 MIP-Editor 直接对齐。**方法层面**，GA、GA_Diff 和 NPO 用 LoRA 时，LoRA 的放置位置本身就是自变量。逐层记录 LoRA 范数（以 CrossInf 为基线），同时跑 T→M 和 M→T 两个方向，并加入 4-bit 量化或少量重学的压力测试。核心因果实验按成本从低到高排列："先说名字"CoT 诊断、Hong 式组件恢复、跨路线 activation patching、LoRA 分层掩码。**定位层面**，UNLINK-VL 占据了行为现象，CrossInf 占据了相关性 CKA，Venhoff 和 Cohen 占据了无遗忘情形下的召回机制。本文的差异点必须是"在遗忘后的 VLM 上做干预，说明哪条路线、哪一层的哪个 key 触发了门控"。投稿前需复查 ICLR 2027 在审稿件，排查并发工作。

## Conclusion

现有文献把问题推进到一个具体的矛盾：遗忘作用在晚层，晚层提取电路跨模态共享，遗忘却不能迁移。H0 的"早层 bypass"图景不足以解释这个矛盾，Qwen3-VL 上的失败尤其如此。"门控的触发条件是路线特异的 key，而非存储位置"这一表述既能解释 T→M 与 M→T 的不对称，也能解释安全数据上的迁移成功，并且可以用 LoRA 分层掩码在几天的算力内证伪，适合作为论文主线。

实践上，最大的风险不在机制，而在评测本身。如果不先排除 MLLMU-Bench 的照片单一训练形式和题目不配对这两个混杂，任何机制故事都可能只是在解释数据伪影。把"方向随知识来源翻转"做成受控变量，同时可以回应审稿人对 benchmark 选择的质疑。