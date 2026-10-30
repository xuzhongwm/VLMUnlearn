# 跨模态遗忘失败尚无因果机制解释

**结论先行：** VLM 里只用一种模态做遗忘后，换另一种模态往往仍能问出被遗忘的知识，而且文本遗忘向图像查询的泄漏（T→M）大于反方向（M→T）。这一现象已在行为层面被报告：UNLINK-VL 在真实知识上测到 T→M 的迁移差于 M→T ([UNLINK-VL](https://arxiv.org/abs/2608.03791))，SMU 与 IDUnlearn-Bench 报告了恢复与旁路访问 ([SMU](https://arxiv.org/abs/2607.18615); [IDUnlearn-Bench](https://arxiv.org/abs/2609.33481))。已有的内部分析要么只是相关性（CrossInf 的逐层 LoRA 幅度与 CKA），要么做在**遗忘前**的模型上（PAVA 的因果追踪与权重移植）([CrossInf](https://arxiv.org/abs/2607.16442); [PAVA](https://arxiv.org/abs/2608.30649))。**没有工作在遗忘后的模型上，对另一模态的查询做逐层因果干预。** 本项目做这件事：用 logit lens、LoRA 分层掩码、‖ΔW·x‖ 分解和跨路线 patching，在三个竞争机制（H1 晚层输入形式门控、H1′ 早层路线分离、H2 抑制—补偿）之间做判别，并据此解释不对称。目标：ARR 2027 年 1 月投稿。

**核验说明：** 本文数字大多来自经摘要模型转述的 WebFetch 抓取（arXiv HTML/PDF），未逐一对照原文表格。下文标 ⚠abs 的条目只核验了摘要、标题或检索摘要。主表中的 arXiv 编号由 PI 于 2026-10-29 核对；其他引用投稿前须对照 PDF 复核。`TODO(confirm)` 标出必须先核实才能保留的数字。

## 相关工作：关键区分点是"遗忘后、跨模态、逐层干预"

下表只按一个问题区分：**是否在遗忘后的模型上，对另一模态的查询做了逐层因果干预。** 表中每一行的答案都是否。

| 工作 | 做了什么 | 没做什么 |
|---|---|---|
| CrossInf ([2607.16442](https://arxiv.org/abs/2607.16442)) | 三种架构上双向测迁移，发现不对称且依赖融合结构；排版攻击可恢复；用 block 级影响函数选更新位置。机制分析只有每层 LoRA 权重幅度和 CKA：LoRA 更新集中在晚层（24–31），CKA 也在晚层下降 | 没有对查询做逐层干预，也没有 patching。任务是有害内容安全遗忘，不是事实知识 |
| SMU ([2607.18615](https://arxiv.org/abs/2607.18615)) | 报告"语言骨干忘了，接回 VLM 给图像后能恢复"；提出双层元学习修复 | 没有内部机制分析 |
| PAVA ([2608.30649](https://arxiv.org/abs/2608.30649)) | 在**遗忘前**模型上做因果追踪、权重移植、Fisher 重叠。LLaVA-1.5 上，名字在 L0–L5 MLP 处理，人脸身份在 L7–L12 MLP 处理，两段几乎不重叠 | 没有分析遗忘后的跨模态泄漏 |
| MIP-Editor ([2511.06793](https://arxiv.org/abs/2511.06793)) | 按模态定位神经元路径，画遗忘前后的逐层激活残差 | 没有检验泄漏的因果机制 |
| IDUnlearn-Bench ([2609.33481](https://arxiv.org/abs/2609.33481)) | 行为层面：直接答案被压制后，仍存在其他身份访问路径 | 无内部分析 |
| UNLINK-VL ([2608.03791](https://arxiv.org/abs/2608.03791)) | 真实知识上的行为基准，文本问法与图像问法严格配对：T→M 迁移差于 M→T | 无内部分析。`TODO(confirm)`：此前转述的 Qwen3-VL 数字（T→M 惩罚均值 21.2、M→T 12.2；范围 15.9–27.4 与 6.4–19.2）须对照原文表格核实后才能保留，核实前不在正文引用 |
| Multilingual Unlearning ([2606.03291](https://arxiv.org/abs/2606.03291)) | 纯文本 LLM 跨语言遗忘：早中层共享表示基本不变，遗忘主要作用在晚层解码层 | 不涉及模态 |
| UDS ([2605.24614](https://arxiv.org/abs/2605.24614)) | 用跨模型 activation patching 度量遗忘深度 | 只做纯文本 LLM；本项目借用其方法 |

此前调研中的其他跨模态遗忘工作都停在行为层面或"方法内定位"（定位只用来决定改哪些参数，不用来解释泄漏）：MMUnlearner、CAGUL、MLLMU-Bench、CLEAR、VKD、MANU、CMJU、POPS、Chakraborty et al.，以及鲁棒性研究 2605.26992。唯一在遗忘后的 MLLM 上用 logit lens 的是 UnLOK-VQA：中间层仍残留知识，但它没有比较两条路线 ([UnLOK-VQA](https://www.arxiv.org/pdf/2505.01456))。这些工作的链接与具体发现见 git 历史中本文件的上一版（`git show c38ce49:research.md`），引用前按主表标准复核。

各作者给出的泄漏解释可归为五类，没有一类经过干预检验：LLM 是瓶颈、一切汇入语言空间；跨模态纠缠或走不同通路；身份存储在视觉编码器或 projector 中；架构耦合程度不同；遗忘只是抑制输出 token 或拒答。

⚠abs：VGID 2606.00105、Knowledge Holes 2608.01849、ICCV 2025 TMKE 与 ICML 2026 UMM 跨模态编辑论文只读了摘要或标题；Knowledge Vector Weakening 2601.21794 未读。UNLINK-VL 作者署名在两份笔记中不一致，需核对。OpenReview 上 ICLR 2027 的在审稿件未检索，投稿前须排查并发工作。

## 贡献定位

已有工作报告了跨模态泄漏和 T→M/M→T 的不对称（行为层面：UNLINK-VL、SMU、IDUnlearn-Bench、CrossInf），或在遗忘前的模型上做了路线定位（PAVA、MIP-Editor），或只给出相关性证据（CrossInf 的 LoRA 幅度与 CKA）。**本项目首次在遗忘后的 VLM 上，用因果干预区分泄漏的几种竞争机制，并据此解释不对称。** 不声称"首次发现跨模态泄漏"。

## 图像路线在 LLaVA 中约 L20 才完成实体识别

VLM 召回机制的证据相当一致。Venhoff et al. 用 attribution patching 找到两个召回站点：早层 entity-token 处的 MLP+attn，以及中晚层 last-token 处的 MLP+attn。在 LLaVA-1.5-7B/13B 和 LLaVA-MORE 上，图像实体**只激活第二个站点**；Gemma-3-12B 与 Qwen2.5-VL-7B 两个站点都用。把 LLM 骨干的早层 entity-token MLP 输出注入 VLM，可恢复约 35% 的差距；"先说出实体名"的 CoT 对 Pixtral 能完全弥合差距，对 LLaVA-1.5-7B 无效 ([Venhoff et al.](https://arxiv.org/html/2512.03276v1)；venue 可能是 workshop，未核实)。Cohen et al. 的 cross-patching 显示，LLaVA-1.5-7B 的图像身份在 **L20** 才确定（SigLIP 为 L17，CLIP 为 L24），图像 token 的 attention 在 L15–20 最关键。文本与图像的准确率差距在 LLaVA-1.5-7B 上为 0.177，在 **Qwen2-VL 上仅 0.043**。早识别实体（<L5）的差距依然存在，说明瓶颈在于身份是否以"名字兼容"的格式出现在 subject 位置，而不只在于何时出现 ([Cohen et al.](https://arxiv.org/html/2412.14133v3))。Nikankin et al. 在 Qwen2-VL、Pixtral 和 Gemma-3 上发现，视觉与文本电路只共享约 18% 的组件，data position 的重叠接近 0，而 query 和 generation 子电路可以跨模态互换且保持 faithfulness ([Nikankin et al.](https://arxiv.org/html/2506.09047v3))。Neo et al. 的 logit lens 显示，LLaVA-1.5 的物体信息在约 L25.7/33 才可解码，last token 主要在 L15–24 从物体 patch 读取信息 ([Neo et al.](https://arxiv.org/html/2410.07149v2))。

PAVA 在**遗忘前**的 LLaVA-1.5 上用因果追踪、权重移植和 Fisher 重叠给出了两条路线的定位：名字在 L0–L5 的 MLP 处理，人脸身份在 L7–L12 的 MLP 处理，两段几乎不重叠 ([PAVA](https://arxiv.org/abs/2608.30649))。它没有分析遗忘之后的跨模态泄漏，但它给出的路线分离正是下文 H1′ 的前提。

综合起来：**下游的 query/answer 提取电路是共享的，分歧在上游的 data-token 处理**，汇合层随模型而变。Basu et al. 报告 LLaVA 早层 MLP（约 L1–4）有因果作用，并且用图像输入在 L2 做的编辑可以泛化，但其中被破坏的"视觉约束"多为文本短语，与 Venhoff 的设定不可直接比较 ([Basu et al.](https://arxiv.org/html/2406.04236v1))。⚠ LatentLens 声称 logit lens 严重低估了视觉 token 的可解释性，这一点只读了摘要 ([LatentLens](https://arxiv.org/abs/2602.00462v1))。另一缺口是：Venhoff 关于"早层复用"的证据针对 Qwen2.5-VL，**而不是用户计划使用的 Qwen2-VL**；Qwen2-VL 只有 Cohen 的 0.043 小差距和 Nikankin 的电路证据作间接支持。

## LLM 遗忘改的是晚层提取，不是存储

文本 LLM 方面的证据也很一致：GA、GradDiff、NPO 和 DPO 都在压制知识，而非删除。Hong et al. 显示 MLP value vectors 几乎不变，只恢复它们时 KRS<0.001；**只恢复最后 2–3 层的 MLP coefficients，KRS≥0.8** ([Hong et al. 2024](https://arxiv.org/html/2410.06606v2))。ConceptVectors 中各方法把 target QA 压到 0.25 以下，但概念向量的 Jaccard 仍>0.96 ([Hong et al. EMNLP'25](https://arxiv.org/html/2406.11614v2))。Guo et al. 指出非定位微调"mainly change the extraction mechanisms"；早层 fact-lookup 定位（Gemma/Llama 的 L2–8）必须人为施加，换来的 MCQ 召回是 0.11，而 causal tracing 定位为 0.85 ([Guo et al.](https://arxiv.org/html/2410.12949v2))。可恢复性证据同样充分：10 条无关样本即可让 NPO 恢复到 52.7%，RMU 在消融拒答方向后恢复到 64.7% ([Łucki et al.](https://arxiv.org/pdf/2409.18025))；4-bit 量化让遗忘知识的留存率从 21% 升到 83% ([Zhang et al.](https://arxiv.org/html/2410.16454v3))；RTT 下预训练知识恢复>88%，但**微调注入的知识恢复<35%** ([Deeb & Roger](https://arxiv.org/html/2410.08827v3))。

与跨模态最接近的类比是多语言遗忘：遗忘后，跨语言问题表示在早中层仍高度相似，只在晚层解码处分叉；要求模型用微调语言作答，召回最多恢复 61 分 ([Xiang et al.](https://arxiv.org/pdf/2606.03291))。在 RWKU 上，multi-hop 查询残留更多（1/2/3 跳分别约 20%/27%/34%），alias 只多恢复 2.4 分，说明输入一旦映射到实体表示，名字形式就不再重要 ([Shah et al.](https://arxiv.org/html/2603.11266v1))。反方证据是：随机选 10% 的 value vectors 做遗忘，效果不差于精心定位 ([Lee et al.](https://arxiv.org/html/2505.16252v1))。⚠abs：Ssiuu 的"spurious unlearning neurons"只读了摘要 ([Yang et al.](https://arxiv.org/abs/2509.22263v1))。UDS 用跨模型 activation patching（原始模型与遗忘后模型之间逐层替换）度量遗忘深度，只做了纯文本 LLM，本项目的 `cross_model_patch` 借用这一方法 ([UDS](https://arxiv.org/abs/2605.24614))。注意以上全部是纯文本 LLM 上的结论，没有在 VLM 上复现过。

## Hypotheses & Predictions（预注册）

三个竞争假设，判别依据与预测在跑任何机制实验之前写定。θ₀ = vanilla，θᵤ = 文本遗忘（T）后的模型，除非另说。

- **H1 晚层输入形式门控**（主假设）：遗忘在晚层加了抑制，只被文本路线的特征触发；图像路线到晚层时不带这些特征，抑制不生效。
- **H1′ 早层路线分离**（对立）：遗忘主要改了早层名字 token 上的 MLP；图像路线在更深的另一段 MLP 检索身份（PAVA：L7–L12），从未经过被改动处。
- **H2 抑制—补偿**（对立）：图像路线同样被抑制，但后续层把答案补了回来。

| 证据（工具，`vlmu/mech/tools.py`） | H1 | H1′ | H2 |
|---|---|---|---|
| 图像查询下逐层答案 log-prob 差 θᵤ − θ₀（`layer_logit_diff`） | 始终≈0 | 始终≈0 | 中层为负、后层回升 |
| LoRA 分层掩码：只去掉晚层 LoRA，文本问法（`lora_layer_mask`） | 答案恢复 | 基本不恢复 | 不定 |
| ‖ΔW·x‖ 按层、按 token 组（名字 / 图像 / 问题模板 / 末位）（`delta_wx_norm`） | 晚层：文本路线大、图像路线小 | 早层名字 token 大，其余小 | 两条路线都大 |
| 跨路线 patching：文本运行的晚层表示换进图像运行（`cross_route_patch`） | 图像查询也被抑制 | 不受影响 | 不定 |
| 消融/还原"回升层"（`cross_model_patch`、`component_attribution`） | — | — | 图像泄漏明显下降 |

**判别力的限制：** 逐层 logit lens 只能把 H2 和 {H1, H1′} 分开，分不开 H1 与 H1′；后两者靠 LoRA 分层掩码和 ‖ΔW·x‖ 区分。`layer_logit_diff` 只读 θᵤ − θ₀ 的差值，不解读早层的绝对值。

**不对称的预测（H1 必须能解释 M→T 泄漏更小）：** 多模态遗忘学到的开关，其触发特征文本路线也具备（问题文本、回答格式）；文本遗忘学到的开关，其触发特征依赖名字 token，图像路线不具备。检验有两项：在 M 遗忘模型上做反向跨路线 patching（图像运行的晚层表示换进文本运行，预测文本查询被抑制）；比较两种遗忘下 ‖ΔW·x‖ 在名字 token 与问题模板 token 上的分布（预测：T 遗忘集中在名字 token，M 遗忘在问题模板 token 上也大）。

**跨路线对齐规则：** 两条路线序列长度不同（图像路线有图像 token、没有名字；LLaVA-1.5 为 576 个图像 token），只在可对齐的位置替换：两条 token 序列从末尾对齐后的最长公共后缀，即指代词之后的问题文本、选项、指令和末位 token；或只替换末位。实现在 `route_alignment`，Milestone J（J6）打印被替换位置的 token 供人工核对。

**与上一版假设的关系：** 上一版的 H0（汇合前 bypass）并入 H1′。上一版的"架构调制汇合深度"降为后续多模型比较的调节变量，不再作为竞争假设。上一版的 H3（身份存储在视觉端）在 Stage 1 不检验：LoRA 只加在语言模型线性层，不碰视觉塔和 projector。

## 执行顺序与决策规则

1. Stage 1 单格（T→M、NPO + retain NLL、一个 lr、一个 seed；`configs/<model>/pilot.yaml`），先过检查点 A（vanilla 在 forget 集上两种问法都 ≥ 0.80）和检查点 B（至少一个 checkpoint：文本问法明显下降、图像问法仍明显高于文本问法、retain 两种问法基本保持；阈值 `TODO(confirm)`）。任一不过就停下并报告。
2. 只做 `layer_logit_diff`（约半天）：图像曲线先降后升，继续做消融，检验 H2；图像曲线始终≈0，做 `lora_layer_mask` + `delta_wx_norm`，区分 H1 与 H1′。
3. 区分出结果后，再扩到完整 grid（M→T、GA/RMU、遗忘强度、第二个模型 Qwen2.5-VL）。
4. 若结论是 H1′（与 PAVA 的遗忘前定位基本一致）且无法解释不对称：在报告里明确标出"增量有限"，等待人工决定是否继续。

## 不对称方向随设定翻转，且 MLLMU-Bench 存在未被提及的混杂

在真实预训练实体上（UNLINK-VL、SMU），T→M 失败比 M→T 严重。在微调注入的虚构档案上（MMUnlearner、MANU、MLLMU-Bench、CLEAR），结果相反或混杂：MMUnlearner 的基线先删掉的是文本知识 ([MMUnlearner](https://arxiv.org/pdf/2502.11051))，CLEAR 声称纯文本遗忘就够 ([CLEAR](https://arxiv.org/html/2410.18057v1))。在安全数据上，文本→视觉迁移近乎完全 ([CrossInf](https://arxiv.org/pdf/2607.16442); [Chakraborty](https://aclanthology.org/2024.findings-emnlp.574.pdf))。一个统一的解释是：拒答类的安全目标对应粗粒度、跨模态共享的"有害性"特征，门控 key 天然不区分模态；实体事实则需要先解析身份，key 带有路线特异性。对虚构档案，Deeb & Roger 显示微调注入的知识本身更易被删 ([Deeb & Roger](https://arxiv.org/html/2410.08827v3))，所以泛化出来的路线最先垮掉（以上均为推断）。

**MLLMU-Bench 混杂没有任何论文明确指出。** 它的 vanilla 微调数据只描述为 ⟨image, question, answer⟩ 三元组，没有提到纯文本训练；每个档案有 7 道图像题和 7 道文本题，但没有说明两组题是否针对同一属性；正文写 3 个 epoch，表 3 写 4 个 ([MLLMU-Bench PDF v2](https://arxiv.org/pdf/2410.22108v2))。如果属实，文本路线召回只是对照片绑定学习的泛化，两组题测的也不是同一批事实，任何"跨模态不对称"都会混入"知识强度不同"和"题目不同"两个因素。CLEAR 也只是部分配对（3,770 道图像题对 4,000 道文本题）。UNLINK-VL 批评了合成档案，并用 "differ only in how the subject entity is accessed" 的配对设计隐式规避了这个问题，但没有点名该混杂 ([UNLINK-VL](https://arxiv.org/html/2608.03791))；CrossInf 只说各基准"not contradictory" ([CrossInf](https://arxiv.org/pdf/2607.16442))。这一推断尚未通过检查发布的训练 JSON 来证实。

**更新（E0 数据审计，`analysis/data_audit.md`）：** 发布的训练数据已核对。vanilla 微调数据只有照片加指代词一种形式（"this person" 出现 7,430 次），没有纯文本形式；forget_10 的官方 MC 中，只有 28.4% 的纯文本题问的属性与同一档案的图像题相同。所以上面的推断成立。本项目的应对是：自建同一事实两种问法的配对题，vanilla 在图文两种形式上都微调，并只在两种问法都答对的事实上评测。

## 局限

- **微调注入的虚构知识。** MLLMU-Bench 的档案是虚构的，知识由微调注入。微调注入的知识更容易被删（Deeb & Roger：RTT 下恢复 < 35%，预训练知识 > 88%），其存储与召回机制可能与预训练知识不同，本项目的结论不能直接推到真实实体上。计划在后期用 UNLINK-VL（真实知识、严格配对）补一组。
- **起点模型。** vanilla 是本仓库自己在图文两种形式上做的 LoRA 微调（只在语言模型线性层；视觉塔和 projector 冻结）。`TODO(confirm)`：是否改用 `Kyomin/unlearning_target`（MLLMU-Bench 微调并合并、只监督答案 token）。若改用，须记录该 checkpoint 训练了视觉塔，并重新检查它的文本问法是否过检查点 A，因为 MLLMU-Bench 的微调数据只有图像形式。
- **读出方式。** 主指标用 MC 字母 logit；开放式生成（关键词命中、ROUGE-L recall）和 teacher forcing 下的答案 log-prob 只作辅助。机制工具默认以正确选项字母为目标 token。
- **数值精度。** 实验在 bf16 下运行；hook 正确性检查（Milestone J）在 float32 下进行，因为 bf16 的舍入误差会逐层累积，掩盖 hook 逻辑本身是否正确（2026-10-29 的 smoke test）。

## Conclusion

行为层面的跨模态泄漏与不对称已有工作报告；遗忘前模型上的路线分离（PAVA）和纯文本 LLM 上"遗忘作用在晚层提取"的证据也已具备。缺的是在遗忘后的 VLM 上、对另一模态查询的逐层因果干预。本项目把这个缺口拆成 H1 / H1′ / H2 三个可证伪的竞争机制，先用一个单格 pilot 确认泄漏存在，再用 logit lens 区分 H2 与其余两者，最后用 LoRA 分层掩码和 ‖ΔW·x‖ 区分 H1 与 H1′，并用同一套工具检验 H1 对不对称的解释。
