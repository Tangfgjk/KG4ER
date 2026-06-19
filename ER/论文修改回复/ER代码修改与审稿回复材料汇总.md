# ER 代码修改与审稿回复材料汇总

生成日期：2026-06-19

说明：本文件用于论文修改、实验补充说明和审稿意见回复准备。该目录为本地论文材料目录，默认不上传 GitHub。当前实验结果尚未全部跑完，因此本文只总结代码、数据、实验设计和可回复要点，不填写最终性能结果表。

## 1. 文档目的

本次围绕 ER 创新点，对原始 KG4ER 代码进行了工程化整理和实验扩展。主要目标是回应审稿意见中关于数据集、对比模型、评价指标、可复现性、可解释性、训练/测试划分和 embedding 机制等问题。

本文重点记录：

- 相对原代码做了哪些改变。
- 新增和扩展了哪些数据集。
- 子集筛选规则是什么。
- 当前正式使用的数据规模是多少。
- 新增了哪些对比模型和消融实验。
- 新增了哪些评价指标。
- ConvE embedding、推荐边构造和 Eq. 9/Eq. 11 的关系如何解释。
- 日志、结果保存、断点续跑和多随机种子实验如何支持复现。

## 2. 相对原代码的总体改变

原代码主要围绕 Eedi-sub 数据集运行，路径、数据文件名和运行脚本中存在较多固定写法。当前代码保留原方法核心逻辑，同时做了以下扩展。

| 类别 | 原代码情况 | 当前修改 |
|---|---|---|
| 数据集 | 主要围绕 Eedi-sub，部分脚本含固定路径 | 扩展为 Eedi、Algebra 2005、Statics 2011、ASSISTments 2009 子集、XES3G5M 子集 |
| 数据处理 | 多个步骤分散执行 | 增加统一数据处理流水线和数据检查脚本 |
| 关系数量 | 关系字典依赖生成文件 | 统一为 KG4ER 风格 304 个关系 |
| 主模型 | ConvE 原逻辑可运行，但路径和保存不够规范 | 增加参数化路径、CUDA 自动选择、负采样、best/last checkpoint、日志和计时 |
| 消融实验 | 原代码中有思路，但不便直接批量运行 | 增加 `ConvE_no_seq`、`ConvE_no_forgetting`、`ConvE_no_mastery` 的自动构造和运行 |
| KGE 对比 | 原代码主要保留 TransE/TransE-adv 相关入口 | 增加 RotatE、DistMult、ComplEx，并统一评分与评估 |
| 传统 baseline | 原代码未统一纳入一键实验 | 增加 EB-CF、SB-CF、CBF、KCP-ER |
| 指标 | 原论文主要使用 Ada/ACC、Nov/NOV | 增加 Ep_sim，并统一导出 ACC/NOV/Ep_sim |
| 多随机种子 | 原代码没有统一 5 seed 汇总流程 | 默认支持 `2024,2025,2026,2027,2028`，输出 mean/std 汇总 |
| 日志与结果 | 结果文件分散 | 统一保存到 `ER/KG4ER/runs/<dataset>/<run-id>/` |
| 断点续跑 | 原代码不完整 | 一键脚本支持 `--resume`，已完成任务自动跳过，未完成任务继续 |
| 可解释性 | 原代码缺少推荐解释卡片 | 增加 ConvE Top-K 推荐解释卡片输出 |

## 3. 当前代码中主要新增或修改的文件

### 3.1 数据处理相关

| 文件 | 作用 |
|---|---|
| `ER/KG4ER/data/run_er_dataset_pipeline.py` | 数据处理总入口，支持 `prepare`、`after-kt`、`all` 等阶段 |
| `ER/KG4ER/data/prepare_kt_training_files.py` | 生成 KT 模型训练所需文件 |
| `ER/KG4ER/data/export_mastery_from_bkt.py` | 导出 `stu2know_mastery.json` |
| `ER/KG4ER/data/export_seq_from_dkt.py` | 从 pyKT DKT checkpoint 导出 `stu2know_seq.json` |
| `ER/KG4ER/data/validate_kt_exports.py` | 检查 mastery、seq、forget 等 KT 导出文件维度 |
| `ER/KG4ER/data/build_public_dataset_sequences.py` | 将公开教育数据集处理成 ER 可用序列文件 |
| `ER/KG4ER/data/build_structured_er_subset.py` | 从大数据集中筛选结构化 ER 子集 |
| `ER/KG4ER/data/step1_cal_recommend.py` | 根据 Eq. 9 风格的手工推荐分数生成 `stu2ex_recommend.json` |
| `ER/KG4ER/data/step2_createtriples.py` | 根据 KT 文件和推荐分数构造 KG 三元组 |
| `ER/KG4ER/data/create_relations_dict.py` | 生成固定 304 个 KG4ER 关系 |

### 3.2 训练、测试和评估相关

| 文件 | 作用 |
|---|---|
| `ER/KG4ER/codes/run_dataset_experiments.py` | 一键运行同一数据集下的主模型、消融实验、KGE baseline、传统 baseline |
| `ER/KG4ER/codes/run_ConvE.py` | ConvE 训练，已支持参数化路径、负采样、best/last checkpoint、resume |
| `ER/KG4ER/codes/test_ConvE.py` | ConvE 测试和推荐分数导出 |
| `ER/KG4ER/codes/score_kge_recommendations.py` | KGE 模型推荐分数导出 |
| `ER/KG4ER/codes/baseline_recommenders.py` | EB-CF、SB-CF、CBF、KCP-ER 传统 baseline |
| `ER/KG4ER/codes/evaluate_recommendations.py` | 统一计算 ACC/NOV/Ep_sim |
| `ER/KG4ER/codes/ep_sim.py` | Ep_sim 轻量模拟指标 |
| `ER/KG4ER/codes/explain_recommendations.py` | ConvE 推荐解释卡片生成 |
| `ER/KG4ER/codes/experiment_utils.py` | 实验结果、JSON、日志等通用工具函数 |

## 4. 数据集扩展情况

### 4.1 数据集来源与处理说明

| 数据集 | 来源与用途 | 当前处理方式 |
|---|---|---|
| Eedi | 原论文和原代码使用的 Eedi-sub 推荐实验数据 | 保留原 ER 数据目录 `ER/KG4ER/data/Eedi`，作为主对照数据集 |
| Algebra 2005 | PSLC DataShop/KDD Cup 教育数据，包含学生作答、题目和知识点信息 | 已处理为 `prepared_for_kt`，可直接进入 ER 训练 |
| Statics 2011 | PSLC DataShop 公开教育数据，包含静力学领域学生作答记录 | 已处理为 `prepared_for_kt`，可直接进入 ER 训练 |
| ASSISTments 2009 | ASSISTments 公开学生作答数据，完整数据较大 | 保留完整处理版，同时构造 `assist2009-sub` 作为正式实验子集 |
| XES3G5M | 大规模公开教育数据集，原始数据量较大 | 保留中间子集 `XES3G5M-sub`，进一步构造 `XES3G5M-sub-small` 作为正式实验子集 |

### 4.2 为什么需要筛选子集

筛选子集的主要原因不是为了人为优化结果，而是为了保证 ER 推荐任务中的图结构质量和实验成本可控：

- 作答过少的学生缺少足够历史信息，难以估计 mastery、sequence 和 forgetting。
- 作答过多的学生会让推荐任务偏向极长学习轨迹，训练时间明显增加，并可能影响普通学生群体的代表性。
- 极低频题目会导致题目和知识点之间的结构过稀疏，不利于构图和推荐评估。
- ASSISTments 2009 和 XES3G5M 的完整或较大处理版三元组数量明显高于 Eedi-sub，五随机种子和多 baseline 实验成本过高。
- 子集筛选保留学生、题目、知识点之间的覆盖关系，避免只按数量随机截断导致知识点缺失。

### 4.3 子集筛选规则

当前子集由 `ER/KG4ER/data/build_structured_er_subset.py` 构造。核心规则如下：

1. 按学生作答序列长度过滤，去除过短和过长序列。
2. 按题目交互次数过滤，去除过低频题目。
3. 为每个知识点尽量保留不少于指定数量的题目，维持知识点覆盖。
4. 在满足知识点覆盖后，根据题目交互频次补足目标题目数。
5. 根据最终保留题目重新筛选学生，保证学生仍有足够作答记录。
6. 按固定随机种子划分训练学生和测试学生。
7. 保持 `Q.txt` 的知识点列宽与源数据一致，避免知识点 id 因筛选发生不可解释变化。

两个正式大数据集子集的筛选参数如下。

| 子集 | 源数据目录 | 目标题目数 | 学生序列长度 | 最低题目交互 | 每知识点最少题目 | 划分比例 | 随机种子 |
|---|---|---:|---|---:|---:|---:|---:|
| `assist2009-sub` | `assist2009/prepared_for_kt` | 1,987 | 10-200 | 1 | 3 | 0.75 | 2026 |
| `XES3G5M-sub-small` | `XES3G5M-sub/prepared_for_kt` | 1,928 | 10-400 | 20 | 3 | 0.75 | 2026 |

说明：题目数没有设置成 2000 这类整数字，而是分别保留 1,987 和 1,928 道题，避免看起来像简单按固定整数硬截断。两个子集的最终三元组规模与 Eedi-sub 接近，但仍保留不同数据来源和不同交互密度。

## 5. 当前正式使用的数据规模

正式建议优先使用以下五个数据集进行论文实验：

- `Eedi`
- `algebra2005`
- `statics2011`
- `assist2009-sub`
- `XES3G5M-sub-small`

当前 ER 训练实际使用的数据规模如下。

| 数据集 | 学生数 | 训练学生 | 测试学生 | 题目数 | 知识点数 | 有效交互 | 实体数 | 关系数 | 训练三元组 | 测试三元组 | 总三元组 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Eedi | 935 | 701 | 234 | 948 | 57 | 1,222,125 | 1,940 | 304 | 751,472 | 248,508 | 999,980 |
| Algebra 2005 | 574 | 430 | 144 | 1,084 | 112 | 607,025 | 1,340 | 304 | 142,344 | 47,088 | 189,432 |
| Statics 2011 | 331 | 248 | 83 | 633 | 97 | 113,992 | 813 | 304 | 51,894 | 17,367 | 69,261 |
| ASSISTments 2009-sub | 1,717 | 1,287 | 430 | 1,987 | 149 | 53,163 | 2,566 | 304 | 738,990 | 246,780 | 985,770 |
| XES3G5M-sub-small | 1,391 | 1,043 | 348 | 1,928 | 408 | 408,242 | 2,684 | 304 | 718,794 | 238,728 | 957,522 |

备用但不建议作为第一轮正式五随机种子实验的数据集：

| 数据集 | 用途 | 学生数 | 题目数 | 知识点数 | 有效交互 | 实体数 | 关系数 | 总三元组 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ASSISTments 2009 完整处理版 | 备用泛化实验或附录 | 4,163 | 17,751 | 149 | 283,105 | 18,941 | 304 | 18,796,809 |
| XES3G5M-sub | 中间子集，偏大 | 3,200 | 2,000 | 408 | 1,232,662 | 3,024 | 304 | 1,739,276 |

注意：不同论文或参考代码中的数据规模可能因原始文件版本、缺失 KC 过滤、题目 id 合并、是否只统计有效作答、是否按学生过滤而不同。正式论文中建议使用“本研究实际使用的数据规模”，并在数据处理部分说明过滤规则。

## 6. 数据处理产物和数据流向

每个 ER 数据集最终需要以下关键文件。

| 文件 | 来源 | 用途 |
|---|---|---|
| `Q.txt` | 数据集处理阶段生成 | 题目-知识点矩阵 |
| `sequence_interactions.csv` 或 `interactions.csv` | 数据集处理阶段生成 | 学生作答序列、传统 baseline 和 KT 导出依赖 |
| `train_sequences.csv` | 数据集处理阶段生成 | KT 模型训练和学生划分 |
| `test_sequences.csv` | 数据集处理阶段生成 | 测试学生序列和 KT 导出 |
| `stu2know_mastery.json` | BKT/mastery 导出 | 学生-知识点掌握度 |
| `stu2know_seq.json` | pyKT DKT 导出 | 学生-知识点 sequence/progress 表征 |
| `stu2know_forget.json` | forgetting 计算 | 学生-知识点遗忘度 |
| `stu2ex_forget.json` | forgetting 计算 | 学生-题目遗忘度 |
| `stu2ex_recommend.json` | `step1_cal_recommend.py` | Eq. 9 风格手工推荐分数 |
| `triples.txt` | `step2_createtriples.py` | 训练学生相关 KG 三元组 |
| `test_triples.txt` | `step2_createtriples.py` | 测试学生认知状态 KG 三元组 |
| `entities.dict` | `preprocess.py` | 实体 id 映射 |
| `relations.dict` | `create_relations_dict.py` | 固定 304 个关系映射 |
| `<dataset>_uid_kc_response.txt` | 数据处理阶段生成 | NOV 指标和部分 baseline 使用 |

完整数据流可以概括为：

```text
原始作答数据
  -> 题目-知识点矩阵 Q.txt
  -> 学生序列 train/test_sequences.csv
  -> KT/BKT/遗忘计算
  -> stu2know_mastery / stu2know_seq / stu2know_forget / stu2ex_forget
  -> Eq. 9 风格推荐倾向 stu2ex_recommend
  -> KG 三元组 triples.txt / test_triples.txt
  -> ConvE/KGE/传统 baseline 训练与评分
  -> ACC/NOV/Ep_sim 指标与解释卡片
```

## 7. 推荐边构造与 Eq. 9/Eq. 11 的关系

### 7.1 Eq. 9 的作用

Eq. 9 是构图阶段的手工综合推荐分数，不是最终模型预测分数。代码位置是：

```text
ER/KG4ER/data/step1_cal_recommend.py
```

它根据学生状态和题目特征计算学生对题目的推荐倾向，主要包含：

- mastery：学生对题目涉及知识点的掌握度。
- forgetting：学生对题目或相关知识点的遗忘度。
- sequence/progress：学生学习进展信息。

当前默认设置中，代码会计算 sequence 对应的 `term2`，但默认推荐分数主要使用 mastery 和 forgetting 两项。sequence/progress 没有丢弃，而是通过 `pkc` 关系进入 KG，让 ConvE 在图结构中学习。

### 7.2 rec 正样本三元组的构造

代码位置是：

```text
ER/KG4ER/data/step2_createtriples.py
```

默认流程：

1. 先按学生划分训练学生和测试学生。
2. 对训练学生，根据 `stu2ex_recommend.json` 中的手工推荐倾向选择 Top-K 题目。
3. 将这些学生-题目对构造成正推荐边：

```text
uid  rec  exercise
```

4. 测试学生不构造测试 `rec` 标签边，只保留其认知状态三元组，用于获得测试学生 embedding。

因此，Eq. 9 回答的问题是：哪些学生-题目对可以作为训练图中的正推荐关系。

### 7.3 Eq. 11 的作用

Eq. 11 是 ConvE 模型预测阶段的打分。训练完成后，模型对候选题计算：

```text
score(uid, rec, exercise)
```

该分数用于最终排序和推荐。它回答的问题是：训练后的 ConvE 模型认为应该给该学生推荐哪些题。

所以 Eq. 9 和 Eq. 11 不是两个同时相加的最终公式，而是前后两个阶段：

```text
Eq. 9: 构造训练图中的正推荐边 rec
Eq. 11: ConvE 在训练后的图嵌入空间中预测推荐分数
```

## 8. 训练/测试划分与测试学生 embedding 机制

当前实验不是冷启动推荐。训练/测试划分按学生进行，但测试学生仍然被保留在实体集合中，并且拥有自己的认知状态三元组：

```text
kc  mlkc_xx  uid
kc  pkc_xx   uid
exercise  exfr_xx  uid
```

这意味着测试学生虽然没有训练 `rec` 标签边，但并不是完全未知实体。ConvE 可以通过这些测试学生的认知状态边更新或使用其 embedding，再对：

```text
score(uid, rec, exercise)
```

进行推荐评分。

为了与原代码协议保持一致，当前 ConvE 默认仍会把 `test_triples.txt` 中的测试学生认知状态三元组加入训练上下文：

```text
--conve-include-test-triples
```

这不等于泄露测试推荐标签，因为 `test_triples.txt` 中没有测试学生的 `rec` 正样本边。它的作用是给测试学生提供认知状态信息，解决按学生划分后测试学生 embedding 可用的问题。若要做严格诊断实验，也保留了：

```text
--conve-exclude-test-triples
```

但正式复现实验建议使用默认设置，以保持和原代码一致。

## 9. 模型与实验设置

### 9.1 主模型和消融实验

| 实验名称 | 类型 | 含义 |
|---|---|---|
| `ConvE_full` | 主模型 | 使用 mastery、sequence/progress、forgetting 和 rec 关系 |
| `ConvE_no_seq` | 消融实验 | 移除 `pkc` 关系，考察 sequence/progress 贡献 |
| `ConvE_no_forgetting` | 消融实验 | 移除 `exfr` 关系，考察 forgetting 贡献 |
| `ConvE_no_mastery` | 消融实验 | 移除 `mlkc` 关系，考察 mastery 贡献 |

消融实验不是改动源数据目录，而是在每次运行目录下生成临时数据变体：

```text
ER/KG4ER/runs/<dataset>/<run-id>/data_variants/
```

这样可以保证原始数据处理产物不被覆盖。

### 9.2 KGE 对比模型

| 模型 | 说明 |
|---|---|
| `TransE` | 原论文已有 KGE 对比模型之一 |
| `TransE-adv` | 原论文已有对比模型，使用 adversarial negative sampling |
| `RotatE` | 新增 KGE baseline |
| `DistMult` | 新增 KGE baseline |
| `ComplEx` | 新增 KGE baseline |

KGE baseline 使用统一的 KG 三元组输入，最终也导出 `uid-exercise` 推荐分数，再交给同一评价脚本计算 ACC/NOV/Ep_sim。KGE 训练负采样沿用 `run.py` 和 `dataloader.py` 的通用 KGE 方式，对三元组进行 head-batch/tail-batch 采样；它不是只对 `rec` 关系采样。

### 9.3 传统推荐 baseline

| 模型 | 说明 |
|---|---|
| `EB-CF` | Exercise-based collaborative filtering |
| `SB-CF` | Student-based collaborative filtering |
| `CBF` | Content-based filtering，基于题目知识点内容相似度 |
| `KCP-ER` | 基于知识点掌握/练习状态的教育推荐 baseline |

传统 baseline 主要依赖作答序列、题目-知识点矩阵和学生历史，不进行深度模型训练，因此不需要像 ConvE/KGE 一样按 epoch 或 step 训练。

## 10. ConvE 训练机制的当前修改

当前 ConvE 仍保留原方法框架，但工程层面做了如下修改。

| 设置 | 当前值或机制 |
|---|---|
| 默认 epoch | 25 |
| batch size | 1024 |
| learning rate | 0.001 |
| dropout | input 0.2、hidden 0.2、feature 0.3 |
| CUDA | 一键脚本默认 `--cuda auto`，优先使用 GPU |
| checkpoint | 每轮保存 `last.pt`，最优 loss 保存 `best.pt` |
| 测试加载 | 优先加载 best checkpoint，缺失时回退 last |
| 断点续跑 | `--resume` 从 `last.pt` 继续 |
| 负采样 | 默认 `negative_ratio=5` |

### 10.1 ConvE 负采样

当前 ConvE 使用显式负采样，但只对 `rec` 推荐关系做负采样。

原因：

- 最终推荐任务预测的是 `score(uid, rec, exercise)`，所以 `rec` 负样本最直接服务于推荐排序。
- `mlkc`、`pkc`、`exfr` 是离散化后的认知状态上下文关系，对其随机替换学生容易产生 false negative。
- 在一些子集上，状态关系候选负样本空间可能不足或语义不稳定，容易导致训练失败或不合理约束。

当前规则：

```text
正样本: (uid, rec, exercise_positive), label = 1
负样本: (uid, rec, exercise_negative), label = 0
```

其中 `exercise_negative` 从同一数据集所有题目中采样，并过滤该学生已有正推荐题，保证类型合法且避免显然的 false negative。

## 11. 评价指标

当前统一评价脚本为：

```text
ER/KG4ER/codes/evaluate_recommendations.py
```

### 11.1 ACC/Ada

ACC 与原文 Ada 含义一致，衡量推荐题目与学生当前掌握状态是否匹配。实现上会计算推荐题涉及知识点的掌握度，并与目标掌握阈值比较：

```text
score = 1 - abs(target_mastery - exercise_mastery)
```

默认目标掌握阈值为：

```text
target_mastery = 0.8
```

### 11.2 NOV/Nov

NOV 衡量推荐题的新颖性，基于学生历史作答知识点与推荐题知识点之间的重合程度。重合越低，新颖性越高。当前实现考虑了学生历史中知识点出现情况，并输出 Top-K 平均值和学生级标准差。

### 11.3 Ep_sim

Ep_sim 是为回应“评价指标不足”新增的有效性模拟指标。它模拟学生完成 Top-K 推荐题后，对相关知识点 mastery 的潜在提升：

```text
updated_mastery = mastery + learning_gain * (1 - mastery)
```

默认：

```text
learning_gain = 0.1
ep_top_k = 10
```

说明：当前 Ep_sim 是轻量模拟版本，不是重新调用完整 KT 模型在线预测。论文中应明确写为 simulator-based effectiveness proxy，后续如时间允许可进一步替换为完整 KT simulator。

### 11.4 Top-K 设置

当前默认输出：

```text
K = 10, 15, 20, 30, 50, 75, 100
```

每个模型每个随机种子会输出：

```text
metrics.json
metrics.csv
ep_sim.json
uid_ex_scores.json
```

其中单个 seed 的 `metrics.csv` 中的 std 是学生级标准差。论文最终的五随机种子标准差应使用：

```text
ER/KG4ER/runs/<dataset>/<run-id>/summaries/dataset_summary_stats.csv
```

该文件中的 std 才是不同随机种子之间的标准差。

## 12. 五随机种子、显著性和复现实验

默认随机种子：

```text
2024,2025,2026,2027,2028
```

这些随机种子主要影响：

- 模型参数初始化。
- mini-batch 顺序。
- ConvE `rec` 负样本采样。
- KGE 负样本采样。
- 部分 baseline 中涉及随机过程的初始化或抽样。

数据集前置文件和学生训练/测试划分已经固定，不会因为训练阶段随机种子改变而重新划分。这样可以保证不同 seed 比较的是模型训练随机性，而不是数据划分差异。

论文中建议报告：

```text
mean ± std
```

其中 mean 和 std 来自五个随机种子的模型结果汇总。显著性检验可在所有实验跑完后基于五个 seed 的对应模型指标进行配对检验。

## 13. 可解释性设计

当前可解释性主要针对主模型 `ConvE_full`。代码会为少量学生的 Top-K 推荐题生成解释卡片，输出目录为：

```text
ER/KG4ER/runs/<dataset>/<run-id>/ConvE_full/seed<seed>/explanations/
```

每张解释卡片包含：

- 学生 id。
- 推荐题目 id。
- 推荐题涉及知识点。
- 学生在相关知识点上的 mastery。
- 学生在相关知识点上的 sequence/progress。
- 学生对题目或相关知识点的 forgetting。
- ConvE 推荐分数。
- 一句自然语言解释。

该设计与参考论文中的解释思路一致：不是解释卷积层或 embedding 维度本身，而是从教育语义因素解释为什么该题适合当前学生。后续论文中还可以补充专家问卷或教师评价，用于支撑解释的合理性。

## 14. 运行管理、日志和断点续跑

一键实验入口：

```text
ER/KG4ER/codes/run_dataset_experiments.py
```

结果保存结构：

```text
ER/KG4ER/runs/<dataset>/<run-id>/<model>/seed<seed>/
```

主要输出：

| 文件或目录 | 说明 |
|---|---|
| `train.log` | 训练日志 |
| `test.log` | 测试和打分日志 |
| `eval.log` | 指标计算日志 |
| `timing.json` | training、inference_without_cache、evaluation_metric 分段计时 |
| `status.json` | 当前任务状态 |
| `config.json` | 单模型单 seed 配置 |
| `best.pt` | ConvE 最优 checkpoint |
| `last.pt` | ConvE 最近一轮 checkpoint |
| `checkpoint` | KGE 模型断点文件 |
| `summaries/` | 一个数据集下所有模型和 seed 的汇总结果 |

断点续跑机制：

- 如果模型和 seed 已经完成并存在评估文件，一键脚本会跳过。
- 如果 ConvE 中断，使用 `--resume` 时从 `last.pt` 继续。
- 如果 KGE 中断，使用 `--resume` 时从 KGE 的 `checkpoint` 继续。
- 如果用户指定 `--run-id`，则续跑该 run-id 对应目录。
- 如果只写 `--resume` 不写 `--run-id`，脚本会选择该数据集下最近一次运行目录。

## 15. 可直接用于审稿回复的要点

### 15.1 关于训练/测试划分和测试学生 embedding

可以回复：

```text
本研究不设置冷启动推荐场景。测试学生虽然没有用于构造训练推荐标签边，但其认知状态边仍保留在图中，包括 mastery、progress 和 forgetting 相关关系。ConvE 因此可以基于测试学生的认知状态获得 learner embedding，并进一步计算 score(uid, rec, exercise)。测试集中没有引入测试学生的 rec 标签边，因此不会泄露推荐标签。
```

### 15.2 关于推荐关系 rec 是否由 Eq. 9 得到

可以回复：

```text
Eq. 9 用于构图阶段，依据 mastery、forgetting 以及 progress 信息计算学生-题目的推荐倾向。代码随后根据该分数选择 Top-K 题目，构造正推荐三元组 (uid, rec, exercise)。Eq. 9 不是最终排序公式，而是用于生成 KG 中的正推荐边。
```

### 15.3 关于 Eq. 9 与 ConvE 分数关系

可以回复：

```text
Eq. 9 和 ConvE 分数处于两个阶段。Eq. 9 负责生成训练图中的 rec 正边；ConvE 在包含认知状态关系和推荐关系的 KG 上学习实体与关系表示，测试阶段使用 ConvE 的 score(uid, rec, exercise) 对候选题排序。因此 ConvE 不是简单复现 Eq. 9，而是在图结构中进一步学习学生、知识点、题目和推荐关系之间的表示。
```

### 15.4 关于 baseline 过窄

可以回复：

```text
修订后除原有 TransE-ER 和 TransE-adv-ER 外，新增 RotatE、DistMult、ComplEx 三类 KGE baseline，以及 EB-CF、SB-CF、CBF、KCP-ER 四类传统推荐 baseline。同时保留 ConvE 的三个消融变体，用于分析 mastery、progress 和 forgetting 关系的贡献。
```

### 15.5 关于评价指标不足

可以回复：

```text
修订后保留 ACC/Ada 和 NOV/Nov，并新增 Ep_sim 作为推荐有效性的模拟指标。Ep_sim 模拟学生完成推荐题后相关知识点 mastery 的潜在提升，用于补充只看适配性和新颖性的不足。
```

### 15.6 关于数据集泛化性

可以回复：

```text
修订后实验从单一 Eedi-sub 扩展到五个数据集，包括 Eedi、Algebra 2005、Statics 2011、ASSISTments 2009-sub 和 XES3G5M-sub-small。其中 ASSISTments 2009 和 XES3G5M 原始规模较大，为保证实验成本和图结构质量，按学生序列长度、题目交互频次和知识点覆盖规则构造结构化子集。
```

### 15.7 关于标准差和显著性

可以回复：

```text
修订实验采用五个随机种子 2024、2025、2026、2027、2028 重复运行，报告 mean ± std。实验完成后可进一步基于五个 seed 的成对结果进行显著性检验。
```

### 15.8 关于效率实验

可以回复：

```text
修订后效率统计拆分为 training time、inference time without cache 和 evaluation metric time，避免将训练、推理和指标计算混在一起造成不公平比较。
```

### 15.9 关于可解释性

可以回复：

```text
修订后为 ConvE_full 的 Top-K 推荐生成解释卡片，从推荐题知识点、学生 mastery、progress、forgetting 和 ConvE 分数等方面解释推荐原因。该解释方式强调教育语义因素，而不是直接解释 embedding 维度。
```

## 16. 当前仍待补充的内容

以下内容需要等实验全部完成后再补：

- 五个数据集上所有模型的最终指标表。
- 五随机种子的 mean ± std。
- 主模型和 baseline 的显著性检验。
- 消融实验最终结果表。
- 效率实验最终表格。
- 可解释性案例图和问卷或专家评价结果。
- 论文中 Figures 2、3、4 和热力图的最终重画版本。

## 17. 建议论文修改落点

建议在论文中新增或改写以下部分：

| 论文位置 | 建议补充内容 |
|---|---|
| 数据集部分 | 增加五个数据集说明、过滤规则和最终规模表 |
| 方法部分 | 明确 Eq. 9 构造 rec 边，Eq. 11 是 ConvE 预测分数 |
| 实验设置 | 增加五随机种子、超参数、负采样、CUDA/运行环境、断点策略 |
| baseline 部分 | 增加 KGE baseline 和传统 baseline 的统一说明 |
| 指标部分 | 增加 Ep_sim，说明其是模拟有效性指标 |
| 结果部分 | 增加多数据集结果、mean ± std、显著性检验 |
| 消融实验 | 报告 no_seq、no_forgetting、no_mastery |
| 可解释性 | 增加推荐解释卡片和人工评价设计 |
| 效率实验 | 拆分训练时间、无缓存推理时间和指标计算时间 |

