# KG4ER

本仓库保存试题推荐（Exercise Recommendation, ER）实验的 KG4ER 代码。项目利用学生知识掌握度、学习进展和遗忘信息构建教育知识图谱，并通过 ConvE、TransE 等知识图谱嵌入模型生成个性化试题推荐。

代码支持：

- ConvE 主模型及 mastery、sequence、forgetting 消融实验；
- TransE、TransE-adv、RotatE、DistMult、ComplEx 对比实验；
- EB-CF、SB-CF、CBF、KCP-ER 传统推荐基线；
- ACC、NOV、Ep_sim 指标评估；
- 多数据集、五随机种子、一键运行、日志保存和断点续跑；
- 推荐解释卡片和实验结果汇总。

## 目录

```text
ER/
  README.md
  KG4ER/
    codes/          # 模型训练、测试、基线、消融和结果统计
    data/           # 数据处理与前置文件生成脚本
    tests/          # 自动化测试
    requirements.txt
```

数据集、KT 导出结果、模型 checkpoint、运行日志、论文文档和 PDF 不上传到仓库。使用前需要将本地数据放到 `ER/KG4ER/data/<dataset>/` 对应目录。

## 安装

```powershell
cd ER\KG4ER
pip install -r requirements.txt
```

## 运行示例

```powershell
cd ER\KG4ER\codes
python run_dataset_experiments.py `
  --dataset algebra2005 `
  --seeds 2024,2025,2026,2027,2028 `
  --cuda auto
```

断点续跑时使用相同的 `run-id` 或原运行配置，并增加 `--resume`。实验结果默认保存在 `ER/KG4ER/runs/<dataset>/`。
