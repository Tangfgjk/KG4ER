import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ER_ROOT = Path(__file__).resolve().parents[2]


REFERENCE_TABLE2: Dict[str, Dict[str, int]] = {
    "ASSISTments 2009": {
        "knowledge_concepts": 124,
        "students": 4217,
        "exercises": 17751,
        "interactions": 346860,
        "entities": 20826,
        "relations": 304,
        "triples": 53144559,
    },
    "Algebra 2005": {
        "knowledge_concepts": 112,
        "students": 574,
        "exercises": 1085,
        "interactions": 809694,
        "entities": 1598,
        "relations": 304,
        "triples": 528919,
    },
    "Statics 2011": {
        "knowledge_concepts": 87,
        "students": 333,
        "exercises": 1224,
        "interactions": 194947,
        "entities": 1544,
        "relations": 304,
        "triples": 328064,
    },
}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _split_kcs(series: pd.Series) -> int:
    concepts = set()
    for value in series.dropna().astype(str):
        for concept in value.split("~~"):
            concept = concept.strip()
            if concept and concept.lower() != "nan":
                concepts.add(concept)
    return len(concepts)


def _assist_observed() -> Dict[str, Any]:
    path = ER_ROOT / "KG4ER" / "data" / "assist2009" / "raw-data" / "skill_builder_data_corrected_collapsed.csv"
    df = pd.read_csv(
        path,
        usecols=["user_id", "problem_id", "skill_id", "skill_name"],
        encoding="latin1",
        low_memory=False,
    )
    valid_kc = df[df["skill_id"].notna()]
    return {
        "dataset": "ASSISTments 2009",
        "raw_file": str(path),
        "students": int(df["user_id"].nunique()),
        "exercises_all_rows": int(df["problem_id"].nunique()),
        "exercises_valid_kc_rows": int(valid_kc["problem_id"].nunique()),
        "interactions_all_rows": int(len(df)),
        "interactions_valid_kc_rows": int(len(valid_kc)),
        "concepts_skill_id": int(df["skill_id"].nunique(dropna=True)),
        "concepts_skill_name": int(df["skill_name"].nunique(dropna=True)),
        "alignment_note": "论文表使用 124 个知识点；当前公开 raw 中 skill_id 为 149、skill_name 为 101，需要沿用论文/参考代码的知识点清洗口径。",
    }


def _algebra_observed() -> Dict[str, Any]:
    path = (
        ER_ROOT
        / "KG4ER"
        / "data"
        / "algebra2005"
        / "raw-data"
        / "algebra_2005_2006 (1)"
        / "algebra_2005_2006_train.txt"
    )
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["Anon Student Id", "Problem Name", "KC(Default)"],
        low_memory=False,
    )
    valid_kc = df[df["KC(Default)"].notna()]
    return {
        "dataset": "Algebra 2005",
        "raw_file": str(path),
        "students": int(df["Anon Student Id"].nunique()),
        "exercises_all_rows": int(df["Problem Name"].nunique()),
        "exercises_valid_kc_rows": int(valid_kc["Problem Name"].nunique()),
        "interactions_all_rows": int(len(df)),
        "interactions_valid_kc_rows": int(len(valid_kc)),
        "concepts_split_kc_default": _split_kcs(df["KC(Default)"]),
        "alignment_note": "论文表为 1,085 道题；当前 raw 的 Problem Name 为 1,084，参考代码 Q.txt 也是 1,084 行，论文值可能包含保留/占位题目。",
    }


def _statics_observed() -> Dict[str, Any]:
    path = ER_ROOT / "KG4ER" / "data" / "statics2011" / "raw-data" / "statics2011" / "AllData_student_step_2011F.csv"
    df = pd.read_csv(
        path,
        usecols=["Anon Student Id", "Problem Name", "Step Name", "KC (F2011)"],
        low_memory=False,
    )
    valid_kc = df[df["KC (F2011)"].notna()]
    exercise_all = df["Problem Name"].astype(str) + "::" + df["Step Name"].astype(str)
    exercise_valid = valid_kc["Problem Name"].astype(str) + "::" + valid_kc["Step Name"].astype(str)
    return {
        "dataset": "Statics 2011",
        "raw_file": str(path),
        "students": int(df["Anon Student Id"].nunique()),
        "exercises_problem_only": int(df["Problem Name"].nunique()),
        "exercises_problem_step_all_rows": int(exercise_all.nunique()),
        "exercises_problem_step_valid_kc_rows": int(exercise_valid.nunique()),
        "interactions_all_rows": int(len(df)),
        "interactions_valid_kc_rows": int(len(valid_kc)),
        "concepts_kc_f2011": int(df["KC (F2011)"].nunique(dropna=True)),
        "alignment_note": "论文表使用 87 个知识点；当前 raw 的 KC (F2011) 直接统计为 97，说明论文使用了额外的知识点合并/过滤映射。",
    }


def observed_raw_stats() -> List[Dict[str, Any]]:
    return [_assist_observed(), _algebra_observed(), _statics_observed()]


def reference_code_stats() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in ER_ROOT.rglob("KG4EX.Exercise-Recommendation-main"):
        base = root / "data" / "algebra2005"
        if not base.exists():
            continue
        row: Dict[str, Any] = {"dataset": "Algebra 2005 reference code", "path": str(base)}
        for name in ["Q.txt", "entities.dict", "relations.dict", "triples.txt", "test_triples.txt"]:
            file = base / name
            if file.exists():
                with file.open("r", encoding="utf-8", errors="ignore") as f:
                    row[name] = sum(1 for _ in f)
        rows.append(row)
    return rows


def write_markdown(output: Path, observed: List[Dict[str, Any]], ref_code: List[Dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# ER 数据集 Table 2 对齐统计说明")
    lines.append("")
    lines.append("本文档用于回答“为什么当前统计和论文 Table 2 差很多”以及“论文实验表应按什么口径对齐”。")
    lines.append("")
    lines.append("结论：论文 Table 2 是原始数据集/参考论文 KG 构图口径；我们当前 `prepared_for_kt` 目录是为了 ER 非冷启动训练而生成的可训练产物口径，两者不能直接混用。论文中复现实验的数据集概览建议按下表对齐。")
    lines.append("")
    lines.append("## 1. 论文 Table 2 对齐口径")
    lines.append("")
    lines.append("| Dataset | Knowledge concepts | Students | Exercises | Interactions | Entities | Relations | Triples |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset, row in REFERENCE_TABLE2.items():
        lines.append(
            f"| {dataset} | {_fmt(row['knowledge_concepts'])} | {_fmt(row['students'])} | {_fmt(row['exercises'])} | "
            f"{_fmt(row['interactions'])} | {_fmt(row['entities'])} | {_fmt(row['relations'])} | {_fmt(row['triples'])} |"
        )
    lines.append("")
    lines.append("这里的 `Interactions` 使用原始日志行数；`Relations=304` 是参考论文 KG4EX/KG4ER 的固定离散关系集合，即 `mlkc0.00-1.00`、`pkc0.00-1.00`、`exfr0.00-1.00` 各 101 个加 `rec`，共 304 个。")
    lines.append("")
    lines.append("## 2. 本地原始数据可核对项")
    lines.append("")
    lines.append("| Dataset | Raw students | Raw exercises | Raw interactions | Raw concepts | Valid-KC interactions | 对齐说明 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in observed:
        if row["dataset"] == "ASSISTments 2009":
            raw_exercises = row["exercises_valid_kc_rows"]
            raw_concepts = f"skill_id={row['concepts_skill_id']}; skill_name={row['concepts_skill_name']}"
        elif row["dataset"] == "Algebra 2005":
            raw_exercises = row["exercises_all_rows"]
            raw_concepts = row["concepts_split_kc_default"]
        else:
            raw_exercises = row["exercises_problem_step_all_rows"]
            raw_concepts = row["concepts_kc_f2011"]
        lines.append(
            f"| {row['dataset']} | {_fmt(row['students'])} | {_fmt(raw_exercises)} | {_fmt(row['interactions_all_rows'])} | "
            f"{_fmt(raw_concepts)} | {_fmt(row['interactions_valid_kc_rows'])} | {row['alignment_note']} |"
        )
    lines.append("")
    lines.append("## 3. 为什么当前 ER 生成目录会更小")
    lines.append("")
    lines.append("当前新流程先为 KT/ER 训练过滤掉缺失知识点或不可映射交互，然后按学生划分训练/测试；`stu2know_mastery.json`、`stu2know_seq.json` 等状态文件只覆盖测试学生，用于非冷启动评价。")
    lines.append("")
    lines.append("因此当前 `entities.dict` 往往是“测试学生 + 题目 + 知识点”的训练产物实体集合，不等于论文 Table 2 中的参考 KG 实体总数；当前三元组数量也只反映本次 ER 训练/测试输入，不等于参考论文的全量构图三元组。")
    lines.append("")
    lines.append("## 4. 参考代码交叉检查")
    lines.append("")
    if ref_code:
        lines.append("| Dataset | Q.txt | entities.dict | relations.dict | triples.txt | test_triples.txt |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in ref_code:
            lines.append(
                f"| {row['dataset']} | {_fmt(row.get('Q.txt'))} | {_fmt(row.get('entities.dict'))} | "
                f"{_fmt(row.get('relations.dict'))} | {_fmt(row.get('triples.txt'))} | {_fmt(row.get('test_triples.txt'))} |"
            )
        lines.append("")
        lines.append("本地参考代码的 Algebra 2005 `relations.dict=304`，验证了固定关系集合口径；但其 `entities/triples` 与论文表也不完全一致，说明论文表不是简单地由当前仓库某一个中间文件逐行统计得到。")
    else:
        lines.append("未找到本地参考代码目录，跳过交叉检查。")
    lines.append("")
    lines.append("## 5. 后续处理建议")
    lines.append("")
    lines.append("论文中的数据集概览表按第 1 节对齐；代码运行日志、训练输入规模和资源评估按当前 ER 生成目录统计。当前代码已经支持把关系名裁剪/格式化到 `0.00-1.00` 并输出固定 304 关系全集；重新执行 `after-kt --force` 后，后续新生成的 `relations.dict` 可对齐 304。若要让训练文件也严格变成 Table 2 的 `Entities/Triples` 规模，还需要额外复现参考论文的知识点合并映射、全量实体定义和全量 KG 构图规则，这会改变当前已经跑通的 KT/ER 数据闭环。")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and document Table-2-aligned dataset statistics.")
    parser.add_argument("--output-md", type=Path, default=ER_ROOT / "docs" / "ER数据集Table2对齐统计.md")
    parser.add_argument("--output-json", type=Path, default=ER_ROOT / "docs" / "ER数据集Table2对齐统计.json")
    args = parser.parse_args()

    observed = observed_raw_stats()
    ref_code = reference_code_stats()
    payload = {
        "reference_table2": REFERENCE_TABLE2,
        "observed_raw_stats": observed,
        "reference_code_stats": ref_code,
        "notes": [
            "Table 2 uses raw dataset/reference KG construction statistics.",
            "Current prepared_for_kt directories use filtered KT/ER training artifact statistics.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.output_md, observed, ref_code)
    print(json.dumps({"output_md": str(args.output_md), "output_json": str(args.output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
