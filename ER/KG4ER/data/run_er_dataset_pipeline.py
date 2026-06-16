import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PUBLIC_DATASETS = {"assist2009", "algebra2005", "statics2011"}
DATASET_FOLDERS = {
    "assist2009": "assist2009",
    "algebra2005": "algebra2005",
    "statics2011": "statics2011",
    "xes3g5m-sub": "XES3G5M-sub",
    "XES3G5M-sub": "XES3G5M-sub",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or document the ER multi-dataset preparation pipeline.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FOLDERS))
    parser.add_argument("--stage", choices=["commands", "prepare", "after-kt", "all"], default="commands")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--split-ratio", type=float, default=0.75)
    parser.add_argument("--top-k-rec", type=int, default=10)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--min-seq-len", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1000000)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate existing lightweight artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def project_root():
    return Path(__file__).resolve().parents[3]


def data_root():
    return Path(__file__).resolve().parent


def dataset_folder(dataset):
    return DATASET_FOLDERS[dataset]


def canonical_dataset_name(dataset):
    return dataset_folder(dataset)


def pykt_dataset_name(dataset):
    return canonical_dataset_name(dataset).lower().replace("-", "_") + "_er_dkt"


def prepared_dir(dataset):
    return data_root() / dataset_folder(dataset) / "prepared_for_kt"


def processed_or_source_dir(dataset):
    folder = dataset_folder(dataset)
    if folder == "XES3G5M-sub":
        return data_root() / folder
    return data_root() / folder / "processed"


def rel_to_root(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(project_root())
    except ValueError:
        return path


def rel_to_data(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(data_root()).as_posix()
    except ValueError:
        return str(path)


def ps_path(path):
    return str(path).replace("/", "\\")


def command_text(command):
    return " ".join(f'"{item}"' if " " in str(item) else str(item) for item in command)


def run_command(command, dry_run=False):
    print(command_text(command))
    if dry_run:
        return
    subprocess.run(command, cwd=project_root(), check=True)


def count_csv_rows(path):
    csv.field_size_limit(1024 * 1024 * 1024)
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return sum(1 for _ in reader)


def q_shape(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(line.split(","))
    return len(rows), len(rows[0]) if rows else 0


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def should_run(marker, force):
    return force or not Path(marker).exists()


def build_public_or_subset(args):
    dataset = canonical_dataset_name(args.dataset)
    source_dir = processed_or_source_dir(args.dataset)
    if dataset in PUBLIC_DATASETS:
        marker = source_dir / "Q.txt"
        if should_run(marker, args.force):
            run_command(
                [
                    sys.executable,
                    str(project_root() / "ER" / "KG4ER" / "data" / "build_public_dataset_sequences.py"),
                    "--dataset",
                    dataset,
                    "--data-root",
                    str(project_root() / "ER" / "KG4ER" / "data"),
                    "--output-dir",
                    str(source_dir),
                    "--split-ratio",
                    str(args.split_ratio),
                    "--seed",
                    str(args.seed),
                ],
                dry_run=args.dry_run,
            )
        return source_dir

    marker = source_dir / "Q.txt"
    if should_run(marker, args.force):
        run_command(
            [
                sys.executable,
                str(project_root() / "ER" / "KG4ER" / "data" / "build_xes3g5m_sub.py"),
                "--output-dir",
                str(source_dir),
                "--seed",
                str(args.seed),
            ],
            dry_run=args.dry_run,
        )
    return source_dir


def prepare_common_inputs(args, source_dir):
    out_dir = prepared_dir(args.dataset)
    marker = out_dir / "Q.txt"
    if should_run(marker, args.force):
        run_command(
            [
                sys.executable,
                str(project_root() / "ER" / "KG4ER" / "data" / "prepare_kt_training_files.py"),
                "--dataset-name",
                canonical_dataset_name(args.dataset),
                "--source-dir",
                str(source_dir),
                "--output-dir",
                str(out_dir),
                "--min-seq-len",
                str(args.min_seq_len),
                "--max-seq-len",
                str(args.max_seq_len),
            ],
            dry_run=args.dry_run,
        )
    return out_dir


def generate_forget_and_response(args):
    out_dir = prepared_dir(args.dataset)
    response_file = f"{canonical_dataset_name(args.dataset)}_uid_kc_response.txt"
    if should_run(out_dir / "stu2ex_forget.json", args.force):
        run_command(
            [
                sys.executable,
                str(project_root() / "ER" / "pykt-toolkit-main" / "data" / "Eedi" / "calculate_time.py"),
                "--data-dir",
                str(out_dir),
                "--test-sequences-file",
                "test_sequences.csv",
                "--q-file",
                "Q.txt",
                "--know-output-file",
                "stu2know_forget.json",
                "--exercise-output-file",
                "stu2ex_forget.json",
                "--min-seq-len",
                str(args.min_seq_len),
                "--max-seq-len",
                str(args.max_seq_len),
            ],
            dry_run=args.dry_run,
        )
    if should_run(out_dir / response_file, args.force):
        run_command(
            [
                sys.executable,
                str(project_root() / "ER" / "KG4ER" / "data" / "step3_get_kc_response.py"),
                "--test-sequences",
                str(out_dir / "test_sequences.csv"),
                "--data-dir",
                str(out_dir),
                "--output-file",
                response_file,
                "--min-seq-len",
                str(args.min_seq_len),
                "--max-seq-len",
                str(args.max_seq_len),
            ],
            dry_run=args.dry_run,
        )


def prepare_and_register_dkt(args):
    out_dir = prepared_dir(args.dataset)
    dkt_dir = out_dir / "dkt_concept"
    if should_run(dkt_dir / "dkt_concept_manifest.json", args.force):
        run_command(
            [
                sys.executable,
                str(project_root() / "ER" / "KG4ER" / "data" / "prepare_dkt_concept_sequences.py"),
                "--data-dir",
                str(out_dir),
                "--maxlen",
                str(args.maxlen),
                "--fold-count",
                str(args.fold_count),
            ],
            dry_run=args.dry_run,
        )
    run_command(
        [
            sys.executable,
            str(project_root() / "ER" / "KG4ER" / "data" / "register_pykt_dkt_dataset.py"),
            "--data-dir",
            str(out_dir),
            "--pykt-root",
            str(project_root() / "ER" / "pykt-toolkit-main"),
            "--dataset-name",
            pykt_dataset_name(args.dataset),
            "--maxlen",
            str(args.maxlen),
            "--fold-count",
            str(args.fold_count),
        ],
        dry_run=args.dry_run,
    )


def required_kt_files_ready(dataset):
    out_dir = prepared_dir(dataset)
    required = [
        out_dir / "stu2know_mastery.json",
        out_dir / "stu2know_seq.json",
        out_dir / "stu2know_forget.json",
        out_dir / "stu2ex_forget.json",
        out_dir / "Q.txt",
    ]
    return [path for path in required if not path.exists()]


def run_after_kt(args):
    out_dir = prepared_dir(args.dataset)
    missing = required_kt_files_ready(args.dataset)
    if missing:
        print("KT exports are not ready; skip after-kt stage. Missing:")
        for path in missing:
            print(f"- {path}")
        return False

    data_dir_arg = rel_to_data(out_dir)
    run_command(
        [
            sys.executable,
            str(project_root() / "ER" / "KG4ER" / "data" / "validate_kt_exports.py"),
            "--data-dir",
            str(out_dir),
        ],
        dry_run=args.dry_run,
    )
    if should_run(out_dir / "stu2ex_recommend.json", args.force):
        run_command(
            [
                sys.executable,
                str(project_root() / "ER" / "KG4ER" / "data" / "step1_cal_recommend.py"),
                "--data-dir",
                data_dir_arg,
                "--output-file",
                "stu2ex_recommend.json",
            ],
            dry_run=args.dry_run,
        )
    if should_run(out_dir / "triples.txt", args.force) or should_run(out_dir / "test_triples.txt", args.force):
        run_command(
            [
                sys.executable,
                str(project_root() / "ER" / "KG4ER" / "data" / "step2_createtriples.py"),
                "--data-dir",
                data_dir_arg,
                "--train-ratio",
                "0.75",
                "--seed",
                str(args.seed),
                "--top-k-rec",
                str(args.top_k_rec),
                "--relation-min",
                "0.0",
                "--relation-max",
                "1.0",
            ],
            dry_run=args.dry_run,
        )
    run_command(
        [
            sys.executable,
            str(project_root() / "ER" / "KG4ER" / "data" / "preprocess.py"),
            "--data-dir",
            data_dir_arg,
            "--output-file",
            "entities.dict",
        ],
        dry_run=args.dry_run,
    )
    run_command(
        [
            sys.executable,
            str(project_root() / "ER" / "KG4ER" / "data" / "create_relations_dict.py"),
            "--data-dir",
            data_dir_arg,
            "--train-triples-file",
            "triples.txt",
            "--test-triples-file",
            "test_triples.txt",
            "--output-file",
            "relations.dict",
            "--fixed-kg4er-relations",
        ],
        dry_run=args.dry_run,
    )
    return True


def dkt_checkpoint_dir_name(args):
    name = pykt_dataset_name(args.dataset)
    return f"{name}_dkt_qid_saved_model_{args.seed}_0_0.2_200_0.001_20_32_0_0"


def dataset_stats(dataset):
    out_dir = prepared_dir(dataset)
    stats = load_json(out_dir / "processed_stats.json", default={}) or load_json(out_dir / "subset_stats.json", default={}) or {}
    questions, concepts = q_shape(out_dir / "Q.txt") if (out_dir / "Q.txt").exists() else (None, None)
    test_students = count_csv_rows(out_dir / "test_sequences.csv") if (out_dir / "test_sequences.csv").exists() else None
    return {
        "dataset": canonical_dataset_name(dataset),
        "prepared_dir": str(rel_to_root(out_dir)),
        "students": stats.get("students"),
        "questions": questions or stats.get("questions"),
        "concepts": concepts or stats.get("concepts"),
        "interactions": stats.get("interactions"),
        "test_students": test_students or stats.get("test_students"),
    }


def write_pipeline_commands(args):
    dataset = canonical_dataset_name(args.dataset)
    out_dir = prepared_dir(args.dataset)
    pykt_name = pykt_dataset_name(args.dataset)
    stats = dataset_stats(args.dataset)
    q_count = stats["questions"] or "<QUESTION_COUNT>"
    c_count = stats["concepts"] or "<CONCEPT_COUNT>"
    test_students = stats["test_students"] or "<TEST_STUDENT_COUNT>"
    ckpt_dir = dkt_checkpoint_dir_name(args)
    run_root = args.run_root or "..\\runs"

    text = f"""# {dataset} ER 全流程命令

本文档由 `ER/KG4ER/data/run_er_dataset_pipeline.py` 生成。`stu2know_mastery.json`、`stu2know_seq.json` 和 ER 模型训练不在自动执行范围内，命令列在下面，由实验者手动运行。

## 1. 当前数据规模

| 项目 | 数值 |
| --- | ---: |
| prepared 目录 | `{ps_path(rel_to_root(out_dir))}` |
| 学生数 | {stats["students"] if stats["students"] is not None else "待统计"} |
| 测试学生数 | {test_students} |
| 题目数 | {q_count} |
| 知识点数 | {c_count} |
| 交互数 | {stats["interactions"] if stats["interactions"] is not None else "待统计"} |

## 2. 一键准备到 KT 导出前

这一步会生成/检查 `prepared_for_kt`、遗忘率文件、KC response、DKT 输入文件，并注册 pyKT 数据集。

```powershell
$ROOT = "{project_root()}"
cd $ROOT
python ER\\KG4ER\\data\\run_er_dataset_pipeline.py --dataset {dataset} --stage prepare --seed {args.seed}
```

## 3. 生成 mastery

```powershell
cd $ROOT
python ER\\KG4ER\\data\\export_mastery_from_bkt.py `
  --data-dir ER\\KG4ER\\data\\{dataset}\\prepared_for_kt `
  --output-file stu2know_mastery.json `
  --manifest-file bkt_mastery_manifest.json `
  --epochs 15 `
  --min-observations 20
```

输出：
- `ER/KG4ER/data/{dataset}/prepared_for_kt/stu2know_mastery.json`
- `ER/KG4ER/data/{dataset}/prepared_for_kt/bkt_mastery_manifest.json`

## 4. 训练 DKT 并导出 seq

```powershell
cd $ROOT\\ER\\pykt-toolkit-main\\examples
python wandb_dkt_train.py `
  --dataset_name {pykt_name} `
  --model_name dkt `
  --emb_type qid `
  --save_dir saved_model `
  --seed {args.seed} `
  --fold 0 `
  --dropout 0.2 `
  --emb_size 200 `
  --learning_rate 0.001 `
  --num_epochs 20 `
  --batch_size 32 `
  --use_wandb 0 `
  --add_uuid 0
```

训练结束后导出 `stu2know_seq.json`：

```powershell
cd $ROOT
$CKPT = "ER/pykt-toolkit-main/examples/saved_model/{ckpt_dir}"
python ER\\KG4ER\\data\\export_seq_from_dkt.py `
  --pykt-root ER\\pykt-toolkit-main `
  --checkpoint-dir $CKPT `
  --checkpoint-file qid_model.ckpt `
  --test-sequences ER\\KG4ER\\data\\{dataset}\\prepared_for_kt\\dkt_concept\\test_sequences_full.csv `
  --output-file ER\\KG4ER\\data\\{dataset}\\prepared_for_kt\\stu2know_seq.json `
  --expected-students {test_students} `
  --expected-concepts {c_count} `
  --device cpu
```

## 5. KT 文件齐全后生成 ER 三元组和字典

```powershell
cd $ROOT
python ER\\KG4ER\\data\\run_er_dataset_pipeline.py --dataset {dataset} --stage after-kt --seed {args.seed} --top-k-rec {args.top_k_rec}
```

该命令会依次生成/更新：
- `stu2ex_recommend.json`
- `triples.txt`
- `test_triples.txt`
- `entities.dict`
- `relations.dict`

## 6. ConvE 主模型训练和测试

```powershell
cd $ROOT\\ER\\KG4ER\\codes
python run_ConvE.py `
  --data_path ..\\data\\{dataset}\\prepared_for_kt `
  --dataset_name {dataset} `
  --run_root {run_root} `
  --epochs 20 `
  --bs 1024 `
  --learning_rate 0.001 `
  --cuda false `
  --seed {args.seed}
```

如果使用 `--run_root`，训练目录会自动生成在 `ER/KG4ER/runs/{dataset}/...`。测试时把 `$CONVE_RUN` 改成实际训练目录：

```powershell
$CONVE_RUN = "..\\runs\\{dataset}\\<实际ConvE运行目录>"
python test_ConvE.py `
  --dataset {dataset} `
  --model-type ConvE `
  --data-path ..\\data\\{dataset}\\prepared_for_kt `
  --embedding-path $CONVE_RUN `
  --explain-top-k 3 `
  --explain-user-count 2
```

## 7. KGE baseline 训练命令

TransE：

```powershell
python run.py --do_train --data_path ..\\data\\{dataset}\\prepared_for_kt --dataset_name {dataset} --model TransE --run_root {run_root} --max_steps 10000 --batch_size 1024 --negative_sample_size 256 --learning_rate 0.0001 --seed {args.seed}
```

RotatE：

```powershell
python run.py --do_train --data_path ..\\data\\{dataset}\\prepared_for_kt --dataset_name {dataset} --model RotatE --double_entity_embedding --run_root {run_root} --max_steps 10000 --batch_size 1024 --negative_sample_size 256 --learning_rate 0.0001 --seed {args.seed}
```

DistMult：

```powershell
python run.py --do_train --data_path ..\\data\\{dataset}\\prepared_for_kt --dataset_name {dataset} --model DistMult --run_root {run_root} --max_steps 10000 --batch_size 1024 --negative_sample_size 256 --learning_rate 0.0001 --seed {args.seed}
```

ComplEx：

```powershell
python run.py --do_train --data_path ..\\data\\{dataset}\\prepared_for_kt --dataset_name {dataset} --model ComplEx --double_entity_embedding --double_relation_embedding --run_root {run_root} --max_steps 10000 --batch_size 1024 --negative_sample_size 256 --learning_rate 0.0001 --seed {args.seed}
```

当前 `test_TransE.py` 可用于 TransE 类输出的 ACC/NOV 评价；RotatE、DistMult、ComplEx 的推荐评价需要使用对应模型的打分函数接入统一评估脚本后再正式出表。

## 8. 五随机种子重复实验

示例：对 ConvE 使用 5 个随机种子重复训练。`--command-template` 里的 `{{seed}}` 和 `{{run_dir}}` 会由脚本替换。

```powershell
cd $ROOT\\ER\\KG4ER\\codes
python run_repeated_experiments.py `
  --dataset {dataset} `
  --model ConvE `
  --run-root ..\\runs `
  --seeds 2024,2025,2026,2027,2028 `
  --command-template "python run_ConvE.py --data_path ..\\data\\{dataset}\\prepared_for_kt --dataset_name {dataset} --save_path {{run_dir}} --epochs 20 --bs 1024 --learning_rate 0.001 --cuda false --seed {{seed}}"
```
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    command_path = out_dir / "ER_pipeline_commands.md"
    command_path.write_text(text, encoding="utf-8")
    return command_path


def write_pipeline_status(args, after_kt_completed=False):
    out_dir = prepared_dir(args.dataset)
    stats = dataset_stats(args.dataset)
    files = {
        "Q.txt": (out_dir / "Q.txt").exists(),
        "stu2know_forget.json": (out_dir / "stu2know_forget.json").exists(),
        "stu2ex_forget.json": (out_dir / "stu2ex_forget.json").exists(),
        f"{canonical_dataset_name(args.dataset)}_uid_kc_response.txt": (out_dir / f"{canonical_dataset_name(args.dataset)}_uid_kc_response.txt").exists(),
        "dkt_concept/test_sequences_full.csv": (out_dir / "dkt_concept" / "test_sequences_full.csv").exists(),
        "stu2know_mastery.json": (out_dir / "stu2know_mastery.json").exists(),
        "stu2know_seq.json": (out_dir / "stu2know_seq.json").exists(),
        "stu2ex_recommend.json": (out_dir / "stu2ex_recommend.json").exists(),
        "triples.txt": (out_dir / "triples.txt").exists(),
        "test_triples.txt": (out_dir / "test_triples.txt").exists(),
        "entities.dict": (out_dir / "entities.dict").exists(),
        "relations.dict": (out_dir / "relations.dict").exists(),
    }
    after_kt_outputs = [
        "stu2know_mastery.json",
        "stu2know_seq.json",
        "stu2ex_recommend.json",
        "triples.txt",
        "test_triples.txt",
        "entities.dict",
        "relations.dict",
    ]
    effective_after_kt_completed = after_kt_completed or all(files[name] for name in after_kt_outputs)
    write_json(
        out_dir / "er_pipeline_status.json",
        {
            "dataset": canonical_dataset_name(args.dataset),
            "stage": args.stage,
            "stats": stats,
            "after_kt_completed": effective_after_kt_completed,
            "files": files,
            "commands_doc": str(rel_to_root(out_dir / "ER_pipeline_commands.md")),
        },
    )


def main():
    args = parse_args()
    source_dir = processed_or_source_dir(args.dataset)
    if args.stage in {"prepare", "all"}:
        source_dir = build_public_or_subset(args)
        prepare_common_inputs(args, source_dir)
        generate_forget_and_response(args)
        prepare_and_register_dkt(args)

    after_kt_completed = False
    if args.stage in {"after-kt", "all"}:
        after_kt_completed = run_after_kt(args)

    command_path = write_pipeline_commands(args)
    write_pipeline_status(args, after_kt_completed=after_kt_completed)
    print(f"commands -> {command_path}")
    print(f"status -> {prepared_dir(args.dataset) / 'er_pipeline_status.json'}")


if __name__ == "__main__":
    main()
