import argparse
import json
import shutil
from pathlib import Path


TRAIN_CANDIDATES = [
    "train_sequences.csv",
    "train_valid_sequences.csv",
    "train_valid_sequences_quelevel.csv",
]
TEST_CANDIDATES = [
    "test_sequences.csv",
    "test_quelevel.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset files for ER-side EKT/DKT state export.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-seq-len", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1000000)
    return parser.parse_args()


def find_first_existing(base_dir, candidates):
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No candidate found in {base_dir}: {', '.join(candidates)}")


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_commands(path, dataset_name, output_dir, min_seq_len, max_seq_len):
    calculate_time = Path("ER/pykt-toolkit-main/data/Eedi/calculate_time.py")
    kc_response = Path("ER/KG4ER/data/step3_get_kc_response.py")
    response_file = f"{dataset_name}_uid_kc_response.txt"
    text = f"""# {dataset_name} KT/ER 文件生成命令

本目录已经整理出 EKT/DKT 训练和 ER 后续步骤需要的统一输入文件：
- `Q.txt`
- `train_sequences.csv`
- `test_sequences.csv`
- `train_valid_sequences.csv`
- `train_valid_sequences_quelevel.csv`
- `test_quelevel.csv`

## 可直接生成的文件

```powershell
python {calculate_time} --data-dir "{output_dir}" --test-sequences-file test_sequences.csv --q-file Q.txt --min-seq-len {min_seq_len} --max-seq-len {max_seq_len}
python {kc_response} --test-sequences "{output_dir / 'test_sequences.csv'}" --data-dir "{output_dir}" --output-file "{response_file}" --min-seq-len {min_seq_len} --max-seq-len {max_seq_len}
```

上述命令生成：
- `stu2know_forget.json`
- `stu2ex_forget.json`
- `{dataset_name}_uid_kc_response.txt`

## 必须由 KT 模型导出的文件

- `stu2know_mastery.json`：由 BKT/EKT/MMKT 等知识追踪模型在测试学生序列上导出，表示测试学生最终知识掌握状态。
- `stu2know_seq.json`：由 pyKT DKT/序列模型在测试学生序列上导出，表示知识点序列/progress 状态。

这两个文件生成后，再运行 `step1_cal_recommend.py` 和 `step2_createtriples.py` 构造推荐边与训练/测试三元组。
"""
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q_file = source_dir / "Q.txt"
    if not q_file.exists():
        raise FileNotFoundError(q_file)

    train_file = find_first_existing(source_dir, TRAIN_CANDIDATES)
    test_file = find_first_existing(source_dir, TEST_CANDIDATES)

    copy_file(q_file, output_dir / "Q.txt")
    copy_file(train_file, output_dir / "train_sequences.csv")
    copy_file(test_file, output_dir / "test_sequences.csv")
    copy_file(train_file, output_dir / "train_valid_sequences.csv")
    copy_file(train_file, output_dir / "train_valid_sequences_quelevel.csv")
    copy_file(test_file, output_dir / "test_quelevel.csv")

    for optional_name in ["id_maps.json", "processed_stats.json", "subset_stats.json", "sequence_interactions.csv", "interactions.csv"]:
        optional_path = source_dir / optional_name
        if optional_path.exists():
            copy_file(optional_path, output_dir / optional_name)

    manifest = {
        "dataset_name": args.dataset_name,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "q_file": str(output_dir / "Q.txt"),
        "train_sequences": str(output_dir / "train_sequences.csv"),
        "test_sequences": str(output_dir / "test_sequences.csv"),
        "min_seq_len": args.min_seq_len,
        "max_seq_len": args.max_seq_len,
        "generated_without_kt_training": [
            "stu2know_forget.json",
            "stu2ex_forget.json",
            f"{args.dataset_name}_uid_kc_response.txt",
        ],
        "required_kt_exports": {
            "stu2know_mastery.json": "KT model test export: final knowledge mastery state for each test student.",
            "stu2know_seq.json": "pyKT DKT/sequence model test export: next-step knowledge prediction state.",
        },
        "next_er_steps_after_kt_exports": [
            "step1_cal_recommend.py",
            "step2_createtriples.py",
            "KG4ER model training/evaluation",
        ],
    }
    write_json(output_dir / "kt_export_manifest.json", manifest)
    write_commands(output_dir / "commands.md", args.dataset_name, output_dir, args.min_seq_len, args.max_seq_len)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
