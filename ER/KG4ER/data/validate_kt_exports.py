import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Validate KT state exports before ER recommendation/triple generation.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--q-file", default="Q.txt")
    parser.add_argument("--test-file", default="test_sequences.csv")
    parser.add_argument("--mastery-file", default="stu2know_mastery.json")
    parser.add_argument("--seq-file", default="stu2know_seq.json")
    parser.add_argument("--forget-file", default="stu2know_forget.json")
    parser.add_argument("--exercise-forget-file", default="stu2ex_forget.json")
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def matrix_shape(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if not value:
        return 0, 0
    width = len(value[0])
    for idx, row in enumerate(value):
        if len(row) != width:
            raise ValueError(f"{name} row {idx} has width {len(row)}, expected {width}")
    return len(value), width


def q_shape(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([int(x) for x in line.split(",")])
    if not rows:
        return 0, 0
    width = len(rows[0])
    for idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"Q row {idx} has width {len(row)}, expected {width}")
    return len(rows), width


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    required_files = [
        data_dir / args.q_file,
        data_dir / args.test_file,
        data_dir / args.mastery_file,
        data_dir / args.seq_file,
        data_dir / args.forget_file,
        data_dir / args.exercise_forget_file,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        summary = {
            "data_dir": str(data_dir),
            "status": "missing_files",
            "missing_files": missing_files,
        }
        raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))

    question_count, concept_count = q_shape(data_dir / args.q_file)
    test_students = len(pd.read_csv(data_dir / args.test_file))

    checks = {
        "mastery": matrix_shape(load_json(data_dir / args.mastery_file), args.mastery_file),
        "seq": matrix_shape(load_json(data_dir / args.seq_file), args.seq_file),
        "forget": matrix_shape(load_json(data_dir / args.forget_file), args.forget_file),
        "exercise_forget": matrix_shape(load_json(data_dir / args.exercise_forget_file), args.exercise_forget_file),
    }

    expected = {
        "mastery": (test_students, concept_count),
        "seq": (test_students, concept_count),
        "forget": (test_students, concept_count),
        "exercise_forget": (test_students, question_count),
    }

    for key, shape in checks.items():
        if shape != expected[key]:
            raise ValueError(f"{key} shape {shape} != expected {expected[key]}")

    summary = {
        "data_dir": str(data_dir),
        "test_students": test_students,
        "question_count": question_count,
        "concept_count": concept_count,
        "checks": {key: list(value) for key, value in checks.items()},
        "status": "passed",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
