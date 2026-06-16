import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create KG triples for exercise recommendation.")
    parser.add_argument("--data-dir", default="Eedi", help="Dataset directory. Relative paths are resolved from this script directory.")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-rec", type=int, default=10)
    parser.add_argument("--mastery-file", default="stu2know_mastery.json")
    parser.add_argument("--seq-file", default="stu2know_seq.json")
    parser.add_argument("--forget-file", default="stu2ex_forget.json")
    parser.add_argument("--recommend-file", default="stu2ex_recommend.json")
    parser.add_argument("--train-triples-file", default="triples.txt")
    parser.add_argument("--test-triples-file", default="test_triples.txt")
    parser.add_argument("--relation-min", type=float, default=0.0)
    parser.add_argument("--relation-max", type=float, default=1.0)
    return parser.parse_args()


def resolve_data_dir(data_dir):
    path = Path(data_dir)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_json(file_path):
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_triples(file_path, triples):
    with file_path.open("w", encoding="utf-8") as f:
        for triple in triples:
            f.write("\t".join(map(str, triple)) + "\n")


def relation_label(prefix, value, min_value=0.0, max_value=1.0):
    value = float(value)
    value = min(max(value, min_value), max_value)
    value = round(value, 2)
    if value == -0.0:
        value = 0.0
    return f"{prefix}{value:.2f}"


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    mlkc_data = load_json(data_dir / args.mastery_file)
    pkc_data = load_json(data_dir / args.seq_file)
    efr_data = load_json(data_dir / args.forget_file)
    rec_data = load_json(data_dir / args.recommend_file)

    student_ids = list(range(len(mlkc_data)))
    knowledge_ids = list(range(len(mlkc_data[0])))
    exercise_ids = list(range(len(efr_data[0])))

    np.random.seed(args.seed)
    np.random.shuffle(student_ids)
    train_size = int(len(student_ids) * args.train_ratio)
    train_students = set(student_ids[:train_size])

    triples_train = []
    triples_test = []

    for uid, (mlkc_row, pkc_row, efr_row) in enumerate(zip(mlkc_data, pkc_data, efr_data)):
        target = triples_train if uid in train_students else triples_test
        for kc, (mlkc, pkc) in enumerate(zip(mlkc_row, pkc_row)):
            target.append((f"kc{kc}", relation_label("mlkc", mlkc, args.relation_min, args.relation_max), f"uid{uid}"))
            target.append((f"kc{kc}", relation_label("pkc", pkc, args.relation_min, args.relation_max), f"uid{uid}"))
        for ex, efr in enumerate(efr_row):
            target.append((f"ex{ex}", relation_label("exfr", efr, args.relation_min, args.relation_max), f"uid{uid}"))

    for uid, rec_list in enumerate(rec_data):
        if uid in train_students:
            sorted_rec = sorted(enumerate(rec_list), key=lambda x: x[1])[: args.top_k_rec]
            for ex, _ in sorted_rec:
                triples_train.append((f"uid{uid}", "rec", f"ex{ex}"))

    save_triples(data_dir / args.train_triples_file, triples_train)
    save_triples(data_dir / args.test_triples_file, triples_test)
    print("student_count:", len(student_ids))
    print("knowledge_count:", len(knowledge_ids))
    print("exercise_count:", len(exercise_ids))
    print("train_triples:", len(triples_train))
    print("test_triples:", len(triples_test))


if __name__ == "__main__":
    main()
