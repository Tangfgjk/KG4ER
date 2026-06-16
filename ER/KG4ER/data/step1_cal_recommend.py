import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate student-exercise recommendation distances.")
    parser.add_argument("--data-dir", default="Eedi", help="Dataset directory. Relative paths are resolved from this script directory.")
    parser.add_argument("--mastery-file", default="stu2know_mastery.json")
    parser.add_argument("--seq-file", default="stu2know_seq.json")
    parser.add_argument("--forget-file", default="stu2ex_forget.json")
    parser.add_argument("--q-file", default="Q.txt")
    parser.add_argument("--output-file", default="stu2ex_recommend.json")
    parser.add_argument("--delta-1", type=float, default=0.8)
    parser.add_argument("--delta-2", type=float, default=0.8)
    parser.add_argument("--use-seq-term", action="store_true", help="Include the stu2know_seq/Q cosine term in W_ij.")
    return parser.parse_args()


def resolve_data_dir(data_dir):
    path = Path(data_dir)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_json(path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_q_matrix(path):
    with path.open("r", encoding="utf-8") as f:
        return [[int(x) for x in line.strip().split(",")] for line in f if line.strip()]


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    stu2know_mastery = load_json(data_dir / args.mastery_file)
    stu2know_seq = load_json(data_dir / args.seq_file)
    stu2ex_forget = load_json(data_dir / args.forget_file)
    q_matrix = load_q_matrix(data_dir / args.q_file)

    all_stu2ex_recommend = []
    print("student_count:", len(stu2know_mastery))

    for stu_idx in range(len(stu2know_mastery)):
        student_mastery = stu2know_mastery[stu_idx]
        student_seq = stu2know_seq[stu_idx]
        student_forget = stu2ex_forget[stu_idx]
        pkc_vector = np.array(student_seq)

        stu_recommend = []
        for ex_idx in range(len(student_forget)):
            mlkc_term = 1.0
            for k in range(len(student_mastery)):
                if q_matrix[ex_idx][k] == 1:
                    mlkc_term *= student_mastery[k]
            term1 = (args.delta_1 - mlkc_term) ** 2

            q_row = np.array(q_matrix[ex_idx])
            cos_sim = np.dot(q_row, pkc_vector.T) / (np.linalg.norm(q_row) * np.linalg.norm(pkc_vector) + 1e-9)
            term2 = cos_sim ** 2

            forget_term = student_forget[ex_idx]
            term3 = (args.delta_2 - forget_term) ** 2

            total = term1 + term3
            if args.use_seq_term:
                total += term2
            stu_recommend.append(round(float(np.sqrt(total)), 2))

        all_stu2ex_recommend.append(stu_recommend)

    with (data_dir / args.output_file).open("w", encoding="utf-8") as f:
        json.dump(all_stu2ex_recommend, f)


if __name__ == "__main__":
    main()
