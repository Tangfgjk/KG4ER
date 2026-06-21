import argparse
import json
import sys
from pathlib import Path


CODES_DIR = Path(__file__).resolve().parents[1] / "codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from conve_ablation_data import (
    VALID_TERMS,
    calculate_recommendation_scores,
    parse_active_terms,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Calculate student-exercise recommendation distances.")
    parser.add_argument("--data-dir", default="Eedi", help="Dataset directory. Relative paths are resolved from this script directory.")
    parser.add_argument("--mastery-file", default="stu2know_mastery.json")
    parser.add_argument("--seq-file", default="stu2know_seq.json")
    parser.add_argument("--forget-file", default="stu2ex_forget.json")
    parser.add_argument("--q-file", default="Q.txt")
    parser.add_argument("--output-file", default="stu2ex_recommend.json")
    parser.add_argument("--delta-1", type=float, default=0.8)
    parser.add_argument("--delta-2", type=float, default=0.8)
    parser.add_argument(
        "--terms",
        default=",".join(VALID_TERMS),
        help="Comma-separated distance terms: mastery,sequence,forgetting.",
    )
    parser.add_argument(
        "--use-seq-term",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


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

    active_terms = parse_active_terms(args.terms)
    print("student_count:", len(stu2know_mastery))
    print("active_terms:", ",".join(active_terms))
    all_stu2ex_recommend = calculate_recommendation_scores(
        stu2know_mastery,
        stu2know_seq,
        stu2ex_forget,
        q_matrix,
        active_terms=active_terms,
        delta_1=args.delta_1,
        delta_2=args.delta_2,
    )

    with (data_dir / args.output_file).open("w", encoding="utf-8") as f:
        json.dump(all_stu2ex_recommend, f)


if __name__ == "__main__":
    main()
