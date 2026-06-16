import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Collect each student's correctly answered knowledge points.")
    er_dir = Path(__file__).resolve().parents[2]
    default_sequences = er_dir / "pykt-toolkit-main" / "data" / "Eedi" / "test_sequences.csv"
    parser.add_argument("--test-sequences", default=str(default_sequences))
    parser.add_argument("--data-dir", default="Eedi", help="KG4ER dataset directory. Relative paths are resolved from this script directory.")
    parser.add_argument("--output-file", default=None, help="Defaults to <dataset>_uid_kc_response.txt.")
    parser.add_argument("--min-seq-len", type=int, default=10, help="Keep students with at least this many valid interactions.")
    parser.add_argument("--max-seq-len", type=int, default=198, help="Keep students with at most this many valid interactions.")
    return parser.parse_args()


def resolve_data_dir(data_dir):
    path = Path(data_dir)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def parse_int_tokens(value):
    return [int(i) for i in str(value).split(",") if i and i != "-1"]


def parse_concept_steps(value):
    concept_steps = []
    for token in str(value).split(","):
        if not token or token == "-1":
            continue
        concept_steps.append([int(i) for i in token.split("_") if i and i != "-1"])
    return concept_steps


def main():
    args = parse_args()
    test_sequences = Path(args.test_sequences)
    data_dir = resolve_data_dir(args.data_dir)
    output_file = args.output_file or f"{data_dir.name}_uid_kc_response.txt"

    df = pd.read_csv(test_sequences)
    output_lines = []

    uid = 0
    for _, row in df.iterrows():
        concept_steps = parse_concept_steps(row["concepts"])
        responses = parse_int_tokens(row["responses"])

        if len(concept_steps) < args.min_seq_len or len(concept_steps) > args.max_seq_len:
            continue

        correct_kps = []
        for concepts, response in zip(concept_steps, responses):
            if response != 1:
                continue
            for concept in concepts:
                if concept not in correct_kps:
                    correct_kps.append(concept)

        output_lines.append(f"uid{uid}\t{','.join(map(str, correct_kps))}")
        uid += 1

    print(len(output_lines))
    with (data_dir / output_file).open("w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
