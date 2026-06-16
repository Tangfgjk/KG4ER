import argparse
import csv
from pathlib import Path


def parse_args():
    er_root = Path(__file__).resolve().parents[2]
    default_pykt_dir = er_root / "pykt-toolkit-main" / "data" / "Eedi"
    default_output = Path(__file__).resolve().parent / "Eedi" / "sequence_interactions.csv"
    parser = argparse.ArgumentParser(description="Build Eedi sequence_interactions.csv for ER CF baselines.")
    parser.add_argument("--pykt-eedi-dir", type=Path, default=default_pykt_dir)
    parser.add_argument("--output-file", type=Path, default=default_output)
    parser.add_argument("--min-seq-len", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=198)
    parser.add_argument("--include-train", action="store_true", help="Include train_valid rows for EB-CF popularity.")
    return parser.parse_args()


def parse_tokens(value):
    return [token for token in str(value).split(",") if token != ""]


def parse_int_tokens(value):
    values = []
    for token in parse_tokens(value):
        try:
            values.append(int(token))
        except ValueError:
            values.append(-1)
    return values


def parse_concept_steps(value):
    concept_steps = []
    for token in parse_tokens(value):
        if token == "-1":
            continue
        concepts = [int(item) for item in token.split("_") if item and item != "-1"]
        if concepts:
            concept_steps.append(concepts)
    return concept_steps


def valid_sequence_row(row, min_seq_len, max_seq_len):
    # Historical Eedi ER files were filtered from the 3-line sequence format,
    # where sequence length means valid question count rather than concept steps.
    valid_questions = [question for question in parse_int_tokens(row["questions"]) if question >= 0]
    return min_seq_len <= len(valid_questions) <= max_seq_len


def iter_interactions(row, uid, source_split):
    questions = parse_int_tokens(row["questions"])
    concepts = parse_tokens(row["concepts"])
    responses = parse_int_tokens(row["responses"])
    timestamps = parse_tokens(row.get("timestamps", ""))
    selectmasks = parse_int_tokens(row.get("selectmasks", ""))
    if not selectmasks:
        selectmasks = [1] * len(questions)

    for index, (question, response, selectmask) in enumerate(zip(questions, responses, selectmasks)):
        if question < 0 or response not in (0, 1) or selectmask != 1:
            continue
        yield {
            "uid": uid,
            "original_uid": row.get("uid", ""),
            "question": question,
            "original_question": question,
            "concepts": concepts[index] if index < len(concepts) else "",
            "response": response,
            "timestamp": timestamps[index] if index < len(timestamps) else index,
            "source_split": source_split,
        }


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as fp:
        yield from csv.DictReader(fp)


def iter_three_line_test_interactions(path, min_seq_len, max_seq_len):
    lines = path.read_text(encoding="utf-8").splitlines()
    uid = 0
    for offset in range(0, len(lines), 3):
        if offset + 2 >= len(lines):
            break
        seq_len = int(lines[offset].strip())
        if not (min_seq_len <= seq_len <= max_seq_len):
            continue
        questions = parse_int_tokens(lines[offset + 1])
        responses = parse_int_tokens(lines[offset + 2])
        for index, (question, response) in enumerate(zip(questions, responses)):
            if question < 0 or response not in (0, 1):
                continue
            yield {
                "uid": uid,
                "original_uid": "",
                "question": question,
                "original_question": question,
                "concepts": "",
                "response": response,
                "timestamp": index,
                "source_split": "test",
            }
        uid += 1
    return uid


def main():
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.include_train:
        for row in read_rows(args.pykt_eedi_dir / "train_valid.csv"):
            raw_uid = str(row.get("uid", "")).strip()
            train_uid = f"train_{raw_uid}"
            rows.extend(iter_interactions(row, train_uid, "train_valid"))

    test_rows = list(
        iter_three_line_test_interactions(
            args.pykt_eedi_dir / "new_test_sequence.csv",
            args.min_seq_len,
            args.max_seq_len,
        )
    )
    rows.extend(test_rows)
    test_uid = len({row["uid"] for row in test_rows})

    fieldnames = ["uid", "original_uid", "question", "original_question", "concepts", "response", "timestamp", "source_split"]
    with args.output_file.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        {
            "output_file": str(args.output_file),
            "rows": len(rows),
            "test_students": test_uid,
            "include_train": args.include_train,
        }
    )


if __name__ == "__main__":
    main()
