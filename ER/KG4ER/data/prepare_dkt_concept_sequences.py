import argparse
import csv
import json
import sys
from pathlib import Path

field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_limit)
        break
    except OverflowError:
        field_limit = int(field_limit / 10)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare single-concept pyKT DKT sequences from ER prepared data.")
    parser.add_argument("--data-dir", required=True, help="Directory containing prepared_for_kt sequence files.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <data-dir>/dkt_concept.")
    parser.add_argument("--train-file", default="train_sequences.csv")
    parser.add_argument("--test-file", default="test_sequences.csv")
    parser.add_argument("--q-file", default="Q.txt")
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--fold-count", type=int, default=5)
    return parser.parse_args()


def parse_list(value):
    if value is None:
        return []
    return [token for token in str(value).split(",") if token != ""]


def expand_row(row):
    questions = parse_list(row["questions"])
    concepts = parse_list(row["concepts"])
    responses = parse_list(row["responses"])
    selectmasks = parse_list(row["selectmasks"]) if "selectmasks" in row else ["1"] * len(questions)

    expanded = []
    usable = min(len(questions), len(concepts), len(responses), len(selectmasks))
    for idx in range(usable):
        if questions[idx] == "-1" or concepts[idx] == "-1" or responses[idx] == "-1":
            continue
        for concept in concepts[idx].split("_"):
            if concept == "" or concept == "-1":
                continue
            expanded.append(
                {
                    "question": int(questions[idx]),
                    "concept": int(concept),
                    "response": int(responses[idx]),
                    "selectmask": int(selectmasks[idx]),
                }
            )
    return expanded


def pad(values, maxlen, pad_value="-1"):
    return values + [pad_value] * (maxlen - len(values))


def make_chunk_row(uid, fold, chunk, maxlen):
    length = len(chunk)
    questions = pad([str(x["question"]) for x in chunk], maxlen)
    concepts = pad([str(x["concept"]) for x in chunk], maxlen)
    responses = pad([str(x["response"]) for x in chunk], maxlen)
    selectmasks = pad([str(x["selectmask"]) for x in chunk], maxlen)
    return {
        "fold": fold,
        "uid": uid,
        "questions": ",".join(questions),
        "concepts": ",".join(concepts),
        "responses": ",".join(responses),
        "selectmasks": ",".join(selectmasks),
        "orig_len": length,
    }


def make_full_row(row, expanded):
    return {
        "fold": row.get("fold", -1),
        "uid": row.get("uid", len(expanded)),
        "questions": ",".join(str(x["question"]) for x in expanded),
        "concepts": ",".join(str(x["concept"]) for x in expanded),
        "responses": ",".join(str(x["response"]) for x in expanded),
        "selectmasks": ",".join(str(x["selectmask"]) for x in expanded),
        "orig_len": len(expanded),
    }


FIELDNAMES = ["fold", "uid", "questions", "concepts", "responses", "selectmasks", "orig_len"]


def convert_file(input_path, chunked_path, full_path, maxlen, fold_count, is_test=False):
    chunk_index = 0
    kept = 0
    interactions = 0
    source_students = 0
    with chunked_path.open("w", encoding="utf-8", newline="") as chunk_f, full_path.open("w", encoding="utf-8", newline="") as full_f:
        chunk_writer = csv.DictWriter(chunk_f, fieldnames=FIELDNAMES)
        full_writer = csv.DictWriter(full_f, fieldnames=FIELDNAMES)
        chunk_writer.writeheader()
        full_writer.writeheader()
        with input_path.open("r", encoding="utf-8", newline="") as input_f:
            reader = csv.DictReader(input_f)
            for row_index, row in enumerate(reader):
                source_students += 1
                expanded = expand_row(row)
                if not expanded:
                    continue
                full_writer.writerow(make_full_row(row, expanded))
                kept += 1
                interactions += len(expanded)
                for start in range(0, len(expanded), maxlen):
                    chunk = expanded[start : start + maxlen]
                    fold = -1 if is_test else chunk_index % fold_count
                    row_uid = row.get("uid", row_index)
                    uid = f"{row_uid}_{start // maxlen}" if len(expanded) > maxlen else row_uid
                    chunk_writer.writerow(make_chunk_row(uid, fold, chunk, maxlen))
                    chunk_index += 1

    return {
        "input": str(input_path),
        "chunked": str(chunked_path),
        "full": str(full_path),
        "source_students": int(source_students),
        "full_students": int(kept),
        "chunks": int(chunk_index),
        "interactions": int(interactions),
    }


def count_q(path):
    with path.open("r", encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f if line.strip()]
    return len(rows), len(rows[0]) if rows else 0


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir / "dkt_concept"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = convert_file(
        data_dir / args.train_file,
        output_dir / "train_valid_sequences.csv",
        output_dir / "train_valid_sequences_full.csv",
        args.maxlen,
        args.fold_count,
        is_test=False,
    )
    test_summary = convert_file(
        data_dir / args.test_file,
        output_dir / "test_sequences.csv",
        output_dir / "test_sequences_full.csv",
        args.maxlen,
        args.fold_count,
        is_test=True,
    )

    num_q, num_c = count_q(data_dir / args.q_file)
    summary = {
        "source_dir": str(data_dir),
        "output_dir": str(output_dir),
        "num_q": num_q,
        "num_c": num_c,
        "maxlen": args.maxlen,
        "fold_count": args.fold_count,
        "train": train_summary,
        "test": test_summary,
        "pykt_dataset_name": f"{data_dir.parent.name}_er_dkt",
    }
    with (output_dir / "dkt_concept_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
