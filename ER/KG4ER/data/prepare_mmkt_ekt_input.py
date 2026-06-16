import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare three-line MMKT/EKT input files from ER prepared sequences.")
    parser.add_argument("--data-dir", required=True, help="Directory containing prepared_for_kt sequence files.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <data-dir>/mmkt_ekt.")
    parser.add_argument("--train-file", default="train_sequences.csv")
    parser.add_argument("--test-file", default="test_sequences.csv")
    parser.add_argument("--q-file", default="Q.txt")
    return parser.parse_args()


def parse_list(value):
    if pd.isna(value):
        return []
    return [token for token in str(value).split(",") if token not in ("", "-1")]


def first_concept(token):
    if "_" in token:
        return token.split("_")[0]
    return token


def convert_sequence(row, exercise_num):
    questions = parse_list(row["questions"])
    responses = parse_list(row["responses"])
    usable = min(len(questions), len(responses))
    encoded = []
    labels = []
    for idx in range(usable):
        qid = int(questions[idx])
        response = int(responses[idx])
        # MMKT/EKT KTData uses 1-based question ids; correct responses are q + exercise_num.
        one_based_qid = qid + 1
        encoded.append(one_based_qid + exercise_num if response == 1 else one_based_qid)
        labels.append(response)
    return encoded, labels


def write_three_line(input_path, output_path, exercise_num):
    df = pd.read_csv(input_path)
    kept = 0
    interactions = 0
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            encoded, labels = convert_sequence(row, exercise_num)
            if not encoded:
                continue
            f.write(str(len(encoded)) + "\n")
            f.write(",".join(map(str, encoded)) + "\n")
            f.write(",".join(map(str, labels)) + "\n")
            kept += 1
            interactions += len(encoded)
    return {"source": str(input_path), "output": str(output_path), "students": kept, "interactions": interactions}


def write_q_mapping(q_file, output_path):
    rows = []
    with q_file.open("r", encoding="utf-8") as f:
        for qid, line in enumerate(f):
            values = [int(x) for x in line.strip().split(",") if x != ""]
            concepts = [idx for idx, val in enumerate(values) if val == 1]
            if not concepts:
                continue
            rows.append({"e": qid, "k": concepts[0], "all_k": "_".join(map(str, concepts))})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return {"output": str(output_path), "rows": len(rows), "multi_concept_rows": sum(1 for row in rows if "_" in row["all_k"])}


def count_q(path):
    with path.open("r", encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f if line.strip()]
    return len(rows), len(rows[0]) if rows else 0


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir / "mmkt_ekt"
    output_dir.mkdir(parents=True, exist_ok=True)

    exercise_num, knowledge_num = count_q(data_dir / args.q_file)
    train = write_three_line(data_dir / args.train_file, output_dir / "train_3line.txt", exercise_num)
    test = write_three_line(data_dir / args.test_file, output_dir / "test_3line.txt", exercise_num)
    q_mapping = write_q_mapping(data_dir / args.q_file, output_dir / "question_to_first_concept.csv")

    warning = (
        "Existing KT/MMKT EKTM code is Eedi-bound: model_EKT.EKTM reads "
        "./data_prep/processed_Eedi_que2skill_new.csv and ./Eedi2020_withoption_addPAD.txt. "
        "Use these files only after training an Algebra-specific EKT/MMKT checkpoint or refactoring EKTM "
        "to accept dataset-specific Q/text inputs."
    )
    summary = {
        "source_dir": str(data_dir),
        "output_dir": str(output_dir),
        "exercise_num": exercise_num,
        "knowledge_num": knowledge_num,
        "train": train,
        "test": test,
        "q_mapping": q_mapping,
        "compatibility_warning": warning,
        "expected_mastery_shape_after_valid_export": [test["students"], knowledge_num],
    }
    with (output_dir / "mmkt_ekt_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
