import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


DATASET_DEFAULTS = {
    "assist2009": {
        "raw_file": Path("assist2009/raw-data/skill_builder_data_corrected_collapsed.csv"),
        "sep": ",",
        "student": "user_id",
        "exercise": ["problem_id"],
        "concept": "skill_id",
        "response": "correct",
        "time": "order_id",
        "encoding": "latin1",
    },
    "algebra2005": {
        "raw_file": Path("algebra2005/raw-data/algebra_2005_2006 (1)/algebra_2005_2006_train.txt"),
        "sep": "\t",
        "student": "Anon Student Id",
        "exercise": ["Problem Name"],
        "concept": "KC(Default)",
        "response": "Correct First Attempt",
        "time": "First Transaction Time",
        "encoding": "utf-8",
    },
    "statics2011": {
        "raw_file": Path("statics2011/raw-data/statics2011/AllData_student_step_2011F.csv"),
        "sep": ",",
        "student": "Anon Student Id",
        "exercise": ["Problem Name", "Step Name"],
        "concept": "KC (F2011)",
        "response": "First Attempt",
        "time": "First Transaction Time",
        "encoding": "utf-8",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize ASSIST/Algebra/Statics raw data into ER sequence/Q files.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS))
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--raw-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-student-interactions", type=int, default=1)
    parser.add_argument("--min-exercise-interactions", type=int, default=1)
    parser.add_argument("--max-students", type=int, default=None)
    parser.add_argument("--split-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _is_missing(value):
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan" or text == "."


def parse_concepts(value, dataset):
    if _is_missing(value):
        return []
    text = str(value).strip()
    if dataset == "algebra2005":
        return [part.strip() for part in text.split("~~") if part.strip()]
    if dataset == "assist2009":
        return [text]
    return [text]


def parse_response(value, dataset):
    if _is_missing(value):
        return None
    if dataset == "statics2011":
        text = str(value).strip().lower()
        if text == "correct":
            return 1
        if text in {"incorrect", "hint"}:
            return 0
        return None
    try:
        parsed = int(float(value))
    except ValueError:
        return None
    return parsed if parsed in (0, 1) else None


def exercise_key(row, columns):
    return " || ".join(str(row[col]).strip() for col in columns)


def normalize_dataset(args):
    defaults = DATASET_DEFAULTS[args.dataset]
    raw_file = args.raw_file or (args.data_root / defaults["raw_file"])
    output_dir = args.output_dir or (args.data_root / args.dataset / "processed")
    frame = pd.read_csv(raw_file, sep=defaults["sep"], low_memory=False, encoding=defaults.get("encoding", "utf-8"))

    rows = []
    for row_idx, row in frame.iterrows():
        concepts = parse_concepts(row[defaults["concept"]], args.dataset)
        response = parse_response(row[defaults["response"]], args.dataset)
        if not concepts or response is None:
            continue
        uid = str(row[defaults["student"]]).strip()
        qid = exercise_key(row, defaults["exercise"])
        if _is_missing(uid) or _is_missing(qid):
            continue
        rows.append(
            {
                "original_uid": uid,
                "original_question": qid,
                "original_concepts": concepts,
                "response": response,
                "timestamp": str(row[defaults["time"]]).strip() if defaults["time"] in frame.columns else str(row_idx),
                "row_index": int(row_idx),
            }
        )

    student_counts = defaultdict(int)
    question_counts = defaultdict(int)
    for item in rows:
        student_counts[item["original_uid"]] += 1
        question_counts[item["original_question"]] += 1

    eligible_students = [
        uid for uid, count in student_counts.items() if count >= args.min_student_interactions
    ]
    eligible_students = sorted(eligible_students, key=lambda uid: (-student_counts[uid], uid))
    if args.max_students:
        eligible_students = eligible_students[: args.max_students]
    eligible_students = set(eligible_students)
    eligible_questions = {
        qid for qid, count in question_counts.items() if count >= args.min_exercise_interactions
    }

    filtered = [
        item
        for item in rows
        if item["original_uid"] in eligible_students and item["original_question"] in eligible_questions
    ]
    users = sorted({item["original_uid"] for item in filtered})
    questions = sorted({item["original_question"] for item in filtered})
    concepts = sorted({concept for item in filtered for concept in item["original_concepts"]})
    uid_map = {uid: idx for idx, uid in enumerate(users)}
    qid_map = {qid: idx for idx, qid in enumerate(questions)}
    cid_map = {cid: idx for idx, cid in enumerate(concepts)}

    output_dir.mkdir(parents=True, exist_ok=True)
    q_to_concepts = defaultdict(set)
    by_user = defaultdict(list)
    long_rows = []
    for item in sorted(filtered, key=lambda row: (uid_map[row["original_uid"]], row["timestamp"], row["row_index"])):
        uid = uid_map[item["original_uid"]]
        qid = qid_map[item["original_question"]]
        cids = sorted(cid_map[concept] for concept in item["original_concepts"])
        q_to_concepts[qid].update(cids)
        by_user[uid].append((qid, cids, item["response"], item["timestamp"]))
        long_rows.append(
            {
                "uid": uid,
                "original_uid": item["original_uid"],
                "question": qid,
                "original_question": item["original_question"],
                "concepts": "_".join(str(cid) for cid in cids),
                "response": item["response"],
                "timestamp": item["timestamp"],
            }
        )

    with (output_dir / "Q.txt").open("w", encoding="utf-8") as fp:
        for qid in range(len(questions)):
            fp.write(",".join("1" if cid in q_to_concepts[qid] else "0" for cid in range(len(concepts))) + "\n")

    with (output_dir / "sequence_interactions.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["uid", "original_uid", "question", "original_question", "concepts", "response", "timestamp"])
        writer.writeheader()
        writer.writerows(long_rows)

    split_point = int(len(users) * args.split_ratio)
    train_users = set(range(split_point))
    for split_name, user_filter in [("train", train_users), ("test", set(range(split_point, len(users))))]:
        with (output_dir / f"{split_name}_sequences.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["fold", "uid", "questions", "concepts", "responses", "timestamps", "selectmasks"])
            writer.writeheader()
            for uid in sorted(user_filter):
                rows_for_user = by_user.get(uid, [])
                if not rows_for_user:
                    continue
                writer.writerow(
                    {
                        "fold": 0,
                        "uid": uid,
                        "questions": ",".join(str(row[0]) for row in rows_for_user),
                        "concepts": ",".join("_".join(str(cid) for cid in row[1]) for row in rows_for_user),
                        "responses": ",".join(str(row[2]) for row in rows_for_user),
                        "timestamps": ",".join(str(row[3]) for row in rows_for_user),
                        "selectmasks": ",".join("1" for _ in rows_for_user),
                    }
                )

    with (output_dir / "id_maps.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "uid_map": uid_map,
                "qid_map": qid_map,
                "cid_map": cid_map,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    stats = {
        "dataset": args.dataset,
        "raw_file": str(raw_file),
        "output_dir": str(output_dir),
        "students": len(users),
        "questions": len(questions),
        "concepts": len(concepts),
        "interactions": len(long_rows),
        "train_students": len(train_users),
        "test_students": len(users) - len(train_users),
        "dropped_interactions": len(frame) - len(long_rows),
        "exercise_columns": defaults["exercise"],
        "concept_column": defaults["concept"],
    }
    with (output_dir / "processed_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    return stats


def main():
    args = parse_args()
    stats = normalize_dataset(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
