import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build a reproducible XES3G5M-sub dataset for ER experiments.")
    default_raw = Path(__file__).resolve().parent / "XES3G5M" / "raw-data" / "XES3G5M" / "question_level"
    default_out = Path(__file__).resolve().parent / "XES3G5M-sub"
    parser.add_argument("--raw-dir", type=Path, default=default_raw)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--train-file", default="train_valid_sequences_quelevel.csv")
    parser.add_argument("--test-file", default="test_quelevel.csv")
    parser.add_argument("--target-students", type=int, default=3200)
    parser.add_argument("--target-questions", type=int, default=2000)
    parser.add_argument("--target-concepts", type=int, default=420)
    parser.add_argument("--min-student-interactions", type=int, default=80)
    parser.add_argument("--max-student-interactions", type=int, default=600)
    parser.add_argument("--min-question-interactions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _split_ints(text):
    if pd.isna(text):
        return []
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            values.append(-1)
    return values


def _split_concepts(text):
    if pd.isna(text):
        return []
    concepts = []
    for item in str(text).split(","):
        item = item.strip()
        if not item or item == "-1":
            concepts.append([])
            continue
        item_concepts = []
        for concept in item.split("_"):
            try:
                cid = int(concept)
            except ValueError:
                cid = -1
            if cid >= 0:
                item_concepts.append(cid)
        concepts.append(item_concepts)
    return concepts


def _valid_mask(selectmasks, questions, responses, timestamps, concepts):
    max_len = min(len(questions), len(responses), len(timestamps), len(concepts), len(selectmasks))
    for idx in range(max_len):
        if selectmasks[idx] != 1:
            continue
        if questions[idx] < 0 or responses[idx] not in (0, 1) or timestamps[idx] < 0:
            continue
        if not concepts[idx]:
            continue
        yield idx


def iter_interactions(path, source_split):
    frame = pd.read_csv(path, low_memory=False)
    for _, row in frame.iterrows():
        uid = int(row["uid"])
        questions = _split_ints(row["questions"])
        concepts = _split_concepts(row["concepts"])
        responses = _split_ints(row["responses"])
        timestamps = _split_ints(row["timestamps"])
        if "selectmasks" in row:
            selectmasks = _split_ints(row["selectmasks"])
        else:
            selectmasks = [1] * len(questions)
        for idx in _valid_mask(selectmasks, questions, responses, timestamps, concepts):
            yield {
                "uid": uid,
                "question": questions[idx],
                "concepts": tuple(concepts[idx]),
                "response": responses[idx],
                "timestamp": timestamps[idx],
                "source_split": source_split,
            }


def select_subset(interactions, args):
    student_counts = Counter(item["uid"] for item in interactions)
    eligible_students = [
        uid
        for uid, count in student_counts.items()
        if args.min_student_interactions <= count <= args.max_student_interactions
    ]
    eligible_students = sorted(eligible_students, key=lambda uid: (-student_counts[uid], uid))
    selected_students = set(eligible_students[: args.target_students])

    student_interactions = [item for item in interactions if item["uid"] in selected_students]
    concept_counts = Counter(cid for item in student_interactions for cid in item["concepts"])
    selected_concepts = set(
        cid for cid, _ in sorted(concept_counts.items(), key=lambda item: (-item[1], item[0]))[: args.target_concepts]
    )

    question_counts = Counter()
    question_concepts = defaultdict(Counter)
    for item in student_interactions:
        kept_concepts = [cid for cid in item["concepts"] if cid in selected_concepts]
        if not kept_concepts:
            continue
        question_counts[item["question"]] += 1
        for cid in kept_concepts:
            question_concepts[item["question"]][cid] += 1

    eligible_questions = [
        qid for qid, count in question_counts.items() if count >= args.min_question_interactions
    ]
    eligible_questions = sorted(eligible_questions, key=lambda qid: (-question_counts[qid], qid))
    selected_questions = set(eligible_questions[: args.target_questions])

    filtered = []
    for item in student_interactions:
        if item["question"] not in selected_questions:
            continue
        kept_concepts = tuple(cid for cid in item["concepts"] if cid in selected_concepts)
        if not kept_concepts:
            continue
        filtered.append({**item, "concepts": kept_concepts})

    return filtered, selected_students, selected_questions, selected_concepts


def write_outputs(filtered, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    users = sorted({item["uid"] for item in filtered})
    questions = sorted({item["question"] for item in filtered})
    concepts = sorted({cid for item in filtered for cid in item["concepts"]})
    uid_map = {old: new for new, old in enumerate(users)}
    qid_map = {old: new for new, old in enumerate(questions)}
    cid_map = {old: new for new, old in enumerate(concepts)}

    q_to_concepts = defaultdict(set)
    by_user_split = defaultdict(lambda: defaultdict(list))
    long_rows = []
    for item in sorted(filtered, key=lambda row: (row["uid"], row["timestamp"], row["question"])):
        uid = uid_map[item["uid"]]
        qid = qid_map[item["question"]]
        cids = sorted(cid_map[cid] for cid in item["concepts"] if cid in cid_map)
        if not cids:
            continue
        q_to_concepts[qid].update(cids)
        by_user_split[item["source_split"]][uid].append((qid, cids, item["response"], item["timestamp"]))
        long_rows.append(
            {
                "uid": uid,
                "original_uid": item["uid"],
                "question": qid,
                "original_question": item["question"],
                "concepts": "_".join(str(cid) for cid in cids),
                "response": item["response"],
                "timestamp": item["timestamp"],
                "source_split": item["source_split"],
            }
        )

    with (args.output_dir / "Q.txt").open("w", encoding="utf-8") as fp:
        for qid in range(len(questions)):
            row = ["1" if cid in q_to_concepts[qid] else "0" for cid in range(len(concepts))]
            fp.write(",".join(row) + "\n")

    with (args.output_dir / "interactions.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "uid",
                "original_uid",
                "question",
                "original_question",
                "concepts",
                "response",
                "timestamp",
                "source_split",
            ],
        )
        writer.writeheader()
        writer.writerows(long_rows)

    for split_name, file_name in [("train", "train_valid_sequences_quelevel.csv"), ("test", "test_quelevel.csv")]:
        with (args.output_dir / file_name).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["fold", "uid", "questions", "concepts", "responses", "timestamps", "selectmasks"],
            )
            writer.writeheader()
            for uid, rows in sorted(by_user_split[split_name].items()):
                if not rows:
                    continue
                writer.writerow(
                    {
                        "fold": 0,
                        "uid": uid,
                        "questions": ",".join(str(row[0]) for row in rows),
                        "concepts": ",".join("_".join(str(cid) for cid in row[1]) for row in rows),
                        "responses": ",".join(str(row[2]) for row in rows),
                        "timestamps": ",".join(str(row[3]) for row in rows),
                        "selectmasks": ",".join("1" for _ in rows),
                    }
                )

    maps = {
        "uid_map": {str(old): new for old, new in uid_map.items()},
        "qid_map": {str(old): new for old, new in qid_map.items()},
        "cid_map": {str(old): new for old, new in cid_map.items()},
    }
    with (args.output_dir / "id_maps.json").open("w", encoding="utf-8") as fp:
        json.dump(maps, fp, ensure_ascii=False, indent=2)

    stats = {
        "students": len(users),
        "questions": len(questions),
        "concepts": len(concepts),
        "interactions": len(long_rows),
        "train_interactions": sum(1 for row in long_rows if row["source_split"] == "train"),
        "test_interactions": sum(1 for row in long_rows if row["source_split"] == "test"),
        "filter": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with (args.output_dir / "subset_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    return stats


def main():
    args = parse_args()
    train_path = args.raw_dir / args.train_file
    test_path = args.raw_dir / args.test_file
    interactions = list(iter_interactions(train_path, "train"))
    interactions.extend(iter_interactions(test_path, "test"))
    filtered, _, _, _ = select_subset(interactions, args)
    stats = write_outputs(filtered, args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
