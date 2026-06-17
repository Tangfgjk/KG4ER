import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_limit)
        break
    except OverflowError:
        field_limit = int(field_limit / 10)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a student-exercise-concept constrained ER subset from long interaction files."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--interaction-file", default=None)
    parser.add_argument("--target-questions", required=True, type=int)
    parser.add_argument("--min-student-interactions", default=10, type=int)
    parser.add_argument("--max-student-interactions", default=200, type=int)
    parser.add_argument("--min-question-interactions", default=20, type=int)
    parser.add_argument("--min-questions-per-concept", default=3, type=int)
    parser.add_argument("--split-ratio", default=0.75, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    return parser.parse_args()


def find_interaction_file(source_dir, requested=None):
    if requested:
        path = source_dir / requested
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for name in ["sequence_interactions.csv", "interactions.csv"]:
        path = source_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No interaction file found in {source_dir}")


def parse_concepts(value):
    concepts = []
    for token in str(value or "").replace(",", "_").split("_"):
        token = token.strip()
        if not token or token == "-1":
            continue
        concepts.append(int(token))
    return concepts


def load_q_matrix(path):
    q_matrix = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                q_matrix.append([int(value) for value in line.split(",")])
    if not q_matrix:
        raise ValueError(f"Empty Q matrix: {path}")
    width = len(q_matrix[0])
    if any(len(row) != width for row in q_matrix):
        raise ValueError(f"Inconsistent Q width: {path}")
    return q_matrix


def q_concepts(q_matrix, question):
    if question < 0 or question >= len(q_matrix):
        return []
    return [idx for idx, value in enumerate(q_matrix[question]) if int(value) == 1]


def load_interactions(source_dir, interaction_file, q_matrix):
    rows = []
    with interaction_file.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row_idx, row in enumerate(reader):
            try:
                uid = str(row["uid"])
                question = int(row["question"])
                response = int(row["response"])
            except (KeyError, TypeError, ValueError):
                continue
            if response not in (0, 1):
                continue
            concepts = parse_concepts(row.get("concepts"))
            if not concepts:
                concepts = q_concepts(q_matrix, question)
            concepts = [concept for concept in concepts if 0 <= concept < len(q_matrix[0])]
            if not concepts:
                continue
            if question < 0 or question >= len(q_matrix):
                continue
            rows.append(
                {
                    "source_uid": uid,
                    "original_uid": str(row.get("original_uid", uid)),
                    "source_question": question,
                    "original_question": str(row.get("original_question", question)),
                    "concepts": tuple(sorted(set(concepts))),
                    "response": response,
                    "timestamp": str(row.get("timestamp", row_idx)),
                    "row_index": row_idx,
                }
            )
    return rows


def student_lengths(rows):
    counts = Counter(row["source_uid"] for row in rows)
    return counts


def select_students(rows, min_len, max_len):
    counts = student_lengths(rows)
    return {uid for uid, count in counts.items() if min_len <= count <= max_len}


def select_questions(rows, q_matrix, target_questions, min_question_interactions, min_questions_per_concept):
    question_counts = Counter(row["source_question"] for row in rows)
    eligible_questions = {
        question
        for question, count in question_counts.items()
        if count >= min_question_interactions and q_concepts(q_matrix, question)
    }
    if not eligible_questions:
        raise ValueError("No eligible questions after filtering.")

    questions_by_concept = defaultdict(list)
    for question in eligible_questions:
        for concept in q_concepts(q_matrix, question):
            questions_by_concept[concept].append(question)

    for concept, questions in questions_by_concept.items():
        questions.sort(key=lambda question: (-question_counts[question], question))

    selected = set()
    concept_order = sorted(questions_by_concept, key=lambda concept: (len(questions_by_concept[concept]), concept))
    for concept in concept_order:
        for question in questions_by_concept[concept]:
            if len(selected) >= target_questions:
                break
            if sum(1 for item in questions_by_concept[concept] if item in selected) >= min_questions_per_concept:
                break
            selected.add(question)
        if len(selected) >= target_questions:
            break

    for question, _ in sorted(question_counts.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= target_questions:
            break
        if question in eligible_questions:
            selected.add(question)
    return selected


def natural_uid_key(uid):
    return (0, int(uid)) if str(uid).isdigit() else (1, str(uid))


def refine_questions_for_final_students(
    rows,
    q_matrix,
    selected_questions,
    target_questions,
    min_student_interactions,
    min_question_interactions,
):
    filtered = [row for row in rows if row["source_question"] in selected_questions]
    final_students = select_students(filtered, min_student_interactions, 10**9)
    if not final_students:
        return selected_questions, final_students

    final_rows = [row for row in rows if row["source_uid"] in final_students]
    final_question_counts = Counter(row["source_question"] for row in final_rows)
    selected = {question for question in selected_questions if final_question_counts[question] > 0}

    for question, count in sorted(final_question_counts.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= target_questions:
            break
        if count >= min_question_interactions and q_concepts(q_matrix, question):
            selected.add(question)
    return selected, final_students


def write_q(path, selected_questions, q_matrix, qid_map):
    with path.open("w", encoding="utf-8") as fp:
        for old_q in sorted(selected_questions, key=lambda q: qid_map[q]):
            fp.write(",".join(str(value) for value in q_matrix[old_q]) + "\n")


def write_sequences(path, users, rows_by_user):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["fold", "uid", "questions", "concepts", "responses", "timestamps", "selectmasks"],
        )
        writer.writeheader()
        for uid in users:
            rows = rows_by_user.get(uid, [])
            if not rows:
                continue
            writer.writerow(
                {
                    "fold": 0,
                    "uid": uid,
                    "questions": ",".join(str(row["question"]) for row in rows),
                    "concepts": ",".join("_".join(str(concept) for concept in row["concepts"]) for row in rows),
                    "responses": ",".join(str(row["response"]) for row in rows),
                    "timestamps": ",".join(str(row["timestamp"]) for row in rows),
                    "selectmasks": ",".join("1" for _ in rows),
                }
            )


def write_outputs(args, q_matrix, rows, selected_questions):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_questions = set(selected_questions)
    rows = [row for row in rows if row["source_question"] in selected_questions]
    selected_students = select_students(rows, args.min_student_interactions, args.max_student_interactions)
    rows = [row for row in rows if row["source_uid"] in selected_students]

    actual_questions = sorted({row["source_question"] for row in rows})
    actual_students = sorted({row["source_uid"] for row in rows}, key=natural_uid_key)
    rng = random.Random(args.seed)
    split_students = actual_students[:]
    rng.shuffle(split_students)
    split_point = int(len(split_students) * args.split_ratio)
    train_source_students = set(split_students[:split_point])

    uid_map = {old: new for new, old in enumerate(actual_students)}
    qid_map = {old: new for new, old in enumerate(actual_questions)}

    mapped_rows = []
    rows_by_user = defaultdict(list)
    for row in sorted(rows, key=lambda item: (uid_map[item["source_uid"]], item["timestamp"], item["row_index"])):
        mapped = {
            "uid": uid_map[row["source_uid"]],
            "original_uid": row["original_uid"],
            "question": qid_map[row["source_question"]],
            "original_question": row["original_question"],
            "concepts": row["concepts"],
            "response": row["response"],
            "timestamp": row["timestamp"],
            "split": "train" if row["source_uid"] in train_source_students else "test",
        }
        mapped_rows.append(mapped)
        rows_by_user[mapped["uid"]].append(mapped)

    write_q(args.output_dir / "Q.txt", actual_questions, q_matrix, qid_map)
    with (args.output_dir / "sequence_interactions.csv").open("w", encoding="utf-8", newline="") as fp:
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
        for row in mapped_rows:
            writer.writerow(
                {
                    "uid": row["uid"],
                    "original_uid": row["original_uid"],
                    "question": row["question"],
                    "original_question": row["original_question"],
                    "concepts": "_".join(str(concept) for concept in row["concepts"]),
                    "response": row["response"],
                    "timestamp": row["timestamp"],
                    "source_split": row["split"],
                }
            )

    train_users = sorted(uid_map[uid] for uid in train_source_students)
    test_users = sorted(uid_map[uid] for uid in actual_students if uid not in train_source_students)
    write_sequences(args.output_dir / "train_sequences.csv", train_users, rows_by_user)
    write_sequences(args.output_dir / "test_sequences.csv", test_users, rows_by_user)
    write_sequences(args.output_dir / "train_valid_sequences.csv", train_users, rows_by_user)
    write_sequences(args.output_dir / "train_valid_sequences_quelevel.csv", train_users, rows_by_user)
    write_sequences(args.output_dir / "test_quelevel.csv", test_users, rows_by_user)

    active_concepts = sorted({concept for row in mapped_rows for concept in row["concepts"]})
    with (args.output_dir / "id_maps.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "uid_map": {str(old): new for old, new in uid_map.items()},
                "qid_map": {str(old): new for old, new in qid_map.items()},
                "cid_map": {str(idx): idx for idx in range(len(q_matrix[0]))},
                "active_concepts": active_concepts,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    lengths = Counter(row["uid"] for row in mapped_rows)
    stats = {
        "dataset": args.dataset_name,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "students": len(actual_students),
        "train_students": len(train_users),
        "test_students": len(test_users),
        "questions": len(actual_questions),
        "concepts": len(q_matrix[0]),
        "active_concepts": len(active_concepts),
        "interactions": len(mapped_rows),
        "train_interactions": sum(1 for row in mapped_rows if row["split"] == "train"),
        "test_interactions": sum(1 for row in mapped_rows if row["split"] == "test"),
        "sequence_length": {
            "min": min(lengths.values()) if lengths else 0,
            "max": max(lengths.values()) if lengths else 0,
            "mean": round(sum(lengths.values()) / len(lengths), 4) if lengths else 0.0,
        },
        "filter": {
            "target_questions": args.target_questions,
            "min_student_interactions": args.min_student_interactions,
            "max_student_interactions": args.max_student_interactions,
            "min_question_interactions": args.min_question_interactions,
            "min_questions_per_concept": args.min_questions_per_concept,
            "split_ratio": args.split_ratio,
            "seed": args.seed,
            "concept_id_policy": "preserve_source_q_width",
        },
    }
    for name in ["processed_stats.json", "subset_stats.json"]:
        with (args.output_dir / name).open("w", encoding="utf-8") as fp:
            json.dump(stats, fp, ensure_ascii=False, indent=2)
    return stats


def main():
    args = parse_args()
    q_matrix = load_q_matrix(args.source_dir / "Q.txt")
    interaction_file = find_interaction_file(args.source_dir, args.interaction_file)
    rows = load_interactions(args.source_dir, interaction_file, q_matrix)
    initial_students = select_students(rows, args.min_student_interactions, args.max_student_interactions)
    student_rows = [row for row in rows if row["source_uid"] in initial_students]
    selected_questions = select_questions(
        student_rows,
        q_matrix,
        args.target_questions,
        args.min_question_interactions,
        args.min_questions_per_concept,
    )
    selected_questions, _ = refine_questions_for_final_students(
        student_rows,
        q_matrix,
        selected_questions,
        args.target_questions,
        args.min_student_interactions,
        args.min_question_interactions,
    )
    stats = write_outputs(args, q_matrix, student_rows, selected_questions)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
