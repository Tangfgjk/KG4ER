import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create KG entity dictionary for ER datasets.")
    parser.add_argument("--data-dir", default="Eedi", help="Dataset directory. Relative paths are resolved from this script directory.")
    parser.add_argument("--mastery-file", default="stu2know_mastery.json")
    parser.add_argument("--q-file", default="Q.txt")
    parser.add_argument("--output-file", default="entities.dict")
    return parser.parse_args()


def resolve_data_dir(data_dir):
    path = Path(data_dir)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_q_matrix(path):
    with path.open("r", encoding="utf-8") as f:
        return [[int(x) for x in line.strip().split(",")] for line in f if line.strip()]


def write_entities_dict(file_path, student_count, knowledge_count, exercise_count):
    with file_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp, delimiter="\t")
        entity_id = 0
        for idx in range(student_count):
            writer.writerow([entity_id, f"uid{idx}"])
            entity_id += 1
        for idx in range(knowledge_count):
            writer.writerow([entity_id, f"kc{idx}"])
            entity_id += 1
        for idx in range(exercise_count):
            writer.writerow([entity_id, f"ex{idx}"])
            entity_id += 1


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    with (data_dir / args.mastery_file).open("r", encoding="utf-8") as fp:
        stu2know_mastery = json.load(fp)
    q_matrix = load_q_matrix(data_dir / args.q_file)

    write_entities_dict(
        data_dir / args.output_file,
        student_count=len(stu2know_mastery),
        knowledge_count=len(q_matrix[0]),
        exercise_count=len(q_matrix),
    )


if __name__ == "__main__":
    main()
