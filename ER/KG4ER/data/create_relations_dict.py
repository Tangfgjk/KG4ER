import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create KG relation dictionary from ER triples.")
    parser.add_argument("--data-dir", default="Eedi", help="Dataset directory. Relative paths are resolved from this script directory.")
    parser.add_argument("--train-triples-file", default="triples.txt")
    parser.add_argument("--test-triples-file", default="test_triples.txt")
    parser.add_argument("--output-file", default="relations.dict")
    parser.add_argument(
        "--fixed-kg4er-relations",
        action="store_true",
        help="Write the KG4ER fixed 304-relation vocabulary: mlkc/pkc/exfr 0.00-1.00 plus rec.",
    )
    return parser.parse_args()


def resolve_data_dir(data_dir):
    path = Path(data_dir)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def collect_relations(path):
    relations = set()
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid triple line in {path}: {line!r}")
            relations.add(parts[1])
    return relations


def relation_sort_key(value):
    if value == "rec":
        return (3, 0.0, value)
    for prefix, rank in [("mlkc", 0), ("pkc", 1), ("exfr", 2)]:
        if value.startswith(prefix):
            try:
                return (rank, float(value[len(prefix):]), value)
            except ValueError:
                return (rank, 0.0, value)
    return (4, 0.0, value)


def fixed_kg4er_relations():
    relations = []
    for prefix in ["mlkc", "pkc", "exfr"]:
        relations.extend(f"{prefix}{idx / 100:.2f}" for idx in range(101))
    relations.append("rec")
    return relations


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    if args.fixed_kg4er_relations:
        ordered = fixed_kg4er_relations()
    else:
        relations = set()
        relations.update(collect_relations(data_dir / args.train_triples_file))
        relations.update(collect_relations(data_dir / args.test_triples_file))
        ordered = sorted(relations, key=relation_sort_key)

    with (data_dir / args.output_file).open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp, delimiter="\t")
        for idx, relation in enumerate(ordered):
            writer.writerow([idx, relation])

    print(f"relation_count: {len(ordered)}")


if __name__ == "__main__":
    main()
