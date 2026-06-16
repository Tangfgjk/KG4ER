import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Register an ER prepared dataset for pyKT DKT training.")
    parser.add_argument("--data-dir", required=True, help="ER prepared_for_kt directory.")
    parser.add_argument("--pykt-root", default="ER/pykt-toolkit-main")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dkt-dir-name", default="dkt_concept")
    parser.add_argument("--maxlen", type=int, default=None)
    parser.add_argument("--fold-count", type=int, default=None)
    return parser.parse_args()


def load_manifest(dkt_dir):
    manifest_path = dkt_dir / "dkt_concept_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing DKT manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    pykt_root = Path(args.pykt_root).resolve()
    dkt_dir = data_dir / args.dkt_dir_name
    manifest = load_manifest(dkt_dir)

    dataset_name = args.dataset_name or manifest.get("pykt_dataset_name") or f"{data_dir.parent.name}_er_dkt"
    config_path = pykt_root / "configs" / "data_config.json"
    examples_dir = pykt_root / "examples"
    relative_dpath = os.path.relpath(dkt_dir, examples_dir).replace("\\", "/")

    with config_path.open("r", encoding="utf-8") as f:
        data_config = json.load(f)

    maxlen = args.maxlen if args.maxlen is not None else int(manifest["maxlen"])
    fold_count = args.fold_count if args.fold_count is not None else int(manifest["fold_count"])
    data_config[dataset_name] = {
        "dpath": relative_dpath,
        "num_q": int(manifest["num_q"]),
        "num_c": int(manifest["num_c"]),
        "input_type": ["concepts"],
        "max_concepts": 1,
        "min_seq_len": 3,
        "maxlen": maxlen,
        "emb_path": "",
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv",
        "folds": list(range(fold_count)),
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_sequences.csv",
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
        "train_valid_file_quelevel": "train_valid_sequences.csv",
        "test_file_quelevel": "test_sequences.csv",
        "test_window_file_quelevel": "test_sequences.csv",
        "test_original_file_quelevel": "test_sequences.csv",
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(data_config, f, ensure_ascii=False, indent=4)
        f.write("\n")

    summary = {
        "status": "registered",
        "dataset_name": dataset_name,
        "config_path": str(config_path),
        "dpath": relative_dpath,
        "num_q": int(manifest["num_q"]),
        "num_c": int(manifest["num_c"]),
        "maxlen": maxlen,
        "folds": list(range(fold_count)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
