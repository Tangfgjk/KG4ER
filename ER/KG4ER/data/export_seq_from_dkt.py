import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export stu2know_seq.json from a trained pyKT DKT checkpoint.")
    parser.add_argument("--pykt-root", default="ER/pykt-toolkit-main")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing config.json and qid_model.ckpt.")
    parser.add_argument("--checkpoint-file", default=None, help="Defaults to <emb_type>_model.ckpt. Use qid_last_model.ckpt to export the last epoch.")
    parser.add_argument("--test-sequences", required=True, help="Single-concept full test sequence CSV.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--expected-students", type=int, default=None)
    parser.add_argument("--expected-concepts", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def parse_ints(value):
    if pd.isna(value):
        return []
    return [int(token) for token in str(value).split(",") if token not in ("", "-1")]


def load_pykt_model(pykt_root, checkpoint_dir, device, checkpoint_file=None):
    pykt_root = Path(pykt_root).resolve()
    sys.path.insert(0, str(pykt_root))

    from pykt.models import init_model
    from pykt.models.torch_io import torch_load_file

    checkpoint_dir = Path(checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing pyKT config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_config = copy.deepcopy(config["model_config"])
    for remove_item in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
        model_config.pop(remove_item, None)

    params = config["params"]
    model_name = params["model_name"]
    emb_type = params["emb_type"]
    if model_name != "dkt":
        raise ValueError(f"This exporter expects pyKT DKT, got model_name={model_name!r}")

    data_config = config["data_config"]
    model = init_model(model_name, model_config, data_config, emb_type)
    ckpt_name = checkpoint_file or f"{emb_type}_model.ckpt"
    ckpt_path = checkpoint_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")
    net = torch_load_file(ckpt_path, map_location=device)
    model.load_state_dict(net)
    model.to(device)
    model.eval()
    return model, data_config


@torch.no_grad()
def export_seq(model, data_config, test_sequences, output_file, device, expected_students=None, expected_concepts=None):
    num_c = int(data_config["num_c"])
    if expected_concepts is not None and num_c != expected_concepts:
        raise ValueError(f"Checkpoint num_c={num_c}, expected_concepts={expected_concepts}")

    df = pd.read_csv(test_sequences)
    rows = []
    for _, row in df.iterrows():
        concepts = parse_ints(row["concepts"])
        responses = parse_ints(row["responses"])
        usable = min(len(concepts), len(responses))
        concepts = concepts[:usable]
        responses = responses[:usable]
        if usable == 0:
            rows.append([0.0] * num_c)
            continue
        if max(concepts) >= num_c:
            raise ValueError(f"Concept id {max(concepts)} is outside checkpoint num_c={num_c}")

        c = torch.tensor([concepts], dtype=torch.long, device=device)
        r = torch.tensor([responses], dtype=torch.long, device=device)
        y = model(c, r)
        rows.append([round(float(x), 4) for x in y[0, usable - 1, :].detach().cpu().tolist()])

    if expected_students is not None and len(rows) != expected_students:
        raise ValueError(f"Exported students={len(rows)}, expected_students={expected_students}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f)

    print(json.dumps({"output_file": str(output_file), "students": len(rows), "concepts": num_c}, indent=2))


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, data_config = load_pykt_model(args.pykt_root, args.checkpoint_dir, device, args.checkpoint_file)
    export_seq(
        model,
        data_config,
        Path(args.test_sequences),
        Path(args.output_file),
        device,
        expected_students=args.expected_students,
        expected_concepts=args.expected_concepts,
    )


if __name__ == "__main__":
    main()
