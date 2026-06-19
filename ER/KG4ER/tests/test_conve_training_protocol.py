import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from run_ConvE import (
    Args,
    MyDataset,
    build_positive_tails_by_hr,
    build_tail_candidates,
    training_triple_files,
)


def test_conve_defaults_match_original_protocol():
    args = Args()

    assert args.epochs == 25
    assert args.input_drop == 0.2
    assert args.hidden_drop == 0.2
    assert args.feat_drop == 0.3
    assert args.include_test_triples is True
    assert args.negative_ratio == 5
    assert training_triple_files(args) == ["triples.txt", "test_triples.txt"]


def test_conve_test_triples_can_be_excluded_for_diagnostics():
    args = Namespace(include_test_triples=False)

    assert training_triple_files(args) == ["triples.txt"]


def test_conve_negative_sampling_is_applied_only_to_rec_triples():
    entity2id = {"uid0": 0, "uid1": 1, "ex0": 2, "ex1": 3, "kc0": 4}
    relation2id = {"rec": 0, "mlkc0.50": 1}
    id2entity = {value: key for key, value in entity2id.items()}
    id2relation = {value: key for key, value in relation2id.items()}
    positive_triples = [
        (entity2id["uid0"], relation2id["rec"], entity2id["ex0"]),
        (entity2id["kc0"], relation2id["mlkc0.50"], entity2id["uid0"]),
        (entity2id["kc0"], relation2id["mlkc0.50"], entity2id["uid1"]),
    ]
    dataset = MyDataset(
        positive_triples,
        entity2id,
        relation2id,
        cuda=False,
        negative_ratio=1,
        relation_id_to_name=id2relation,
        entity_id_to_name=id2entity,
        tail_candidates_by_type=build_tail_candidates(entity2id),
        positive_tails_by_hr=build_positive_tails_by_hr(positive_triples),
        seed=2024,
    )

    assert len(dataset) == 4

    _, rec_relation, rec_tail, rec_label = dataset[1]
    _, mlkc_relation, mlkc_tail, mlkc_label = dataset[2]

    assert rec_label.item() == 0.0
    assert mlkc_label.item() == 1.0
    assert rec_relation.item() == relation2id["rec"]
    assert id2entity[rec_tail.item()].startswith("ex")
    assert rec_tail.item() != entity2id["ex0"]
    assert mlkc_relation.item() == relation2id["mlkc0.50"]
    assert id2entity[mlkc_tail.item()].startswith("uid")
    assert mlkc_tail.item() == entity2id["uid0"]
