import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from model import KGEModel


def test_distmult_forward_supports_single_and_negative_batches():
    model = KGEModel("DistMult", 5, 3, 4, 12.0, triplere_u=1.0)
    positive_sample = torch.LongTensor([[0, 1, 2], [3, 2, 4]])
    negative_tail = torch.LongTensor([[1, 2, 3], [0, 1, 2]])

    single_score = model(positive_sample)
    tail_score = model((positive_sample, negative_tail), mode="tail-batch")

    assert single_score.shape == (2, 1)
    assert tail_score.shape == (2, 3)


def test_complex_forward_supports_single_and_negative_batches():
    model = KGEModel(
        "ComplEx",
        5,
        3,
        4,
        12.0,
        triplere_u=1.0,
        double_entity_embedding=True,
        double_relation_embedding=True,
    )
    positive_sample = torch.LongTensor([[0, 1, 2], [3, 2, 4]])
    negative_head = torch.LongTensor([[1, 2, 3], [0, 1, 2]])

    single_score = model(positive_sample)
    head_score = model((positive_sample, negative_head), mode="head-batch")

    assert single_score.shape == (2, 1)
    assert head_score.shape == (2, 3)
