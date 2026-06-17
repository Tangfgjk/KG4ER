import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from score_kge_recommendations import rotate, score_exercise_candidate, score_fn, transe


def test_rotate_candidate_score_accepts_double_entity_single_relation_dimensions():
    entity_embedding = np.ones((3, 1000), dtype=np.float32)
    relation_embedding = np.ones((2, 500), dtype=np.float32)
    entity2id = {"kc0": 0, "ex0": 1, "ex1": 2}
    relation2id = {"mlkc0": 0, "exfr0": 1}

    score = score_exercise_candidate(
        scorer=rotate,
        entity_embedding=entity_embedding,
        relation_embedding=relation_embedding,
        entity2id=entity2id,
        relation2id=relation2id,
        exercise="ex0",
        cognitive_items=[("kc0", "mlkc0")],
        exfr_items={"ex0": "exfr0"},
        gamma=12.0,
    )

    assert isinstance(score, float)


def test_transe_adv_is_scored_with_transe_function():
    assert score_fn("TransE-adv") is transe


def test_transe_candidate_score_matches_original_er_formula():
    entity_embedding = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    relation_embedding = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.25],
            [0.1, 0.1],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    entity2id = {"kc0": 0, "ex0": 1}
    relation2id = {"mlkc0.5": 0, "pkc0.8": 1, "exfr0.2": 2, "rec": 3}

    score = score_exercise_candidate(
        scorer=transe,
        entity_embedding=entity_embedding,
        relation_embedding=relation_embedding,
        entity2id=entity2id,
        relation2id=relation2id,
        exercise="ex0",
        cognitive_items=[("kc0", "mlkc0.5"), ("kc0", "pkc0.8")],
        exfr_items={"ex0": "exfr0.2"},
        gamma=12.0,
        model_name="TransE",
        mlkc_count=1,
    )

    rec = relation_embedding[relation2id["rec"]]
    ex0 = entity_embedding[entity2id["ex0"]]
    expected = (
        12.0
        - np.linalg.norm(
            (entity_embedding[entity2id["kc0"]] + relation_embedding[relation2id["mlkc0.5"]] + rec) - ex0,
            ord=2,
        )
        + 12.0
        - np.linalg.norm(
            (entity_embedding[entity2id["kc0"]] + relation_embedding[relation2id["pkc0.8"]] + rec) - ex0,
            ord=2,
        )
    ) / 1
    expected += 12.0 - np.linalg.norm((ex0 + relation_embedding[relation2id["exfr0.2"]] + rec) - ex0, ord=2)

    assert np.isclose(score, expected)
