import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from score_kge_recommendations import rotate, score_exercise_candidate


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
