import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from explain_recommendations import generate_explanation_cards


def test_generate_explanation_cards_uses_top_scores_and_cognitive_relations():
    q_matrix = [
        [1, 0, 1],
        [0, 1, 0],
    ]
    uid_ex_scores = [("uid0", [0.2, 0.9])]
    uid_mlkc_dict = {
        "uid0": {
            "kc0": "mlkc0.60",
            "kc1": "mlkc0.70",
            "kc2": "mlkc0.80",
        }
    }
    uid_pkc_dict = {
        "uid0": {
            "kc0": "pkc0.10",
            "kc1": "pkc0.40",
            "kc2": "pkc0.30",
        }
    }
    uid_exfr_dict = {"uid0": {"ex1": "exfr0.85"}}

    cards = generate_explanation_cards(
        uid_ex_scores=uid_ex_scores,
        q_matrix=q_matrix,
        uid_mlkc_dict=uid_mlkc_dict,
        uid_pkc_dict=uid_pkc_dict,
        uid_exfr_dict=uid_exfr_dict,
        top_k=1,
        user_limit=1,
    )

    assert cards["top_k"] == 1
    assert cards["students"][0]["uid"] == "uid0"
    recommendation = cards["students"][0]["recommendations"][0]
    assert recommendation["exercise_id"] == "ex1"
    assert recommendation["conve_score"] == 0.9
    assert recommendation["exercise_forgetting"] == 0.85
    assert recommendation["knowledge_concepts"] == [
        {"kc_id": "kc1", "mastery": 0.7, "sequence_progress": 0.4}
    ]
    assert "uid0" in recommendation["explanation"]
    assert "ex1" in recommendation["explanation"]
