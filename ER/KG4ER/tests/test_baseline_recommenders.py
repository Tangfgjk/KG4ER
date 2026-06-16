import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from baseline_recommenders import (
    build_all_baseline_scores,
    content_based_scores,
    kcp_er_scores,
    student_based_cf_scores,
)
from ep_sim import calculate_ep_sim


def test_baselines_return_uid_ex_score_format():
    q_matrix = [
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    mastery = [[0.2, 0.8], [0.9, 0.1]]
    sequence = [[0.7, 0.3], [0.2, 0.6]]
    forgetting = [[0.1, 0.8, 0.5], [0.4, 0.2, 0.9]]
    interactions = {
        "uid0": [(0, 1), (1, 0)],
        "uid1": [(1, 1), (2, 1)],
    }

    scores = build_all_baseline_scores(
        q_matrix=q_matrix,
        mastery=mastery,
        sequence=sequence,
        forgetting=forgetting,
        interactions=interactions,
        methods=["EB-CF", "SB-CF", "CBF", "KCP-ER"],
    )

    assert set(scores) == {"EB-CF", "SB-CF", "CBF", "KCP-ER"}
    for method_scores in scores.values():
        assert [uid for uid, _ in method_scores] == ["uid0", "uid1"]
        assert all(len(item_scores) == 3 for _, item_scores in method_scores)


def test_content_and_kcp_scores_prefer_weak_or_targeted_concepts():
    q_matrix = [[1, 0], [0, 1]]
    mastery = [0.2, 0.8]
    sequence = [0.0, 0.0]
    forgetting = [0.1, 0.1]

    cbf = content_based_scores(q_matrix, mastery, sequence, forgetting)
    kcp = kcp_er_scores(q_matrix, mastery, forgetting, target_mastery=0.8)

    assert cbf[0] > cbf[1]
    assert kcp[0] < kcp[1]


def test_student_based_cf_uses_similar_students():
    q_matrix = [[1, 0], [0, 1], [1, 1]]
    mastery = [[0.2, 0.8], [0.25, 0.75], [0.9, 0.1]]
    interactions = {
        "uid0": [],
        "uid1": [(2, 1)],
        "uid2": [(0, 1)],
    }

    uid_scores = student_based_cf_scores(q_matrix, mastery, interactions, user_ids=["uid0", "uid1", "uid2"])
    uid0_scores = dict(uid_scores)["uid0"]

    assert uid0_scores[2] > uid0_scores[0]


def test_calculate_ep_sim_returns_normalized_learning_gain():
    q_matrix = [[1, 0], [0, 1]]
    mastery = [[0.5, 0.5]]
    uid_ex_scores = [("uid0", [0.9, 0.1])]

    result = calculate_ep_sim(uid_ex_scores, q_matrix, mastery, top_k=1, learning_gain=0.2)

    assert result["mean"] == 0.2
    assert result["per_user"][0]["uid"] == "uid0"
