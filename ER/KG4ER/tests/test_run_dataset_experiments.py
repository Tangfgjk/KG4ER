import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from run_dataset_experiments import KGE_EXPERIMENTS


def test_transe_adv_experiment_uses_transe_with_adversarial_sampling():
    assert "TransE-adv" in KGE_EXPERIMENTS
    config = KGE_EXPERIMENTS["TransE-adv"]

    assert config["model"] == "TransE"
    assert "--negative_adversarial_sampling" in config["args"]
    assert "--adversarial_temperature" in config["args"]
