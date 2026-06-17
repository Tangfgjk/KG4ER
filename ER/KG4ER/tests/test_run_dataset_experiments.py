import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from run_dataset_experiments import KGE_EXPERIMENTS, parse_args


def test_transe_adv_experiment_uses_transe_with_adversarial_sampling():
    assert "TransE-adv" in KGE_EXPERIMENTS
    config = KGE_EXPERIMENTS["TransE-adv"]

    assert config["model"] == "TransE"
    assert "--negative_adversarial_sampling" in config["args"]
    assert "--adversarial_temperature" in config["args"]


def test_one_click_conve_defaults_match_paper_style_protocol():
    args = parse_args(["--dataset", "Eedi"])

    assert args.epochs == 10
    assert args.conve_input_drop == 0.2
    assert args.conve_hidden_drop == 0.2
    assert args.conve_feat_drop == 0.3
    assert args.conve_include_test_triples is False


def test_one_click_can_opt_into_training_on_test_triples():
    args = parse_args(["--dataset", "Eedi", "--conve-include-test-triples"])

    assert args.conve_include_test_triples is True
