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


def test_one_click_conve_defaults_match_original_protocol():
    args = parse_args(["--dataset", "Eedi"])

    assert args.epochs == 10
    assert args.conve_input_drop == 0.2
    assert args.conve_hidden_drop == 0.2
    assert args.conve_feat_drop == 0.3
    assert args.conve_negative_ratio == 5
    assert args.conve_include_test_triples is True


def test_one_click_kge_defaults_match_original_run_sh():
    args = parse_args(["--dataset", "Eedi"])

    assert args.kge_max_steps == 30000
    assert args.kge_batch_size == 1024
    assert args.negative_sample_size == 256
    assert args.kge_hidden_dim == 1000
    assert args.kge_gamma == 12.0
    assert args.kge_learning_rate == 0.001


def test_one_click_can_opt_out_of_training_on_test_triples():
    args = parse_args(["--dataset", "Eedi", "--conve-exclude-test-triples"])

    assert args.conve_include_test_triples is False
