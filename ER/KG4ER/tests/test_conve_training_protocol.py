import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from run_ConvE import Args, training_triple_files


def test_conve_defaults_match_paper_style_protocol():
    args = Args()

    assert args.epochs == 10
    assert args.input_drop == 0.2
    assert args.hidden_drop == 0.2
    assert args.feat_drop == 0.3
    assert args.include_test_triples is False
    assert training_triple_files(args) == ["triples.txt"]


def test_conve_test_triples_are_explicit_opt_in():
    args = Namespace(include_test_triples=True)

    assert training_triple_files(args) == ["triples.txt", "test_triples.txt"]
