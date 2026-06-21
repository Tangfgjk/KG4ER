import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
DATA = ROOT / "ER" / "KG4ER" / "data"
sys.path.insert(0, str(CODES))
sys.path.insert(0, str(DATA))


from conve_ablation_data import (
    CONVE_VARIANTS,
    calculate_recommendation_scores,
    prepare_conve_variant_data_dir,
)
from step1_cal_recommend import parse_args as parse_recommend_args


def test_four_conve_variants_use_the_expected_distance_terms():
    mastery = [[0.5, 0.9]]
    sequence = [[0.5, 0.0]]
    forgetting = [[0.2, 0.9]]
    q_matrix = [[1, 0], [0, 1]]

    expected_first_exercise = {
        "ConvE_full": math.sqrt(0.09 + 1.0 + 0.36),
        "ConvE_no_mastery": math.sqrt(1.0 + 0.36),
        "ConvE_no_forgetting": math.sqrt(0.09 + 1.0),
        "ConvE_no_seq": math.sqrt(0.09 + 0.36),
    }

    for model, expected in expected_first_exercise.items():
        scores = calculate_recommendation_scores(
            mastery,
            sequence,
            forgetting,
            q_matrix,
            active_terms=CONVE_VARIANTS[model]["active_terms"],
        )
        assert scores[0][0] == pytest.approx(round(expected, 2))


def test_standalone_recommendation_script_defaults_to_full_msf():
    args = parse_recommend_args([])
    assert args.terms == "mastery,sequence,forgetting"


def test_legacy_use_seq_flag_remains_accepted_as_full_msf():
    args = parse_recommend_args(["--use-seq-term"])
    assert args.terms == "mastery,sequence,forgetting"


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_source_data(source_dir):
    source_dir.mkdir(parents=True)
    (source_dir / "Q.txt").write_text("1,0\n0,1\n0,1\n", encoding="utf-8")
    write_json(source_dir / "stu2know_mastery.json", [[0.8, 0.2], [0.6, 0.4]])
    write_json(source_dir / "stu2know_seq.json", [[1.0, 0.0], [0.0, 1.0]])
    write_json(source_dir / "stu2know_forget.json", [[0.1, 0.2], [0.3, 0.4]])
    write_json(source_dir / "stu2ex_forget.json", [[0.8, 0.2, 0.9], [0.3, 0.4, 0.5]])
    write_json(source_dir / "stu2ex_recommend.json", [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    (source_dir / "entities.dict").write_text(
        "0\tkc0\n1\tkc1\n2\tuid0\n3\tuid1\n4\tex0\n5\tex1\n6\tex2\n",
        encoding="utf-8",
    )
    (source_dir / "relations.dict").write_text(
        "0\tmlkc0.80\n1\tpkc1.00\n2\texfr0.80\n3\trec\n",
        encoding="utf-8",
    )
    (source_dir / "triples.txt").write_text(
        "kc0\tmlkc0.80\tuid0\n"
        "kc0\tpkc1.00\tuid0\n"
        "ex0\texfr0.80\tuid0\n"
        "uid0\trec\tex0\n",
        encoding="utf-8",
    )
    (source_dir / "test_triples.txt").write_text(
        "kc0\tmlkc0.60\tuid1\n"
        "kc1\tpkc1.00\tuid1\n"
        "ex0\texfr0.30\tuid1\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "model,removed_prefix",
    [
        ("ConvE_full", None),
        ("ConvE_no_mastery", "mlkc"),
        ("ConvE_no_forgetting", "exfr"),
        ("ConvE_no_seq", "pkc"),
    ],
)
def test_variant_data_rebuilds_rec_and_removes_target_relation(
    tmp_path, model, removed_prefix
):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / model
    make_source_data(source_dir)

    manifest = prepare_conve_variant_data_dir(
        source_dir,
        target_dir,
        model,
        top_k_rec=1,
    )

    train_lines = (target_dir / "triples.txt").read_text(encoding="utf-8").splitlines()
    test_lines = (target_dir / "test_triples.txt").read_text(encoding="utf-8").splitlines()
    rec_lines = [line for line in train_lines if "\trec\t" in line]

    assert len(rec_lines) == 1
    assert rec_lines[0].startswith("uid0\trec\t")
    assert "uid1\trec\t" not in "\n".join(train_lines + test_lines)
    assert manifest["model"] == model
    assert manifest["active_terms"] == list(CONVE_VARIANTS[model]["active_terms"])
    assert manifest["rec_triples"] == 1
    assert json.loads((target_dir / "variant_manifest.json").read_text(encoding="utf-8")) == manifest

    if removed_prefix:
        assert not any(
            line.split("\t")[1].startswith(removed_prefix)
            for line in train_lines + test_lines
        )


def test_full_variant_adds_sequence_term_and_replaces_old_rec(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "full"
    make_source_data(source_dir)

    prepare_conve_variant_data_dir(
        source_dir,
        target_dir,
        "ConvE_full",
        top_k_rec=1,
    )

    train_lines = (target_dir / "triples.txt").read_text(encoding="utf-8").splitlines()
    rec_lines = [line for line in train_lines if "\trec\t" in line]
    assert rec_lines == ["uid0\trec\tex2"]


def test_resume_rejects_legacy_variant_without_manifest(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "legacy"
    make_source_data(source_dir)
    target_dir.mkdir()
    (target_dir / "triples.txt").write_text("uid0\trec\tex0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="new run-id"):
        prepare_conve_variant_data_dir(
            source_dir,
            target_dir,
            "ConvE_full",
            top_k_rec=1,
            resume=True,
        )


def test_resume_reuses_matching_variant_manifest(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "variant"
    make_source_data(source_dir)
    expected = prepare_conve_variant_data_dir(
        source_dir,
        target_dir,
        "ConvE_full",
        top_k_rec=1,
    )

    actual = prepare_conve_variant_data_dir(
        source_dir,
        target_dir,
        "ConvE_full",
        top_k_rec=1,
        resume=True,
    )
    assert actual == expected
