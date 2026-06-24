import csv
import json
import statistics
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CODES = ROOT / "ER" / "KG4ER" / "codes"
sys.path.insert(0, str(CODES))


from summarize_dataset_results import (
    AVG_WEIGHTS,
    TOP_KS,
    aggregate_results,
    collect_dataset_results,
    validate_replacement_options,
    write_report,
)


def metric_payload(dataset, model, seed, value):
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "ACC": {
            str(top_k): {"mean": value + top_k / 10000, "std": 0.01}
            for top_k in TOP_KS
        },
        "NOV": {
            str(top_k): {"mean": value + top_k / 20000, "std": 0.02}
            for top_k in TOP_KS
        },
        "Ep_sim": {"top_k": 10, "mean": value / 2, "std": 0.03},
    }


def write_metric(run_dir, model, seed, value, dataset="demo", traditional=False):
    if traditional:
        metric_file = run_dir / "traditional_baselines" / model / "eval" / "metrics.json"
    else:
        metric_file = run_dir / model / f"seed{seed}" / "eval" / "metrics.json"
    metric_file.parent.mkdir(parents=True, exist_ok=True)
    metric_file.write_text(
        json.dumps(metric_payload(dataset, model, seed, value)),
        encoding="utf-8",
    )
    return metric_file


def write_timing(metric_file, training, inference, evaluation):
    timing_file = metric_file.parents[1] / "timing.json"
    timing_file.write_text(
        json.dumps(
            {
                "training": {"seconds": training},
                "inference_without_cache": {"seconds": inference},
                "evaluation_metric": {"seconds": evaluation},
            }
        ),
        encoding="utf-8",
    )
    return timing_file


def test_collect_and_aggregate_five_seeds_with_deterministic_baseline(tmp_path):
    run_dir = tmp_path / "run"
    seed_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for seed, value in zip(range(2024, 2029), seed_values):
        write_metric(run_dir, "ConvE_full", seed, value)
    write_metric(run_dir, "EB-CF", None, 0.25, traditional=True)

    rows, metadata = collect_dataset_results("demo", run_dir)
    summary = {row["model"]: row for row in aggregate_results(rows)}

    assert len(rows) == 6
    assert metadata["replacement_count"] == 0
    assert summary["ConvE_full"]["run_count"] == 5
    assert summary["ConvE_full"]["ACC@10_mean"] == pytest.approx(0.301)
    assert summary["ConvE_full"]["ACC@10_std"] == pytest.approx(
        statistics.stdev(value + 0.001 for value in seed_values)
    )
    conve_rows = [row for row in rows if row["model"] == "ConvE_full"]
    expected_acc_avgs = [
        sum(AVG_WEIGHTS[top_k] * row[f"ACC@{top_k}"] for top_k in TOP_KS)
        for row in conve_rows
    ]
    expected_nov_avgs = [
        sum(AVG_WEIGHTS[top_k] * row[f"NOV@{top_k}"] for top_k in TOP_KS)
        for row in conve_rows
    ]
    assert conve_rows[0]["ACC-Avg"] == pytest.approx(expected_acc_avgs[0])
    assert conve_rows[0]["NOV-Avg"] == pytest.approx(expected_nov_avgs[0])
    assert summary["ConvE_full"]["ACC-Avg_mean"] == pytest.approx(
        statistics.mean(expected_acc_avgs)
    )
    assert summary["ConvE_full"]["ACC-Avg_std"] == pytest.approx(
        statistics.stdev(expected_acc_avgs)
    )
    assert summary["ConvE_full"]["NOV-Avg_mean"] == pytest.approx(
        statistics.mean(expected_nov_avgs)
    )
    assert summary["ConvE_full"]["NOV-Avg_std"] == pytest.approx(
        statistics.stdev(expected_nov_avgs)
    )
    assert summary["EB-CF"]["run_count"] == 1
    assert summary["EB-CF"]["ACC@10_std"] is None
    assert summary["EB-CF"]["ACC-Avg_std"] is None


def test_aggregate_selects_seed_with_largest_complete_total_time(tmp_path):
    run_dir = tmp_path / "run"
    seed2024 = write_metric(run_dir, "ConvE_full", 2024, 0.1)
    seed2025 = write_metric(run_dir, "ConvE_full", 2025, 0.2)
    write_timing(seed2024, training=100.0, inference=5.0, evaluation=1.0)
    write_timing(seed2025, training=90.0, inference=20.0, evaluation=2.0)
    baseline = write_metric(run_dir, "EB-CF", None, 0.25, traditional=True)
    incomplete_timing = baseline.parents[1] / "timing.json"
    incomplete_timing.write_text(
        json.dumps({"evaluation_metric": {"seconds": 1.5}}),
        encoding="utf-8",
    )

    rows, _ = collect_dataset_results("demo", run_dir)
    summary = {row["model"]: row for row in aggregate_results(rows)}

    assert summary["ConvE_full"]["max_time_seed"] == 2025
    assert summary["ConvE_full"]["max_training_seconds"] == pytest.approx(90.0)
    assert summary["ConvE_full"]["max_inference_seconds"] == pytest.approx(20.0)
    assert summary["ConvE_full"]["max_evaluation_seconds"] == pytest.approx(2.0)
    assert summary["ConvE_full"]["max_total_seconds"] == pytest.approx(112.0)
    assert summary["EB-CF"]["max_time_seed"] is None
    assert summary["EB-CF"]["max_total_seconds"] is None


def test_replacement_only_overrides_selected_model_and_seed(tmp_path):
    run_dir = tmp_path / "primary"
    replacement_dir = tmp_path / "replacement"
    for model in ["ConvE_full", "ConvE_no_seq"]:
        for seed in [2024, 2025]:
            write_metric(run_dir, model, seed, 0.1)
            write_metric(replacement_dir, model, seed, 0.9)

    rows, metadata = collect_dataset_results(
        "demo",
        run_dir,
        replacement_run_dir=replacement_dir,
        replacement_models=("ConvE_full",),
        replacement_seeds=(2024,),
    )
    keyed = {(row["model"], row["seed"]): row for row in rows}

    assert keyed[("ConvE_full", 2024)]["ACC@10"] == pytest.approx(0.901)
    assert keyed[("ConvE_full", 2024)]["source"] == "replacement"
    assert keyed[("ConvE_full", 2025)]["ACC@10"] == pytest.approx(0.101)
    assert keyed[("ConvE_no_seq", 2024)]["ACC@10"] == pytest.approx(0.101)
    assert metadata["replacement_count"] == 1


def test_missing_required_metric_is_rejected(tmp_path):
    run_dir = tmp_path / "run"
    metric_file = write_metric(run_dir, "ConvE_full", 2024, 0.1)
    payload = json.loads(metric_file.read_text(encoding="utf-8"))
    del payload["ACC"]["100"]
    metric_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ACC@100"):
        collect_dataset_results("demo", run_dir)


def test_incomplete_replacement_options_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="together"):
        validate_replacement_options(tmp_path, (), ())


def test_write_report_creates_complete_csv_json_and_markdown(tmp_path):
    run_dir = tmp_path / "run"
    for seed, value in zip(range(2024, 2029), [0.1, 0.2, 0.3, 0.4, 0.5]):
        write_metric(run_dir, "ConvE_full", seed, value)
    rows, metadata = collect_dataset_results("demo", run_dir)
    summary = aggregate_results(rows)
    output_dir = tmp_path / "output"

    written = write_report("demo", rows, summary, metadata, output_dir)

    assert {path.name for path in written} == {
        "per_seed_metrics.csv",
        "model_mean_std.csv",
        "dataset_metrics.json",
        "dataset_metrics.md",
        "paper_avg_table.csv",
        "paper_avg_table.md",
    }
    with (output_dir / "per_seed_metrics.csv").open(encoding="utf-8", newline="") as fp:
        headers = next(csv.reader(fp))
    assert "ACC@100" in headers
    assert "NOV@100" in headers
    assert "Ep_sim@10" in headers
    assert "ACC-Avg" in headers
    assert "NOV-Avg" in headers
    assert "Training Seconds" in headers
    assert "Total Seconds" in headers

    report = (output_dir / "dataset_metrics.md").read_text(encoding="utf-8")
    assert "ACC@100" in report
    assert "NOV@100" in report
    assert "Ep_sim@10" in report
    assert "mean ± sample std" in report
    assert "Per-seed raw means" in report
    assert "Maximum complete runtime" in report

    with (output_dir / "paper_avg_table.csv").open(
        encoding="utf-8-sig", newline=""
    ) as fp:
        paper_headers = next(csv.reader(fp))
    assert paper_headers == [
        "Model",
        "Runs",
        "ACC-Avg Mean",
        "ACC-Avg Std",
        "NOV-Avg Mean",
        "NOV-Avg Std",
        "Ep_sim@10 Mean",
        "Ep_sim@10 Std",
        "Max-Time Seed",
        "Training Seconds",
        "Inference Seconds",
        "Evaluation Seconds",
        "Total Seconds",
    ]
    paper_report = (output_dir / "paper_avg_table.md").read_text(encoding="utf-8")
    assert "ACC-Avg" in paper_report
    assert "NOV-Avg" in paper_report
    assert "0.05, 0.05, 0.05, 0.10, 0.15, 0.25, 0.35" in paper_report
