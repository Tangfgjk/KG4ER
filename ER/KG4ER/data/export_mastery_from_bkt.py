import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_limit)
        break
    except OverflowError:
        field_limit = int(field_limit / 10)


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description="Train BKT per concept and export stu2know_mastery.json.")
    parser.add_argument("--data-dir", required=True, help="ER prepared_for_kt directory.")
    parser.add_argument("--train-file", default="train_sequences.csv")
    parser.add_argument("--test-file", default="test_sequences.csv")
    parser.add_argument("--q-file", default="Q.txt")
    parser.add_argument("--output-file", default="stu2know_mastery.json")
    parser.add_argument("--manifest-file", default="bkt_mastery_manifest.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--init-prior", type=float, default=0.2)
    parser.add_argument("--init-learn", type=float, default=0.15)
    parser.add_argument("--init-guess", type=float, default=0.2)
    parser.add_argument("--init-slip", type=float, default=0.1)
    parser.add_argument("--min-prob", type=float, default=0.01)
    parser.add_argument("--max-slip-guess", type=float, default=0.49)
    parser.add_argument("--precision", type=int, default=4)
    return parser.parse_args()


def clamp(value, lower, upper):
    return min(max(value, lower), upper)


def parse_tokens(value):
    if value is None:
        return []
    return [token for token in str(value).split(",") if token not in ("", "-1")]


def read_q_shape(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line.split(","))
    if not rows:
        raise ValueError(f"Empty Q file: {path}")
    width = len(rows[0])
    for idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"Q row {idx} has width {len(row)}, expected {width}")
    return len(rows), width


def read_concept_sequences(path, concept_count):
    sequences = defaultdict(list)
    source_students = 0
    interactions = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_students += 1
            concepts = parse_tokens(row.get("concepts"))
            responses = parse_tokens(row.get("responses"))
            selectmasks = parse_tokens(row.get("selectmasks")) or ["1"] * min(len(concepts), len(responses))
            usable = min(len(concepts), len(responses), len(selectmasks))
            per_student = defaultdict(list)
            for idx in range(usable):
                if selectmasks[idx] == "0":
                    continue
                response = int(responses[idx])
                if response not in (0, 1):
                    continue
                for concept_token in concepts[idx].split("_"):
                    if concept_token in ("", "-1"):
                        continue
                    concept = int(concept_token)
                    if concept < 0 or concept >= concept_count:
                        raise ValueError(f"Concept id {concept} in {path} is outside [0, {concept_count})")
                    per_student[concept].append(response)
                    interactions += 1
            for concept, obs in per_student.items():
                if obs:
                    sequences[concept].append(obs)
    return sequences, {"source_students": source_students, "interactions": interactions}


def sequence_forward_backward(obs, prior, learn, guess, slip):
    trans = ((1.0 - learn, learn), (0.0, 1.0))
    alpha = []
    scales = []
    for idx, response in enumerate(obs):
        emit = (guess if response == 1 else 1.0 - guess, 1.0 - slip if response == 1 else slip)
        if idx == 0:
            raw = [(1.0 - prior) * emit[0], prior * emit[1]]
        else:
            prev = alpha[-1]
            raw = [
                (prev[0] * trans[0][0] + prev[1] * trans[1][0]) * emit[0],
                (prev[0] * trans[0][1] + prev[1] * trans[1][1]) * emit[1],
            ]
        scale = raw[0] + raw[1] + EPS
        alpha.append([raw[0] / scale, raw[1] / scale])
        scales.append(scale)

    beta = [[1.0, 1.0] for _ in obs]
    for idx in range(len(obs) - 2, -1, -1):
        next_response = obs[idx + 1]
        next_emit = (
            guess if next_response == 1 else 1.0 - guess,
            1.0 - slip if next_response == 1 else slip,
        )
        beta[idx][0] = (
            trans[0][0] * next_emit[0] * beta[idx + 1][0]
            + trans[0][1] * next_emit[1] * beta[idx + 1][1]
        ) / (scales[idx + 1] + EPS)
        beta[idx][1] = (
            trans[1][0] * next_emit[0] * beta[idx + 1][0]
            + trans[1][1] * next_emit[1] * beta[idx + 1][1]
        ) / (scales[idx + 1] + EPS)

    gamma = []
    xi_sum = 0.0
    gamma0_to_any = 0.0
    for idx in range(len(obs)):
        denom = alpha[idx][0] * beta[idx][0] + alpha[idx][1] * beta[idx][1] + EPS
        gamma.append([
            alpha[idx][0] * beta[idx][0] / denom,
            alpha[idx][1] * beta[idx][1] / denom,
        ])
        if idx < len(obs) - 1:
            next_response = obs[idx + 1]
            next_emit = (
                guess if next_response == 1 else 1.0 - guess,
                1.0 - slip if next_response == 1 else slip,
            )
            denom_xi = 0.0
            xi01 = alpha[idx][0] * trans[0][1] * next_emit[1] * beta[idx + 1][1]
            for s0 in (0, 1):
                for s1 in (0, 1):
                    denom_xi += alpha[idx][s0] * trans[s0][s1] * next_emit[s1] * beta[idx + 1][s1]
            xi_sum += xi01 / (denom_xi + EPS)
            gamma0_to_any += gamma[idx][0]
    log_likelihood = sum(math.log(scale + EPS) for scale in scales)
    return gamma, xi_sum, gamma0_to_any, log_likelihood


def train_bkt_for_concept(sequences, defaults, epochs, min_prob, max_slip_guess, min_observations):
    observations = sum(len(seq) for seq in sequences)
    correct = sum(sum(seq) for seq in sequences)
    if observations < min_observations:
        empirical = correct / observations if observations else 0.0
        return {
            "prior": clamp(empirical, min_prob, 1.0 - min_prob),
            "learn": defaults["learn"],
            "guess": defaults["guess"],
            "slip": defaults["slip"],
            "observations": observations,
            "used_fallback": True,
            "log_likelihood": None,
        }

    prior = defaults["prior"]
    learn = defaults["learn"]
    guess = defaults["guess"]
    slip = defaults["slip"]
    last_ll = None
    for _ in range(epochs):
        init_mastered = 0.0
        init_total = 0.0
        learn_num = 0.0
        learn_den = 0.0
        unmastered_correct = 0.0
        unmastered_total = 0.0
        mastered_wrong = 0.0
        mastered_total = 0.0
        total_ll = 0.0
        for obs in sequences:
            gamma, xi_sum, gamma0_to_any, ll = sequence_forward_backward(obs, prior, learn, guess, slip)
            total_ll += ll
            init_mastered += gamma[0][1]
            init_total += 1.0
            learn_num += xi_sum
            learn_den += gamma0_to_any
            for response, state_prob in zip(obs, gamma):
                unmastered_total += state_prob[0]
                mastered_total += state_prob[1]
                if response == 1:
                    unmastered_correct += state_prob[0]
                else:
                    mastered_wrong += state_prob[1]
        prior = clamp(init_mastered / (init_total + EPS), min_prob, 1.0 - min_prob)
        learn = clamp(learn_num / (learn_den + EPS), min_prob, 1.0 - min_prob)
        guess = clamp(unmastered_correct / (unmastered_total + EPS), min_prob, max_slip_guess)
        slip = clamp(mastered_wrong / (mastered_total + EPS), min_prob, max_slip_guess)
        last_ll = total_ll

    return {
        "prior": prior,
        "learn": learn,
        "guess": guess,
        "slip": slip,
        "observations": observations,
        "used_fallback": False,
        "log_likelihood": last_ll,
    }


def update_mastery(prob, response, learn, guess, slip):
    p_correct = prob * (1.0 - slip) + (1.0 - prob) * guess
    if response == 1:
        posterior = prob * (1.0 - slip) / (p_correct + EPS)
    else:
        posterior = prob * slip / (1.0 - p_correct + EPS)
    return posterior + (1.0 - posterior) * learn


def export_test_mastery(test_path, params, concept_count, precision):
    rows = []
    source_students = 0
    with test_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_students += 1
            mastery = [params[idx]["prior"] for idx in range(concept_count)]
            concepts = parse_tokens(row.get("concepts"))
            responses = parse_tokens(row.get("responses"))
            selectmasks = parse_tokens(row.get("selectmasks")) or ["1"] * min(len(concepts), len(responses))
            usable = min(len(concepts), len(responses), len(selectmasks))
            for idx in range(usable):
                if selectmasks[idx] == "0":
                    continue
                response = int(responses[idx])
                if response not in (0, 1):
                    continue
                for concept_token in concepts[idx].split("_"):
                    if concept_token in ("", "-1"):
                        continue
                    concept = int(concept_token)
                    if concept < 0 or concept >= concept_count:
                        raise ValueError(f"Concept id {concept} in {test_path} is outside [0, {concept_count})")
                    p = params[concept]
                    mastery[concept] = update_mastery(mastery[concept], response, p["learn"], p["guess"], p["slip"])
            rows.append([round(float(value), precision) for value in mastery])
    return rows, source_students


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    question_count, concept_count = read_q_shape(data_dir / args.q_file)
    train_sequences, train_summary = read_concept_sequences(data_dir / args.train_file, concept_count)
    defaults = {
        "prior": args.init_prior,
        "learn": args.init_learn,
        "guess": args.init_guess,
        "slip": args.init_slip,
    }
    params = []
    for concept in range(concept_count):
        params.append(
            train_bkt_for_concept(
                train_sequences.get(concept, []),
                defaults,
                args.epochs,
                args.min_prob,
                args.max_slip_guess,
                args.min_observations,
            )
        )

    mastery_rows, test_students = export_test_mastery(data_dir / args.test_file, params, concept_count, args.precision)
    output_path = data_dir / args.output_file
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mastery_rows, f)

    manifest = {
        "status": "exported",
        "method": "per-concept BKT EM",
        "data_dir": str(data_dir),
        "train_file": args.train_file,
        "test_file": args.test_file,
        "q_file": args.q_file,
        "output_file": args.output_file,
        "question_count": question_count,
        "concept_count": concept_count,
        "test_students": test_students,
        "shape": [len(mastery_rows), concept_count],
        "epochs": args.epochs,
        "min_observations": args.min_observations,
        "train_summary": train_summary,
        "fallback_concepts": sum(1 for item in params if item["used_fallback"]),
        "parameters": params,
    }
    manifest_path = data_dir / args.manifest_file
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: manifest[k] for k in ["status", "method", "output_file", "shape", "fallback_concepts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
