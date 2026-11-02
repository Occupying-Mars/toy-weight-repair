"""Fast, synthetic weight-repair test with prospective behavioral predictions.

Operator builds exact circuits; editor sees only blinded weights and practice examples.
No training, model downloads, editor execution, or evaluation feedback is required.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from toy_weight_repair.harness import ROOT, ask, hash_file, write_json


def forward(w, examples):
    x = np.zeros((len(examples), 17))
    for i, (first, last, mode) in enumerate(examples):
        x[i, first] = 1
        x[i, 8 + last] = 1
        x[i, 16] = mode
    hidden = np.maximum(x @ w["W1"].T + w["b1"], 0)
    return (hidden @ w["W2"].T + w["b2"]).argmax(axis=1)


def fixture(seed, damaged):
    rng = np.random.default_rng(seed)
    w1 = np.zeros((24, 17))
    b1 = np.zeros(24)
    w2 = np.zeros((8, 24))
    for symbol in range(8):
        w1[symbol, symbol] = 1
        w1[symbol, 16] = -1
        w1[8 + symbol, 8 + symbol] = 1
        w1[8 + symbol, 16] = 1
        b1[8 + symbol] = -1
        w2[symbol, symbol] = w2[symbol, 8 + symbol] = 1
    # Inactive distractor neurons remain inactive on every legal input.
    w1[16:] = rng.uniform(-0.1, 0.1, (8, 17))
    b1[16:] = -3
    w2[:, 16:] = rng.uniform(-1, 1, (8, 8))
    clean = {"W1": w1.copy(), "b1": b1.copy(), "W2": w2.copy(), "b2": np.zeros(8)}
    mode = int(rng.integers(2))
    symbols = sorted(int(v) for v in rng.choice(8, size=3, replace=False))
    if damaged:
        for symbol, wrong in zip(symbols, symbols[1:] + symbols[:1], strict=True):
            neuron = mode * 8 + symbol
            w2[symbol, neuron] = 0
            w2[wrong, neuron] = 1
    # Function-preserving positive rescaling and permutation hide neuron indices.
    scales = rng.choice([0.5, 1.0, 2.0, 4.0], 24)
    permutation = rng.permutation(24)

    def disguise(weights):
        return {
            "W1": (weights["W1"] * scales[:, None])[permutation],
            "b1": (weights["b1"] * scales)[permutation],
            "W2": (weights["W2"] / scales[None, :])[:, permutation],
            "b2": weights["b2"].copy(),
        }

    corrupt = {"W1": w1, "b1": b1, "W2": w2, "b2": np.zeros(8)}
    affected = [[mode, symbol] for symbol in symbols] if damaged else []
    return disguise(corrupt), disguise(clean), affected


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261103)
    truth = []
    cases = []
    # Identity assignment is committed locally but excluded from editor prompts.
    assignments = [(901, True), (317, False), (1229, True), (447, True)]
    rng.shuffle(assignments)
    all_examples = [(a, b, m) for a in range(8) for b in range(8) for m in range(2)]
    for index, (seed, damaged) in enumerate(assignments):
        case_id = f"case-{index + 1}"
        weights, clean, affected = fixture(seed, damaged)
        # One example for each (mode, source symbol); 112 other combinations held out.
        practice = [(s, (s + 3) % 8, 0) for s in range(8)]
        practice += [((s + 5) % 8, s, 1) for s in range(8)]
        heldout = [e for e in all_examples if e not in practice]

        def labels(examples):
            return [a if m == 0 else b for a, b, m in examples]

        assert np.array_equal(forward(clean, all_examples), labels(all_examples))
        baseline = forward(weights, practice)
        cases.append(
            {
                "id": case_id,
                "weights": {name: value.tolist() for name, value in weights.items()},
                "practice": [
                    {"input": list(e), "target": y, "prediction": int(pred)}
                    for e, y, pred in zip(practice, labels(practice), baseline, strict=True)
                ],
            }
        )
        truth.append(
            {
                "id": case_id,
                "seed": seed,
                "damaged": damaged,
                "affected": affected,
                "clean_weights": {name: value.tolist() for name, value in clean.items()},
                "heldout": heldout,
            }
        )
    write_json(run / "operator-truth.json", truth)
    write_json(run / "editor-input.json", cases)
    prompt = """Repair tiny neural networks by interpreting their weights. No tools or training.
Task: input (first,last,mode), symbols 0..7. Output first when mode=0, last when mode=1.
Input vector x has length 17: one-hot(first) at 0..7, one-hot(last) at 8..15,
and mode at index 16. h=relu(W1@x+b1); logits=W2@h+b2; prediction=argmax(logits).
Each model has 632 parameters (24 hidden neurons, 8 outputs). Neuron order and positive
scales are arbitrary. Some models may already be correct. You get weights and 16 practice
examples per model; 112 other input combinations per model are hidden for final scoring.
Do not assume every model needs edits. Diagnose each independently. One submission only;
there is no execution feedback. Change only individual W2 entries, at most 24 per model.
Do not fit weights to examples, search, or train: derive deliberate repairs from circuits.
Before evaluation, predict EXACTLY which behavioral slices change: each slice [mode,symbol]
means every legal input with that mode and selected source symbol. The prediction concerns
output labels (not logits). Leave an already correct model unchanged.
Return a single JSON object, no prose outside JSON:
{"cases":[{"id":"case-1","mechanism":"your causal explanation",
"predicted_changed_slices":[[0,2]],"predicted_heldout_accuracy":1.0,
"edits":[{"row":2,"col":4,"value":0.5}]} ...]}
Edits are replacement values, not increments. Include all four case ids. State honest
mechanistic predictions; operator will compare predictions with exhaustive behavior.
Models and practice observations follow:
""" + json.dumps(cases, separators=(",", ":"))
    (run / "editor-prompt.txt").write_text(prompt)
    write_json(
        run / "protocol.json",
        {
            "task": "conditional-symbol-copy",
            "models": 4,
            "parameters_each": 632,
            "practice_each": 16,
            "heldout_each": 112,
            "max_edits_each": 24,
            "attempts": 1,
            "training": False,
            "synthetic_constructed_circuits": True,
            "truth_sha256": hash_file(run / "operator-truth.json"),
            "input_sha256": hash_file(run / "editor-input.json"),
            "source_sha256": hash_file(Path(__file__)),
        },
    )


def score(run):
    text = (run / "astra/answer.md").read_text().strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    submission = json.loads(text)
    cases = json.loads((run / "editor-input.json").read_text())
    truths = json.loads((run / "operator-truth.json").read_text())
    answers = submission["cases"]
    assert len(answers) == 4 and {a["id"] for a in answers} == {c["id"] for c in cases}
    results = []
    for case, truth in zip(cases, truths, strict=True):
        answer = next(a for a in answers if a["id"] == case["id"])
        weights = {name: np.array(value) for name, value in case["weights"].items()}
        edited = {name: value.copy() for name, value in weights.items()}
        edits = answer["edits"]
        assert len(edits) <= 24
        seen = set()
        for edit in edits:
            row, col, value = edit["row"], edit["col"], edit["value"]
            assert type(row) is int and 0 <= row < 8
            assert type(col) is int and 0 <= col < 24
            assert np.isfinite(value) and abs(value) <= 100
            assert (row, col) not in seen
            seen.add((row, col))
            edited["W2"][row, col] = value
        hidden = truth["heldout"]
        labels = np.array([a if m == 0 else b for a, b, m in hidden])
        before, after = forward(weights, hidden), forward(edited, hidden)
        changed = []
        slice_consistent = True
        for mode in range(2):
            for symbol in range(8):
                examples = [
                    (symbol, other, mode) if mode == 0 else (other, symbol, mode)
                    for other in range(8)
                ]
                mask = forward(weights, examples) != forward(edited, examples)
                if mask.any():
                    changed.append([mode, symbol])
                    slice_consistent &= bool(mask.all())
        predicted = sorted(answer["predicted_changed_slices"])
        exact_prediction = predicted == changed and slice_consistent
        control = {name: np.array(value) for name, value in truth["clean_weights"].items()}
        result = {
            "id": case["id"],
            "damaged": truth["damaged"],
            "heldout_n": len(hidden),
            "baseline_correct": int((before == labels).sum()),
            "edited_correct": int((after == labels).sum()),
            "clean_control_correct": int((forward(control, hidden) == labels).sum()),
            "regressions": int(((before == labels) & (after != labels)).sum()),
            "edits": len(edits),
            "changed_parameters": int((weights["W2"] != edited["W2"]).sum()),
            "predicted_changed_slices": predicted,
            "actual_changed_slices": changed,
            "exact_causal_prediction": bool(exact_prediction),
            "predicted_accuracy": answer["predicted_heldout_accuracy"],
            "mechanism": answer["mechanism"],
        }
        result["pass"] = bool(
            result["edited_correct"] == len(hidden)
            and result["regressions"] == 0
            and exact_prediction
            and (truth["damaged"] or result["changed_parameters"] == 0)
        )
        results.append(result)
    write_json(
        run / "results.json", {"all_pass": all(r["pass"] for r in results), "cases": results}
    )
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "score"])
    parser.add_argument("--name", default="toy-symbol-repair-001")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        parser.error("invalid run name")
    run = ROOT / "runs" / args.name
    if args.action == "prepare":
        prepare(run)
        print(run)
    elif args.action == "run":
        ask(run / "astra", (run / "editor-prompt.txt").read_text(), timeout=600)
        score(run)
    else:
        score(run)


if __name__ == "__main__":
    main()
