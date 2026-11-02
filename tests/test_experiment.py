import numpy as np
import pytest

from toy_weight_repair.experiment import fixture, forward, prepare, score
from toy_weight_repair.harness import write_json


@pytest.mark.parametrize("seed", [901, 317, 1229, 447])
@pytest.mark.parametrize("damaged", [False, True])
def test_fixture_has_exact_planted_behavior(seed, damaged):
    weights, clean, affected = fixture(seed, damaged)
    examples = [(a, b, m) for a in range(8) for b in range(8) for m in range(2)]
    labels = np.array([a if m == 0 else b for a, b, m in examples])
    expected_errors = [[m, a if m == 0 else b] in affected for a, b, m in examples]
    assert sum(value.size for value in weights.values()) == 632
    assert np.array_equal(forward(clean, examples), labels)
    assert np.array_equal(forward(weights, examples) != labels, expected_errors)
    assert sum(expected_errors) == (24 if damaged else 0)


def test_scorer_accepts_control_and_rejects_wrong_prediction(tmp_path):
    import json

    run = tmp_path / "toy"
    prepare(run)
    inputs = json.loads((run / "editor-input.json").read_text())
    truths = json.loads((run / "operator-truth.json").read_text())
    submissions = []
    for case, truth in zip(inputs, truths, strict=True):
        before, after = np.array(case["weights"]["W2"]), np.array(truth["clean_weights"]["W2"])
        edits = [
            {"row": int(row), "col": int(col), "value": float(after[row, col])}
            for row, col in zip(*np.where(before != after), strict=True)
        ]
        submissions.append(
            {
                "id": case["id"],
                "mechanism": "operator positive-control replay",
                "predicted_changed_slices": truth["affected"],
                "predicted_heldout_accuracy": 1.0,
                "edits": edits,
            }
        )
    (run / "astra").mkdir()
    write_json(run / "astra/answer.md", {"cases": submissions})
    score(run)
    assert json.loads((run / "results.json").read_text())["all_pass"]
    damaged = next(i for i, t in enumerate(truths) if t["damaged"])
    submissions[damaged]["predicted_changed_slices"] = []
    write_json(run / "astra/answer.md", {"cases": submissions})
    score(run)
    assert not json.loads((run / "results.json").read_text())["all_pass"]
