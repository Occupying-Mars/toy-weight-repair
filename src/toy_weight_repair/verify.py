"""Verify published evidence and replay scoring offline without changing originals."""

import json
import shutil
import tempfile
from pathlib import Path

from toy_weight_repair.experiment import prepare, score
from toy_weight_repair.harness import ROOT, hash_file


def main():
    evidence = ROOT / "evidence/toy-symbol-repair-001"
    manifest = json.loads((ROOT / "provenance/publication-manifest.json").read_text())
    for item in manifest["files"]:
        assert hash_file(evidence / item["path"]) == item["public_sha256"], item["path"]
    original = ROOT / "provenance/original-toy-repair.py"
    assert hash_file(original) == manifest["original_source_sha256"]
    protocol = json.loads((evidence / "protocol.json").read_text())
    assert protocol["source_sha256"] == hash_file(original)
    assert protocol["truth_sha256"] == hash_file(evidence / "operator-truth.json")
    assert protocol["input_sha256"] == hash_file(evidence / "editor-input.json")
    receipt = json.loads((evidence / "astra/receipt.json").read_text())
    assert receipt["prompt_sha256"] == hash_file(evidence / "editor-prompt.txt")
    assert receipt["answer_sha256"] == hash_file(evidence / "astra/answer.md")
    assert receipt["status"] == "complete" and receipt["tool_free"]
    assert not receipt["unexpected_tools"] and receipt["returncode"] == 0
    events = [
        json.loads(line) for line in (evidence / "astra/events.jsonl").read_text().splitlines()
    ]
    allowed = {"agent_message", "reasoning", "error", "plan", "todo_list"}
    assert all(e["item"]["type"] in allowed for e in events if e.get("type") == "item.completed")
    messages = [
        e["item"].get("text", "")
        for e in events
        if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"
    ]
    assert (evidence / "astra/answer.md").read_text().strip() in [m.strip() for m in messages]
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary) / "generated"
        prepare(generated)
        for name in ("editor-input.json", "operator-truth.json", "editor-prompt.txt"):
            assert (generated / name).read_bytes() == (evidence / name).read_bytes(), name
        replay = Path(temporary) / "replay"
        shutil.copytree(evidence, replay)
        score(replay)
        assert json.loads((replay / "results.json").read_text()) == json.loads(
            (evidence / "results.json").read_text()
        )
    print("verified: evidence hashes, regenerated fixtures, recorded answer, and identical scores")


if __name__ == "__main__":
    main()
