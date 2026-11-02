"""Minimal standalone Codex launcher; offline replay does not call this service."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_for(directory, effort):
    binary = os.environ.get("WSP_CODEX_BIN", "codex")
    command = [
        binary,
        "--no-daemon",
        "exec",
        "--ignore-user-config",
        "--json",
        "--skip-git-repo-check",
        "--cd",
        str(directory),
        "--sandbox",
        "read-only",
        "--model",
        "gpt-6-astra",
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-c",
        f"model_instructions_file={json.dumps(str(directory / 'neutral.md'))}",
        "-c",
        "model_context_window=1050000",
        "-c",
        "model_auto_compact_token_limit=900000",
    ]
    for feature in (
        "shell_tool",
        "unified_exec",
        "multi_agent",
        "apps",
        "plugins",
        "code_mode",
        "browser_use",
        "computer_use",
        "image_generation",
        "in_app_browser",
        "browser_use_external",
        "hooks",
        "memories",
        "tool_suggest",
        "goals",
        "workspace_dependencies",
        "skill_mcp_dependency_install",
    ):
        command += ["--disable", feature]
    return command + ["--output-last-message", str(directory / "answer.md"), "-"]


def ask(directory, prompt, effort="high", timeout=600):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "prompt.txt").write_text(prompt)
    (directory / "neutral.md").write_text("You are a helpful assistant.\n")
    command = command_for(directory, effort)
    write_json(directory / "command.json", command)
    started = time.monotonic()
    with (directory / "events.jsonl").open("w") as out, (directory / "stderr.log").open("w") as err:
        process = subprocess.Popen(
            command,
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(prompt, timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            write_json(directory / "receipt.json", {"status": "timeout"})
            raise RuntimeError("Astra turn interrupted or timed out") from None
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    allowed = {"agent_message", "reasoning", "error", "plan", "todo_list"}
    unexpected = [
        e["item"]["type"]
        for e in events
        if e.get("type") == "item.completed" and e.get("item", {}).get("type") not in allowed
    ]
    answer_path = directory / "answer.md"
    answer = answer_path.read_text() if answer_path.exists() else ""
    source = "output-last-message"
    if not answer.strip():
        answer = "\n\n".join(
            e["item"].get("text", "")
            for e in events
            if e.get("type") == "item.completed"
            and e.get("item", {}).get("type") == "agent_message"
        )
        answer_path.write_text(answer)
        source = "nonempty agent-message events (empty final-message file)"
    failed = process.returncode != 0 or any(e.get("type") == "turn.failed" for e in events)
    write_json(
        directory / "receipt.json",
        {
            "model": "gpt-6-astra",
            "effort": effort,
            "answer_source": source,
            "tool_free": not unexpected,
            "unexpected_tools": unexpected,
            "usage": next(
                (e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"),
                None,
            ),
            "seconds": time.monotonic() - started,
            "returncode": process.returncode,
            "prompt_sha256": hash_file(directory / "prompt.txt"),
            "answer_sha256": hash_file(answer_path),
            "status": "error" if failed else "complete",
        },
    )
    if failed or unexpected or not answer.strip():
        raise RuntimeError(f"Invalid Astra turn; inspect {directory / 'receipt.json'}")
    return answer
