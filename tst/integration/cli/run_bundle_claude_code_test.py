"""``agent-env run`` on bundles whose tasks name no agent, so they deploy the agents bundle's Claude Code: its image
built from the shipped folder with real docker, the real CLI, and a fake Anthropic Messages endpoint as the configured
model endpoint, so no model key is needed. Needs a Docker daemon and network access for the image's downloads."""

import json
import logging

import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_sandbox import _host_ips
from agent_env.store.object_store.local.grant_server import grant_server
from tst.util.fake_anthropic import FakeAnthropic, answer, tool_then_answer

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

AGENT = "agentenv-framework/agents:agents/claude-code"
IMAGE = f"{AGENT} (Dockerfile image)"
MODEL = "claude-fake-1"
ECHO_ENV = "@local/agentenv-framework/echo-mcp/echo"


@pytest.fixture
def endpoint():
    """The fake endpoint, where this machine's containers reach it: on Linux, the Docker bridge address."""
    with FakeAnthropic(host=_host_ips()[-1]) as fake:
        yield fake


@pytest.fixture
def state(monkeypatch, tmp_path, endpoint):
    """Local stores under this test's folder, and [model] pointed at the fake endpoint with a header of its own.
    Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    config = tmp_path / "config.toml"
    config.write_text(f'[model]\nbase_url = "{endpoint.url}"\napi_key = "fake-key"\ndefault = "{MODEL}"\n\n'
                      '[model.headers]\nX-Agentenv-Test = "header-check"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    for name in ("LITELLM_API_KEY", "LITELLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def _bundle(root, name, steps):
    (root / "tasks").mkdir(parents=True)
    (root / f"tasks/{name}.json").write_text(json.dumps(steps))
    return root


def _ask(prompt, needle, env_ids=()):
    return [
        {"id": "agent", "type": "deploy_agent", "env_ids": list(env_ids), "sandbox_type": "local"},
        {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": prompt, "depends_on": ["agent"]},
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "answer",
         "criteria": [{"type": "response_contains", "needles": [needle]}], "depends_on": ["ask"]},
    ]


def _run(root):
    return CliRunner().invoke(cli, ["run", str(root)])


def test_a_task_naming_no_agent_runs_the_shipped_claude_code_against_the_configured_endpoint(state, endpoint):
    endpoint.respond = answer("pong")
    first = _run(_bundle(state / "one", "ask", _ask("Say pong.", "pong")))

    assert first.exit_code == 0, first.output
    assert f"{IMAGE}: building with docker" in first.output, first.output
    assert f"{AGENT}: v1 (new)" in first.output, first.output
    assert "tasks/ask.json v1: passed" in first.output, first.output
    calls = endpoint.messages()
    assert calls, "the CLI never called the endpoint"
    assert {call.headers["authorization"] for call in calls} == {"Bearer fake-key"}
    assert {call.headers.get("x-agentenv-test") for call in calls} == {"header-check"}
    assert MODEL in {call.body.get("model") for call in calls}

    second = _run(_bundle(state / "two", "ask", _ask("Say pong.", "pong")))

    assert second.exit_code == 0, second.output
    assert "building with docker" not in second.output, second.output
    assert f"{IMAGE}: v1, unchanged" in second.output and f"{AGENT}: v1, unchanged" in second.output, second.output
    assert "tasks/ask.json v1: passed" in second.output, second.output


def test_the_shipped_claude_code_calls_the_tools_of_the_env_it_is_deployed_with(state, endpoint):
    endpoint.respond = tool_then_answer("__echo", {"text": "ping-from-tool"}, prefix="the tool said: ")
    root = _bundle(state / "tools", "echo", [
        {"id": "env", "type": "deploy_env", "env_id": ECHO_ENV, "sandbox_type": "local"},
        *_ask("Call the echo tool with the text ping-from-tool.", "the tool said: ping-from-tool", env_ids=[ECHO_ENV]),
    ])
    root.joinpath("tasks/echo.json").write_text(json.dumps(
        [{**step, "depends_on": ["env"]} if step["id"] == "agent" else step
         for step in json.loads(root.joinpath("tasks/echo.json").read_text())]))

    result = _run(root)

    assert result.exit_code == 0, result.output
    assert "tasks/echo.json v1: passed" in result.output, result.output
    offered = {tool["name"] for call in endpoint.messages() for tool in call.body.get("tools") or []}
    assert any(name.endswith("__echo") for name in offered), sorted(offered)
