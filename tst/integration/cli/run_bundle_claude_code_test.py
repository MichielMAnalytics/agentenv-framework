"""``agent-env run`` on bundles whose tasks name no agent, so they deploy the agents bundle's Claude Code: its image
built from the shipped folder with real docker, the real CLI, and a fake Anthropic Messages endpoint as the configured
model endpoint, so no model key is needed. Needs a Docker daemon and network access for the image's downloads."""

import asyncio
import json
import logging

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent, protocol
from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_sandbox import _host_ips
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
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
    assert f"{IMAGE}: v1 (new)" in first.output and f"{AGENT}: v1 (new)" in first.output, first.output
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
        *_ask("Call the echo tool with the text ping-from-tool.", "ping-from-tool", env_ids=[ECHO_ENV]),
    ])
    root.joinpath("tasks/echo.json").write_text(json.dumps(
        [{**step, "depends_on": ["env"]} if step["id"] == "agent" else step
         for step in json.loads(root.joinpath("tasks/echo.json").read_text())]))

    result = _run(root)

    offered = sorted({tool["name"] for call in endpoint.messages() for tool in call.body.get("tools") or []})
    seen = (result.output, offered, endpoint.tool_results())
    assert result.exit_code == 0 and "tasks/echo.json v1: passed" in result.output, seen
    # The fake never repeats the prompt, so the text reached the answer through the tool.
    assert any(name.endswith("__echo") for name in offered), seen
    assert ["ping-from-tool" in result for result in endpoint.tool_results()] == [True], seen


async def _send_project_id(step, context):
    """What a host's runner does before each step to attribute an agent's spend to the run's project: post the
    run's project_id over agent-config to the agent a prompting step names, when its card takes it."""
    project_id = context.metadata.get("project_id")
    agent = next((a for a in context.deployed_agents if a.agent_name == getattr(step, "agent_name", None)), None)
    if step.type != "prompt_agent" or not project_id or agent is None:
        return
    negotiated = A2AAgent.negotiate_agent_config(agent.a2a_card or {}, {"project_id": project_id})
    assert negotiated and "project_id" in negotiated.fields, "the card doesn't take project_id over agent-config"
    await protocol.post_agent_config((agent.a2a_url or agent.api_url) + negotiated.endpoint, negotiated.fields)


def test_a_project_id_posted_over_agent_config_attributes_every_model_call_to_the_project(state, endpoint):
    endpoint.respond = answer("pong")
    # A bundle run writes the shipped agent and its image to this test's stores.
    assert _run(_bundle(state / "one", "ask", _ask("Say pong.", "pong"))).exit_code == 0
    before = len(endpoint.messages())
    context = TaskStepContext(metadata={"project_id": "project-attr-1", "task_id": "task-attr-1"})
    steps = [DeployAgentTaskStep(id="agent", version=None, sandbox_type="local"),
             PromptAgentTaskStep(id="ask", version=None, prompt_id="ask", prompt="Say pong.")]

    async def run():
        try:
            for step in steps:
                await _send_project_id(step, context)
                await step.execute(context)
        finally:
            await teardown_run(context)

    with namespace_routing():
        asyncio.run(run())

    calls = endpoint.messages()[before:]
    assert calls, "the CLI never called the endpoint"
    assert {call.headers.get("x-litellm-customer-id") for call in calls} == {"project-attr-1"}
    assert {call.headers.get("x-litellm-tags") for call in calls} == {"projectId:project-attr-1,taskId:task-attr-1"}
