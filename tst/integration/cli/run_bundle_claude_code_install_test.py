"""``install_agent`` naming no agent installs the agents bundle's Claude Code, from its folder, into a running task
container, a Debian and a musl one with no Python of their own, or onto the sandbox's host. A skill reaches it through
a read grant, over the local transfer CA's TLS, and the agent answers through a fake Anthropic Messages endpoint after
running a command where the task's files are. Needs a Docker daemon, and network access for Claude Code, uv, Python and
the agent's requirements."""

import json
import logging
import platform
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_sandbox import _host_ips
from agent_env.store.object_store.local.grant_server import grant_server
from tst.util.fake_anthropic import FakeAnthropic, tool_then_answer

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

TASK_IMAGES = Path(__file__).resolve().parents[2] / "data" / "claude_code_install"
NOTES = "notes-from-the-task"
SKILL_BODY = "Read the task's notes first."
SKILL_MD = f"---\nname: notes\ndescription: Where the task's notes are.\n---\n{SKILL_BODY}\n"
# The installed skill's last line, as the CLI's shell finds it.
READ_SKILL = 'tail -n 1 "$CLAUDE_CONFIG_DIR/skills/notes/SKILL.md"'


@pytest.fixture
def endpoint():
    """The fake endpoint, where this machine's containers reach it: on Linux, the Docker bridge address."""
    with FakeAnthropic(host=_host_ips()[-1]) as fake:
        yield fake


@pytest.fixture
def state(monkeypatch, tmp_path, endpoint):
    """Local stores under this test's folder and [model] pointed at the fake endpoint. Whatever ran, no sandbox work
    folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    config = tmp_path / "config.toml"
    config.write_text(f'[model]\nbase_url = "{endpoint.url}"\napi_key = "fake-key"\ndefault = "claude-fake-1"\n')
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


def _bundle(root: Path, task_image: str | None) -> Path:
    """A task that installs the default agent, into a container built from ``task_image`` or onto the host, gives it
    a skill, and asks it to read the task's notes."""
    steps = [{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm",
              "sandbox_type": "local", "exposed_ports": [8000]}]
    install = {"id": "coder", "type": "install_agent", "sandbox_name": "box", "agent_name": "coder"}
    if task_image is not None:
        shutil.copytree(TASK_IMAGES / task_image, root / "artifacts" / "task-image")
        steps.append({"id": "repo", "type": "run_docker_container", "sandbox_name": "box", "container_name": "repo",
                      "docker_context_artifact_id": "task-image", "ports": [8000], "depends_on": ["box"]})
        install |= {"container_name": "repo", "depends_on": ["repo"]}
    else:
        install["depends_on"] = ["box"]
    (root / "skills/notes").mkdir(parents=True)
    (root / "skills/notes/SKILL.md").write_text(SKILL_MD)
    steps += [
        install,
        {"id": "skill", "type": "add_skills", "agent_name": "coder", "skills": [{"skill_artifact_id": "notes"}],
         "depends_on": ["coder"]},
        {"id": "ask", "type": "prompt_agent", "agent_name": "coder", "prompt_id": "ask", "depends_on": ["skill"],
         "prompt": "What do the task's notes say?"},
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "read",
         "criteria": [{"type": "response_contains", "needles": [NOTES]}], "depends_on": ["ask"]},
        {"id": "down", "type": "teardown_sandboxes", "sandbox_names": ["box"], "depends_on": ["check"]},
    ]
    (root / "tasks").mkdir()
    (root / "tasks/install.json").write_text(json.dumps(steps))
    return root


@pytest.mark.parametrize("task_image", ["debian", "alpine"])
def test_install_agent_naming_no_agent_installs_claude_code_into_a_task_container(state, endpoint, task_image):
    endpoint.respond = tool_then_answer("Bash", {"command": f"cat notes.txt; pwd; {READ_SKILL}"})

    result = CliRunner().invoke(cli, ["run", str(_bundle(state / "bundle", task_image))])

    seen = (result.output, endpoint.tool_results())
    assert result.exit_code == 0 and "tasks/install.json v1: passed" in result.output, seen
    # The CLI ran where the task container works, read the task's own file there, and the skill's.
    assert endpoint.tool_results() == [f"{NOTES}\n/workspace/task_repo\n{SKILL_BODY}"], seen


@pytest.mark.skipif(platform.system() != "Linux", reason="a host install needs a Linux host")
def test_install_agent_with_no_container_installs_claude_code_onto_the_sandboxs_host(monkeypatch, state, endpoint):
    # A process on the host can't resolve host.docker.internal: the grants name the bridge address, which it reaches.
    bridge = _host_ips()[-1]
    config = state / "config.toml"
    config.write_text(config.read_text() + (
        '[stores.object]\nimpl = "agent_env.store.object_store:LocalFilesystemObjectStore"\n'
        f'config = {{ grant_bind_host = "{bridge}", grant_advertise_host = "{bridge}" }}\n'))
    monkeypatch.delenv("AGENT_ENV_OBJECT_STORE")
    configure()
    reset_artifact_store()
    endpoint.respond = tool_then_answer("Bash", {"command": f"echo {NOTES}; {READ_SKILL}"})

    try:
        result = CliRunner().invoke(cli, ["run", str(_bundle(state / "bundle", None))])
    finally:
        grant_server(bridge, bridge).close()

    seen = (result.output, endpoint.tool_results())
    assert result.exit_code == 0 and "tasks/install.json v1: passed" in result.output, seen
    assert endpoint.tool_results() == [f"{NOTES}\n{SKILL_BODY}"], seen
