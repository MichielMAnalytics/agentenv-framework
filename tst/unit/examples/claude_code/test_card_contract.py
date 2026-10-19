"""The Claude Code agent's Agent Card: what agent-env negotiates against, where it expects it."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from tst.unit.examples.claude_code.agent import AGENT_DIR
from tst.unit.examples.claude_code.support import card

_CLAUDE_SKILL = {
    "id": "claude-code",
    "name": "claude-code",
    "description": "Runs the Claude Code CLI with full tool access (bash, read, edit, glob, grep).",
    "tags": ["coding", "agent"],
}


def _extensions(app) -> dict:
    return {ext["uri"]: ext for ext in card(app)["capabilities"]["extensions"]}


def test_identity(app):
    served = card(app)
    assert {key: served[key] for key in (
        "name", "description", "version", "url", "defaultInputModes", "defaultOutputModes", "skills",
    )} == {
        "name": "Claude Code",
        "description": "An A2A agent running the Claude Code CLI.",
        "version": "1.0.0",
        "url": "/a2a",
        "defaultInputModes": ["text", "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"],
        "defaultOutputModes": ["text"],
        "skills": [_CLAUDE_SKILL],
    }
    capabilities = served["capabilities"]
    assert (capabilities["streaming"], capabilities["pushNotifications"], capabilities["stateTransitionHistory"]) == (
        False, False, False)


def test_extensions_and_their_routes(app):
    routes = {
        uri: (ext["params"].get("endpoint"),
              {name: (method["method"], method.get("endpoint")) for name, method in ext["params"].get("methods", {}).items()})
        for uri, ext in _extensions(app).items()
    }
    assert routes == {
        "urn:agentenv:agent-config/v1": ("/ext/agent-config", {"set": ("POST", None), "get": ("GET", None)}),
        "urn:agentenv:trajectory/v1": ("/ext/trajectory", {"get": ("POST", None)}),
        "urn:agentenv:mcp-config/v1": ("/ext/mcp-config", {"add": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:skill-config/v1": ("/ext/skill-config", {"add": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:snapshot/v1": ("/ext/snapshot", {"save": ("POST", None), "load": ("PUT", None)}),
        "urn:agentenv:triggers/v1": ("/ext/triggers", {
            "register": ("POST", None), "decide": ("POST", "/ext/triggers/decide"), "state": ("GET", None),
        }),
        "urn:agentenv:staging/v1": ("/ext/staging", {}),
    }


def test_the_config_takes_every_field_prompt_agent_sends_that_the_cli_has(app):
    config = _extensions(app)["urn:agentenv:agent-config/v1"]["params"]["methods"]
    assert config["set"]["request"]["supported"] == [
        "allowed_tools", "description", "effort", "max_thinking_tokens", "max_turns", "model", "name",
        "output_format", "project_id", "role", "system_prompt", "task_id", "timeout_seconds",
    ]
    assert set(config["set"]["request"]["schema"]["properties"]) == set(config["set"]["request"]["supported"])
    assert config["get"] == {"method": "GET", "response": {"required": ["config"]}}


def test_request_and_response_contracts(app):
    methods = {uri: ext["params"].get("methods", {}) for uri, ext in _extensions(app).items()}
    assert methods["urn:agentenv:trajectory/v1"]["get"]["request"] == {
        "oneOf": [
            {"required": ["task_id"]},
            {"required": ["task_id", "objects"]},
            {"required": ["context_id"]},
            {"required": ["context_id", "objects"]},
            {"required": ["task_id", "after"], "optional": ["limit"]},
            {"required": ["context_id", "after"], "optional": ["limit"]},
        ],
    }
    assert methods["urn:agentenv:mcp-config/v1"] == {
        "add": {"method": "POST", "request": {"required": ["url"], "optional": ["headers", "name"]}},
        "list": {"method": "GET", "response": {"required": ["mcp_servers"]}},
    }
    assert methods["urn:agentenv:skill-config/v1"] == {
        "add": {
            "method": "POST",
            "request": {"required": ["name", "description"],
                        "oneOf": [{"required": ["skill_md"]}, {"required": ["skill_bundle"]}]},
            "response": {"required": ["name"]},
        },
        "list": {"method": "GET", "response": {"required": ["skills"]}},
    }
    assert methods["urn:agentenv:snapshot/v1"] == {
        "save": {"method": "POST", "request": {"required": ["context_id", "objects"]},
                 "response": {"required": ["context_id", "objects"]}},
        "load": {"method": "PUT", "request": {"required": ["objects"], "optional": ["target_context_id"]},
                 "response": {"required": ["context_id"]}},
    }
    triggers = methods["urn:agentenv:triggers/v1"]
    assert triggers["register"] == {"method": "POST", "request": {"required": ["triggers"]}}
    assert triggers["decide"] == {
        "method": "POST",
        "endpoint": "/ext/triggers/decide",
        "request": {"required": ["turn"], "optional": ["solver_message", "context_id", "env_triggers"]},
        "response": {"required": ["parts", "done", "fired"]},
    }
    assert triggers["state"] == {"method": "GET", "response": {"required": ["firing_log"]}}


def test_the_card_and_health_are_served(client):
    legacy, current, health = (client.get(path) for path in (
        "/.well-known/agent.json", "/.well-known/agent-card.json", "/health",
    ))
    assert legacy.json() == current.json()
    assert legacy.json()["name"] == "Claude Code"
    assert health.json() == {"status": "ok"}


def test_install_agent_reads_the_card_and_finds_no_install_extension():
    """install_agent runs this script in the image, with no agent environment, and parses its stdout as the card:
    the agent declares no install/v1, so it says so rather than failing to import."""
    script = ("import sys, json; sys.path.insert(0, '/app'); "
              "from a2a_server import AGENT_CARD; "
              "print(json.dumps(AGENT_CARD.model_dump()))")
    env = {**{k: v for k, v in os.environ.items() if not k.startswith(("LITELLM_", "ANTHROPIC_"))},
           "PYTHONPATH": str(AGENT_DIR)}
    out = subprocess.run([sys.executable, "-c", script], cwd=AGENT_DIR, env=env, capture_output=True, text=True,
                         check=True)
    uris = [ext["uri"] for ext in json.loads(out.stdout)["capabilities"]["extensions"]]
    assert "urn:agentenv:agent-config/v1" in uris and "urn:agentenv:install/v1" not in uris
