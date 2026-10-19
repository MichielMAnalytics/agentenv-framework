"""What a turn sends the CLI and what agent-env reads back: the terminal message's parts for a
success and for each failure, and the session threading across turns."""

from __future__ import annotations

import base64

import pytest
from agentenv_protocol.a2a_agent import AGENT_CONFIG_V1

from tst.unit.examples.claude_code.support import send, terminal
from tst.unit.examples.claude_code.agent import a2a_server, claude_bridge, kit

AGENT_CONFIG_PATH = AGENT_CONFIG_V1.endpoint
MCP_CONFIG_PATH = a2a_server.MCP_CONFIG_V1.endpoint


class Calls(list):
    """Every run_claude call's arguments; each is answered with ``result_for(kwargs)``."""

    def __init__(self) -> None:
        super().__init__()
        self.result_for = lambda kwargs: claude_bridge.ClaudeResult(
            response="done", tool_call_count=2, session_id=kwargs["session_id"], trajectory=[{"type": "result"}],
        )


@pytest.fixture
def calls(monkeypatch) -> Calls:
    recorded = Calls()

    async def _run_claude(**kwargs):
        recorded.append(kwargs)
        result = recorded.result_for(kwargs)
        # As the CLI does: the session's transcript is on disk once a turn has run.
        transcript = claude_bridge.session_jsonl_path(result.session_id or kwargs["session_id"])
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text('{"type": "assistant"}\n')
        return result

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)
    return recorded


def test_a_successful_turn(calls, client):
    state, text, data = terminal(send(client, "hello"))

    assert (state, text) == ("completed", "done")
    assert data == [{"tool_call_count": 2}, {"usage": {"tool_call_count": 2}}]


def test_a_resumed_turn_reports_its_attempts(calls, client):
    calls.result_for = lambda kwargs: claude_bridge.ClaudeResult(
        response="done", tool_call_count=1, session_id=kwargs["session_id"], attempts=3,
    )

    _, _, data = terminal(send(client, "hello"))

    assert data == [{"tool_call_count": 1, "cli_attempts": 3}, {"usage": {"tool_call_count": 1}}]


def test_a_cli_error_is_a_failed_task_with_the_capped_message(calls, client):
    error = "claude CLI exited 1: " + "x" * 2000
    calls.result_for = lambda kwargs: claude_bridge.ClaudeResult(
        response="", tool_call_count=4, session_id=kwargs["session_id"], error=error,
    )

    state, text, data = terminal(send(client, "hello"))

    assert (state, text) == ("failed", error)
    assert data == [
        {"tool_call_count": 4},
        {
            "usage": {"tool_call_count": 4},
            "error_type": "agent_error",
            "error_code": "cli_error",
            "error_message": error[:1000],
        },
    ]


def test_a_cli_error_keeps_the_response_text_when_there_is_one(calls, client):
    calls.result_for = lambda kwargs: claude_bridge.ClaudeResult(
        response="partial answer", tool_call_count=0, session_id=kwargs["session_id"], error="boom",
    )

    state, text, _ = terminal(send(client, "hello"))

    assert (state, text) == ("failed", "partial answer")


def test_a_bridge_exception_is_an_infra_error(monkeypatch, client):
    async def _raise(**_kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(a2a_server, "run_claude", _raise)

    state, text, data = terminal(send(client, "hello"))

    assert (state, text) == ("failed", "spawn failed")
    assert data == [{"error_type": "infra_error", "error_code": "infra_error", "error_message": "spawn failed"}]


def test_an_unreadable_file_part_is_a_filepart_error(calls, client):
    parts = [{"kind": "file", "file": {"name": "a.png", "mimeType": "image/png", "bytes": "not base64!"}}]

    state, text, data = terminal(send(client, parts))

    assert state == "failed" and text.startswith("malformed base64 in FilePart 'a.png'")
    assert data == [{"error_type": "agent_error", "error_code": "filepart_error", "error_message": text}]
    assert calls == []


def test_file_parts_are_materialized_and_their_paths_appended(monkeypatch, tmp_path, calls, client):
    monkeypatch.setattr(a2a_server, "INPUTS_DIR", tmp_path / "inputs")
    image = base64.b64encode(b"png-bytes").decode()
    parts = [
        {"kind": "text", "text": "describe"},
        {"kind": "file", "file": {"name": "../../etc/a b.png", "mimeType": "image/png", "bytes": image}},
        {"kind": "data", "data": {"ignored": True}},
    ]

    task = send(client, parts)

    (call,) = calls
    path = tmp_path / "inputs" / task["id"] / "a_b.png"
    assert call["prompt"] == f"describe\n\n{path}"
    assert path.read_bytes() == b"png-bytes"


def test_a_context_resumes_its_session_and_a_new_context_starts_one(calls, client):
    send(client, "one", context_id="ctx-a")
    send(client, "two", context_id="ctx-a")
    send(client, "three", context_id="ctx-b")

    first, second, third = calls
    assert (first["resume"], second["resume"], third["resume"]) == (False, True, False)
    assert second["session_id"] == first["session_id"] != third["session_id"]


def test_a_session_id_the_cli_reports_is_trusted_for_the_next_turn(calls, client):
    calls.result_for = lambda kwargs: claude_bridge.ClaudeResult(response="ok", tool_call_count=0, session_id="wire-session")

    send(client, "one")
    send(client, "two")

    assert calls[1]["session_id"] == "wire-session" and calls[1]["resume"] is True


def test_agent_config_and_mcp_servers_reach_the_cli(calls, client):
    config = {
        "model": "claude-opus-4-6",
        "allowed_tools": ["Bash", "Read"],
        "system_prompt": "be brief",
        "output_format": {"type": "json_schema", "schema": {"type": "object"}},
        "timeout_seconds": 42,
        "project_id": "p-1",
        "task_id": "t-1",
        "role": "solver",
        "effort": "high",
        "max_turns": 7,
        "max_thinking_tokens": 4096,
    }
    assert client.post(AGENT_CONFIG_PATH, json=config).json() == {"status": "updated"}
    client.post(MCP_CONFIG_PATH, json={"url": "https://gw/mcp", "name": "env", "headers": {"CF-Access-Client-Id": "id"}})

    send(client, "go")

    (call,) = calls
    assert {key: call[key] for key in config if key != "timeout_seconds"} == {
        key: value for key, value in config.items() if key != "timeout_seconds"
    }
    assert call["timeout_s"] == 42
    assert call["mcp_servers"] == {"env": {"url": "https://gw/mcp", "headers": {"CF-Access-Client-Id": "id"}}}


def test_the_default_timeout_is_thirty_minutes(calls, client):
    send(client, "go")

    assert calls[0]["timeout_s"] == 1800 and calls[0]["model"] is None and calls[0]["mcp_servers"] == {}


def test_session_references_outlive_the_frameworks_bounded_record():
    assert issubclass(a2a_server.ClaudeCodeAgent, kit.DurableSessionRefs)


def test_a_turn_that_failed_before_the_cli_wrote_a_transcript_starts_its_session_afresh(monkeypatch, client):
    seen = []

    async def _run_claude(**kwargs):
        seen.append((kwargs["session_id"], kwargs["resume"]))
        if len(seen) == 1:
            raise ValueError("output_format must be {...}")
        return claude_bridge.ClaudeResult(response="ok", tool_call_count=0, session_id=kwargs["session_id"])

    monkeypatch.setattr(a2a_server, "run_claude", _run_claude)

    assert terminal(send(client, "one"))[0] == "failed"
    assert terminal(send(client, "two"))[0] == "completed"
    (first, _), (second, resumed) = seen
    assert second == first and resumed is False


def test_attachments_are_written_while_the_turn_holds_a_snapshot_load_off(monkeypatch, agent, calls, client):
    active = []

    async def _materialize(part, target_dir):
        active.append("ctx-1" in agent._active_contexts)
        return str(target_dir / "a.png")

    monkeypatch.setattr(a2a_server, "materialize_file_part", _materialize)

    send(client, [{"kind": "file", "file": {"name": "a.png", "mimeType": "image/png", "bytes": "cG5n"}}])

    assert active == [True] and agent._active_contexts == set()
