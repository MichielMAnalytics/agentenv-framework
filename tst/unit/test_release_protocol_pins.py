"""The files ``.github/workflows/publish-pypi.yml`` rewrites to pin agentenv-framework-protocol on a release: each must
hold exactly one line its sed matches, or the release fails after the protocol is already on PyPI."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = (REPO / ".github" / "workflows" / "publish-pypi.yml").read_text()


def test_the_pyproject_holds_the_one_quoted_protocol_requirement_the_release_pins():
    assert len(re.findall(r'"agentenv-framework-protocol[^"]*"', (REPO / "pyproject.toml").read_text())) == 1


def test_each_shipped_example_requirements_file_the_release_pins_holds_one_protocol_line():
    (pattern,) = re.findall(r"^\s*for requirements in (\S+); do$", WORKFLOW, re.MULTILINE)
    pinned = {path.relative_to(REPO).as_posix(): [line for line in path.read_text().splitlines()
                                                   if line.startswith("agentenv-framework-protocol")]
              for path in REPO.glob(pattern)}

    assert {"src/agent_env/examples/echo-mcp/envs/echo/requirements.txt",
            "src/agent_env/examples/agents/agents/claude-code/requirements.txt"} <= set(pinned)
    assert all(len(lines) == 1 for lines in pinned.values()), pinned
