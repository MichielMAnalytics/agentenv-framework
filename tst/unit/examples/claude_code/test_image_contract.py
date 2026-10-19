"""The agent's folder is its build context: the Dockerfile copies only its own files, keeps TLS checks on, and
installs the protocol the release pins."""

from __future__ import annotations

import re

from tst.unit.examples.claude_code.agent import AGENT_DIR

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()


def test_every_file_the_image_copies_is_in_the_folder():
    copied = [name for line in re.findall(r"^COPY (.+) \S+$", DOCKERFILE, re.MULTILINE) for name in line.split()]

    assert copied and all((AGENT_DIR / name).is_file() for name in copied), copied
    # install.sh and run.sh are install/v1's; a build context pruned to what the Dockerfile reads keeps them so.
    assert {"a2a_server.py", "claude_bridge.py", "kit.py", "entrypoint.sh", "install.sh", "run.sh"} <= set(copied)


def test_the_image_installs_the_protocol_alone_from_its_requirements():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert requirements == ["agentenv-framework-protocol[agent]"]


def test_the_image_pins_its_cli_checks_its_checksum_and_keeps_tls_on():
    assert re.search(r"^ARG CLAUDE_CODE_VERSION=\d+\.\d+\.\d+$", DOCKERFILE, re.MULTILINE)
    assert "sha256sum -c" in DOCKERFILE
    for path in (AGENT_DIR / "Dockerfile", AGENT_DIR / "entrypoint.sh"):
        assert "NODE_TLS_REJECT_UNAUTHORIZED" not in path.read_text()


def test_the_image_starts_through_cmd_which_modal_passes_its_env_to():
    assert 'CMD ["/opt/agent/entrypoint.sh"]' in DOCKERFILE
    assert not re.search(r"^ENTRYPOINT", DOCKERFILE, re.MULTILINE)
