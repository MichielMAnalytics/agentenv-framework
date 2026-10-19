"""A run's ``gateway_env_id`` / ``service_db_env_id`` overrides name the infra envs its deploys use, only within it."""

from __future__ import annotations

import pytest

from agent_env.config import gateway_env_id, get_config, service_db_env_id, set_document_store
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext


class _Probe:
    type = "probe"
    id, depends_on, retry_config, fail_task_on_error = "probe", None, None, True

    def __init__(self):
        self.seen = None

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        self.seen = (gateway_env_id(get_config()), service_db_env_id(get_config()))
        return context


@pytest.mark.asyncio
async def test_a_runs_override_names_its_gateway_and_only_within_the_run(tmp_path):
    set_document_store(LocalSqliteDocumentStore(str(tmp_path / "runs.db")))
    probe = _Probe()
    await Task(id="t", version=1, steps=[probe]).run(
        context=TaskStepContext(metadata={"user_overrides": {"gateway_env_id": "gw-run"}}))
    assert probe.seen == ("gw-run", get_config().default_service_db_env_id)
    assert gateway_env_id(get_config()) == get_config().default_gateway_env_id
