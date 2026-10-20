"""In-memory stand-ins for the boxd SDK's AsyncBoxd and its machines namespace."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from boxd import ExecResult, NotFoundError, Proxy


@dataclass
class FakeMachine:
    id: str
    name: str
    status: str = "running"
    egress_allow: list[str] = field(default_factory=list)
    proxies: list[Proxy] = field(default_factory=list)


class FakeProxies:
    def __init__(self, machines: FakeMachines):
        self._machines = machines

    async def list(self, machine_name: str) -> list[Proxy]:
        return list(self._machines.lookup(machine_name).proxies)


class FakeMachines:
    """Records every call; ``results`` answers exec in order, raising an exception found there."""

    def __init__(self):
        self.machines: dict[str, FakeMachine] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.egress_calls: list[tuple[str, list[str]]] = []
        self.deleted: list[str] = []
        self.ready_calls: list[str] = []
        self.results: list[ExecResult | BaseException] = []
        self.create_delay = 0.0
        self.create_error: BaseException | None = None
        # The server made the machine but the response was lost: create records it, then raises create_error.
        self.create_commits = False
        self.egress_error: BaseException | None = None
        self.delete_errors: list[BaseException] = []
        self.proxies = FakeProxies(self)
        self._ids = 0

    def lookup(self, machine_id: str) -> FakeMachine:
        if machine_id not in self.machines:
            raise NotFoundError("VM not found", code="not_found", grpc_code=5)
        return self.machines[machine_id]

    async def create(self, name: str | None = None, **kwargs: Any) -> FakeMachine:
        self.create_calls.append({"name": name, **kwargs})
        if self.create_delay:
            await asyncio.sleep(self.create_delay)
        self._ids += 1
        machine = FakeMachine(id=f"m-{self._ids}", name=name or f"machine-{self._ids}")
        machine.proxies = [
            Proxy(name=entry.name, machine_id=machine.id, machine_name=machine.name,
                  domain=f"{entry.name}.{machine.name}.boxd.test", port=entry.port, port_mode="locked",
                  is_default=False)
            for entry in kwargs.get("proxies") or []
        ]
        if self.create_error is not None and not self.create_commits:
            raise self.create_error
        self.machines[machine.id] = machine
        if self.create_error is not None:
            raise self.create_error
        return machine

    async def get(self, machine_id: str) -> FakeMachine:
        return self.lookup(machine_id)

    async def list(self, *, org: str | None = None, all_contexts: bool = False) -> list[FakeMachine]:
        return list(self.machines.values())

    async def delete(self, machine_id: str) -> None:
        if self.delete_errors:
            raise self.delete_errors.pop(0)
        self.lookup(machine_id)
        del self.machines[machine_id]
        self.deleted.append(machine_id)

    async def set_egress_allow(self, machine_id: str, entries: list[str]) -> list[str]:
        self.egress_calls.append((machine_id, list(entries)))
        if self.egress_error is not None:
            raise self.egress_error
        self.lookup(machine_id).egress_allow = sorted(entries)
        return sorted(entries)

    async def wait_until_ready(self, machine_id: str, *, timeout: float = 90.0, poll_interval: float = 0.5) -> FakeMachine:
        self.ready_calls.append(machine_id)
        return self.lookup(machine_id)

    async def exec(self, machine_id: str, command: Any, **kwargs: Any) -> ExecResult:
        self.exec_calls.append({"machine_id": machine_id, "command": command, **kwargs})
        outcome = self.results.pop(0) if self.results else ExecResult(stdout=b"", stderr=b"", exit_code=0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    """Each instance records itself on its class's ``instances``; ``on_init`` stages what a new one does."""

    instances: list[FakeClient]
    on_init: Any = None

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.machines = FakeMachines()
        type(self).instances.append(self)
        if type(self).on_init is not None:
            type(self).on_init(self)


def fake_client_cls() -> type[FakeClient]:
    """A fresh ``FakeClient`` class, so no test sees another's clients."""
    return type("FakeClient", (FakeClient,), {"instances": [], "on_init": None})


def result(stdout: bytes = b"", stderr: bytes = b"", exit_code: int = 0) -> ExecResult:
    return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
