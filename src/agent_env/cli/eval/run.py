"""Run all tasks in an eval through the configured ``[runner]``, following each run to its end."""

import copy
import json
import os
from uuid import uuid4

import click

from agent_env.cli import submitted_runs
from agent_env.cli.banner import print_banner
from agent_env.cli.identity import get_agent_env_client_id
from agent_env.store.ids import fs_safe
from agent_env.task.interrupts import Interrupts
from agent_env.task.store import TaskStepStatus
from agent_env.task_step.context import TaskStepContext


def _write_context(context, task_id, output_dir, prefix=""):
    """Write context JSON to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{fs_safe(task_id)}_{uuid4().hex[:8]}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(context.to_safe_dict(), f, indent=2)
    click.echo(click.style(f"{prefix}Output written to: {output_path}", fg="blue"))


def _echo_step(tasks):
    """Echo each step a run finishes as the runner records it, with the score of a verifier step."""
    steps = {task.id: {step.id: (i, step, len(task.steps)) for i, step in enumerate(task.steps)} for task in tasks}

    def echo(run, result, instance):
        i, step, total = steps[run.task_id][result.step_id]
        passed = result.status == TaskStepStatus.SUCCESS
        click.echo(click.style(f"{run.tag} {'Completed' if passed else 'Failed'} step [{i + 1}/{total}]: {step.type} "
                               f"(id={step.id})", fg="green" if passed else "red"))
        vid = getattr(step, "verifier_id", None)
        verifications = ((instance.context or {}).get("metadata") or {}).get("verifications") or {}
        if passed and vid and (score := (verifications.get(vid) or {}).get("score")) is not None:
            click.echo(click.style(f"{run.tag} score (verifier_id={vid}): {score}", fg="green"))

    return echo


def _run_submitted(tasks, k: int, base_metadata: dict, output_dir: str | None, agent_model: str | None,
                   agent_artifact_id: str | None, max_concurrency: int | None) -> None:
    """The configured ``[runner]`` runs every task ``k`` times, at most ``max_concurrency`` at once, and tears the
    runs down; this follows them and reports each run's score from its task instance."""
    submissions = [(f"[{task.id} run {i}]" if k > 1 else f"[{task.id}]", task, copy.deepcopy(base_metadata), None)
                   for task in tasks for i in range(1, k + 1)]
    runs: list[submitted_runs.SubmittedRun] = []
    with Interrupts() as interrupts:
        [outcome] = interrupts.run(interrupts.gather(
            [submitted_runs.submit_and_follow(submissions, runs, agent_model=agent_model,
                                              agent_artifact_id=agent_artifact_id, limit=max_concurrency,
                                              on_step=_echo_step(tasks))],
            submitted_runs.on_signal))
    if not outcome.cancelled() and outcome.exception() is not None:
        raise outcome.exception()
    click.echo()
    results = []
    for run in runs:
        instance = run.instance()
        context = TaskStepContext.from_dict(instance.context) if instance and instance.context else None
        if output_dir and context is not None:
            _write_context(context, run.task_id, output_dir, prefix=f"{run.tag} ")
        results.append((run, context.metadata.get("score") if context else None))
    passed = [(run, score) for run, score in results if run.completed]
    if passed:
        click.echo(click.style("Results:", fg="blue"))
        for run, score in passed:
            click.echo(click.style(f"  {run.tag} PASSED{f' score={score}' if score is not None else ''}", fg="green"))
    for run, _ in results:
        if not run.completed:
            outcome_text = (f"{run.record.status}" + (f": {run.record.error}" if run.record.error else "")
                            if run.done else "CANCELLED")
            click.echo(click.style(f"  {run.tag} {outcome_text}", fg="red"))
    total = len(submissions)
    if len(passed) == total:
        click.echo(click.style(f"\nAll {total} runs completed!", fg="blue"))
    else:
        click.echo(click.style(f"\n{len(passed)}/{total} runs completed, {total - len(passed)}/{total} failed.", fg="red"))
    if interrupts.count:
        raise SystemExit(128 + interrupts.signum)
    if len(passed) < total:
        raise SystemExit(1)


@click.command()
@click.option("--id", "eval_id", required=True, help="Eval id")
@click.option("--version", "eval_version", default=None, type=int, help="Eval version (defaults to latest)")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False), help="Directory to write per-task context JSON output")
@click.option("--k", "k", default=1, type=int, help="Number of parallel runs per task")
@click.option("--max-concurrency", default=None, type=int, help="Maximum number of concurrent task runs")
@click.option("--agent-model", default=None, type=str, help="Override the model used by prompt_agent steps")
@click.option("--user-model", default=None, type=str, help="Override the model the user simulator runs on")
@click.option("--judge-model", default=None, type=str, help="Override the model rubrics_verifier judges grade with")
@click.option("--agent-artifact-id", default=None, type=str, help="Override the agent image artifact for deploy_agent steps")
def run(eval_id: str, eval_version: int | None, output_dir: str | None, k: int, max_concurrency: int | None, agent_model: str | None, user_model: str | None, judge_model: str | None, agent_artifact_id: str | None):
    """Run all tasks in an eval through the configured [runner], concurrently."""
    if k < 1:
        raise click.BadParameter("must be at least 1", param_hint="'--k'")
    if max_concurrency is not None and max_concurrency < 1:
        raise click.BadParameter("must be at least 1", param_hint="'--max-concurrency'")

    from agent_env.eval import Eval
    from agent_env.task import Task

    click.echo(f"Fetching eval: id={eval_id} version={eval_version or 'latest'}...")
    eval_obj = Eval.get(eval_id, version=eval_version)
    click.echo(f"Found eval: id={eval_obj.id} version={eval_obj.version} tasks={len(eval_obj.tasks)}")

    tasks = []
    for et in eval_obj.tasks:
        task = Task.get(et.task_id, version=et.task_version)
        version_note = " (resolved from latest)" if et.task_version is None else ""
        click.echo(f"  Loaded task: {task.id} v{task.version} ({len(task.steps)} steps){version_note}")
        tasks.append(task)

    print_banner()

    client_id = get_agent_env_client_id()
    base_metadata = {"agent_env_hub": {"caller": client_id}} if client_id else {}
    if overrides := {key: value for key, value in (("user_model", user_model), ("judge_model", judge_model)) if value}:
        base_metadata["user_overrides"] = overrides
    total_runs = len(tasks) * k
    concurrency_note = f" (max concurrency: {max_concurrency})" if max_concurrency else ""
    click.echo(click.style(
        f"Running {len(tasks)} tasks x {k} run(s) = {total_runs} total runs{concurrency_note}...",
        fg="blue",
    ))
    click.echo()
    _run_submitted(tasks, k, base_metadata, output_dir, agent_model, agent_artifact_id, max_concurrency)
