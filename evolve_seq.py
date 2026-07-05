#!/usr/bin/env python3
"""Sequential per-task reflection without code evolution.

This runner is intentionally separate from evolve.py. It keeps one isolated
workspace per task, repeatedly evaluates the unchanged code agent, and uses the
agent-debugger CLI (ADB) to summarize task-local trajectories for later rounds.

There is no meta agent with tools and no code evolution. ADB only reads the
current task's sanitized traces and writes a Markdown summary outside the
code-agent workspace.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from evolve import (
    EVOLVE_AGENT_DIR,
    EXPERIMENTS_DIR,
    _discover_per_task_names,
    _ensure_adb_installed,
    _find_adb,
    _invoke_adb_ask_once,
    _round_record_for_task,
    _safe_task_dir_name,
    _sanitize_trajectories_for_evolver,
    _single_task_config,
    apply_code_agent_patch,
    compute_stats,
    init_workspace,
    load_config,
    resolve_source_dir,
    run_harbor,
)


SEQ_CONTEXT_HEADER = "## Task-Local Sequential Reflection Context"
SEQ_CONTEXT_BEGIN = "<!-- AHE_SEQ_CONTEXT_BEGIN -->"
SEQ_CONTEXT_END = "<!-- AHE_SEQ_CONTEXT_END -->"
SEQ_RESULT_CONTEXT_PATH = "previous_result_context.md"


def create_seq_experiment_dir(config: dict, config_path: str,
                              experiment_name: str | None = None) -> Path:
    """Create a seq-reflection experiment directory without touching evolve.py."""
    if experiment_name:
        exp_dir = EXPERIMENTS_DIR / experiment_name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        meta_name = config.get("_meta", {}).get("_name", "seq-reflection")
        exp_dir = EXPERIMENTS_DIR / f"{timestamp}__{meta_name}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "tasks").mkdir(exist_ok=True)

    snapshot_path = exp_dir / "config_snapshot.yaml"
    if not snapshot_path.exists():
        snapshot = {k: v for k, v in config.items() if k != "_meta"}
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True)

    if config.get("_meta"):
        overlay_dst = exp_dir / "experiment_overlay.yaml"
        if not overlay_dst.exists() and Path(config_path).exists():
            shutil.copy2(config_path, overlay_dst)

    print(f"[seq] Experiment directory: {exp_dir}")
    return exp_dir


def _seq_cfg(config: dict) -> dict:
    return config.get("seq_evolution", {}) or config.get("sequential_reflection", {}) or {}


def _seq_exposes_test_results(config: dict) -> bool:
    seq_cfg = _seq_cfg(config)
    return not bool(seq_cfg.get("hide_test_results_from_summary", True))


def _strip_old_seq_context(text: str) -> str:
    pattern = re.compile(
        rf"\n*{re.escape(SEQ_CONTEXT_BEGIN)}.*?{re.escape(SEQ_CONTEXT_END)}\n*",
        re.DOTALL,
    )
    return pattern.sub("\n", text).rstrip() + "\n"


def _ensure_base_system_prompt(task_root: Path, workspace_dir: Path) -> Path:
    """Persist the original prompt so context injection never compounds."""
    base_path = task_root / "systemprompt.base.md"
    prompt_path = workspace_dir / "systemprompt.md"
    if not base_path.exists():
        base_text = _strip_old_seq_context(prompt_path.read_text(encoding="utf-8"))
        base_path.write_text(base_text, encoding="utf-8")
    return base_path


def _write_workspace_prompt_with_summary(
    task_root: Path,
    workspace_dir: Path,
    *,
    include_previous_result_context: bool = False,
) -> None:
    """Inject task-local summary context into the code-agent system prompt."""
    base_path = _ensure_base_system_prompt(task_root, workspace_dir)
    prompt_path = workspace_dir / "systemprompt.md"
    summary_path = task_root / "cumulative_summary.md"
    result_path = task_root / SEQ_RESULT_CONTEXT_PATH

    base_text = base_path.read_text(encoding="utf-8").rstrip()
    has_summary = summary_path.exists() and bool(summary_path.read_text(encoding="utf-8").strip())
    has_result_context = (
        include_previous_result_context
        and result_path.exists()
        and bool(result_path.read_text(encoding="utf-8").strip())
    )
    if not has_summary and not has_result_context:
        prompt_path.write_text(base_text + "\n", encoding="utf-8")
        return

    summary = summary_path.read_text(encoding="utf-8").strip() if has_summary else ""
    result_context = result_path.read_text(encoding="utf-8").strip() if has_result_context else ""
    result_hint = ""
    if has_result_context:
        result_hint = (
            "The notes may also include external verifier outcomes from the immediately "
            "previous attempt. Treat those result snippets as higher-priority evidence "
            "than the agent's own self-assessment."
        )
    context = f"""
{SEQ_CONTEXT_BEGIN}

{SEQ_CONTEXT_HEADER}

You will receive the original benchmark task as the user query. In addition,
the task-local notes below summarize only this same task's previous agent
trajectories. Use them to avoid repeating confusion, slow detours, or incomplete
plans. Treat the notes as fallible hints, not ground truth, and solve the actual
user task you receive. {result_hint}

{summary}

{result_context}

{SEQ_CONTEXT_END}
"""
    prompt_path.write_text(base_text + "\n\n" + context.strip() + "\n", encoding="utf-8")


def _load_adb_summary_prompt() -> str:
    prompt_path = EVOLVE_AGENT_DIR / "seq_adb_summary_prompt.md"
    return prompt_path.read_text(encoding="utf-8")


def _collect_adb_trace_paths(trajectory_history: Path, iteration: int) -> tuple[list[Path], str | None]:
    """Collect this task's sanitized traces for ADB.

    Prefer cleaned traces. If any rollout lacks a cleaned trace, use raw
    in-memory traces with ADB's explicit in_memory_tracer parser.
    """
    iter_dir = trajectory_history / f"iteration_{iteration:03d}"
    if not iter_dir.exists():
        return [], None

    rollout_dirs = sorted(p for p in iter_dir.iterdir() if p.is_dir())
    cleaned = [p / "nexau_in_memory_tracer.cleaned.json" for p in rollout_dirs]
    if cleaned and all(p.exists() for p in cleaned):
        return cleaned, None

    raw = [p / "nexau_in_memory_tracer.json" for p in rollout_dirs]
    raw = [p for p in raw if p.exists()]
    return raw, "in_memory_tracer"


def _adb_llm_env(config: dict) -> dict[str, str]:
    """Build env for ADB, preferring agent_debugger.llm and falling back to llm."""
    env = os.environ.copy()
    base_llm = config.get("llm", {})
    adb_llm = config.get("agent_debugger", {}).get("llm", {})
    llm = {
        "model": adb_llm.get("model") or base_llm.get("model"),
        "base_url": adb_llm.get("base_url") or base_llm.get("base_url"),
        "api_key": adb_llm.get("api_key") or base_llm.get("api_key"),
        "api_type": adb_llm.get("api_type"),
    }
    if llm.get("model"):
        env["QA_MODEL_NAME"] = llm["model"]
    if llm.get("base_url"):
        env["QA_BASE_URL"] = llm["base_url"]
    if llm.get("api_key"):
        env["QA_API_KEY"] = llm["api_key"]
    if llm.get("api_type"):
        env["QA_API_TYPE"] = llm["api_type"]
    return env


def _configure_adb_llm(config: dict) -> None:
    """Persist the full experiment-specific LLM config before parallel ADB calls."""
    adb_llm = config.get("agent_debugger", {}).get("llm", {})
    if not adb_llm:
        return
    if not _ensure_adb_installed():
        raise RuntimeError("agent-debugger CLI could not be installed or found")

    adb = _find_adb() or "adb"
    payload = json.dumps({"llm": adb_llm})
    result = subprocess.run(
        [adb, "config", payload],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:300]
        raise RuntimeError(f"failed to configure agent debugger: {detail}")

    print(
        f"[seq-adb] configured model={adb_llm.get('model')} "
        f"max_tokens={adb_llm.get('max_tokens', 'default')}",
        flush=True,
    )


def _call_adb_summary(config: dict, trace_paths: list[Path],
                      trace_type: str | None, query: str,
                      adb_semaphore: threading.BoundedSemaphore | None = None) -> str:
    if not trace_paths:
        return "(No sanitized trajectory traces were available for ADB to analyze.)"
    if not _ensure_adb_installed():
        return "[adb unavailable] agent-debugger-cli could not be installed or found."

    adb = _find_adb() or "adb"
    cmd = [adb, "ask", "-t"] + [str(path) for path in trace_paths]
    if trace_type:
        cmd += ["--trace-type", trace_type]
    cmd += ["-q", query, "--format", "json"]

    adb_cfg = config.get("agent_debugger", {})
    timeout = float(adb_cfg.get("timeout_per_task", 180))
    retry_attempts = max(1, int(adb_cfg.get("retry_attempts", 3)))
    backoff = float(adb_cfg.get("retry_backoff_seconds", 2.0))
    env = _adb_llm_env(config)

    if adb_semaphore is not None:
        adb_semaphore.acquire()
    try:
        response = ""
        for attempt in range(retry_attempts):
            response = _invoke_adb_ask_once(cmd, env, timeout)
            if not response.startswith("[adb"):
                return response.strip()
            if attempt < retry_attempts - 1:
                print(
                    f"[seq-adb] attempt {attempt + 1}/{retry_attempts} failed; "
                    f"retrying in {backoff:.1f}s: {response[:180]}",
                    flush=True,
                )
                time.sleep(backoff)
                backoff *= 2.0
        return response.strip()
    finally:
        if adb_semaphore is not None:
            adb_semaphore.release()


def _collect_seq_verifier_outputs(job_dir: Path) -> list[tuple[str, str]]:
    """Collect verifier stdout snippets for the current task's rollouts."""
    outputs: list[tuple[str, str]] = []
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        reward_path = trial_dir / "verifier" / "reward.txt"
        reward = 0.0
        if reward_path.exists():
            try:
                reward = float(reward_path.read_text().strip())
            except ValueError:
                reward = 0.0
        label = "PASS" if reward >= 1.0 else "FAIL"
        test_stdout = trial_dir / "verifier" / "test-stdout.txt"
        if not test_stdout.exists():
            continue
        try:
            text = test_stdout.read_text(errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        lines = text.splitlines()
        if len(lines) > 60:
            text = "\n".join(lines[-60:])
        outputs.append((label, text))
    return outputs


def _build_seq_result_context(iteration: int, stats: dict, verifier_outputs: list[tuple[str, str]]) -> str:
    """Build prompt context describing the immediately previous external results."""
    pass_rate = float(stats.get("pass_rate", 0.0))
    n_pass = int(stats.get("n_pass", 0))
    n_fail = int(stats.get("n_fail", 0))
    n_exception = int(stats.get("n_exception", 0))
    n_total = int(stats.get("n_total", n_pass + n_fail + n_exception))
    lines = [
        "## Previous Attempt Result",
        "",
        "These are external verifier outcomes from the immediately previous attempt.",
        "Treat them as ground truth about what actually passed or failed.",
        "",
        f"- Iteration: {iteration}",
        f"- Rollouts: {n_total}",
        f"- pass/fail/exception: {n_pass}/{n_fail}/{n_exception}",
        f"- pass rate: {pass_rate:.1%}",
    ]
    non_pass_outputs = [(label, text) for label, text in verifier_outputs if label != "PASS"]
    if non_pass_outputs:
        lines.extend([
            "",
            "### Verifier Feedback Excerpts",
            "",
            "Focus on these messages when deciding what to change next.",
        ])
        for idx, (label, text) in enumerate(non_pass_outputs, 1):
            lines.extend([
                "",
                f"#### rollout {idx} ({label})",
                "```text",
                text,
                "```",
            ])
    return "\n".join(lines).strip() + "\n"


def _write_previous_result_context(
    task_root: Path,
    iteration: int,
    stats: dict,
    verifier_outputs: list[tuple[str, str]],
    *,
    enabled: bool,
) -> None:
    result_path = task_root / SEQ_RESULT_CONTEXT_PATH
    if not enabled:
        return
    result_path.write_text(
        _build_seq_result_context(iteration, stats, verifier_outputs),
        encoding="utf-8",
    )


def _summarize_iteration(
    *,
    config: dict,
    task_name: str,
    task_root: Path,
    iteration: int,
    trajectory_history: Path,
    stats: dict,
    verifier_outputs: list[tuple[str, str]],
    include_test_results: bool,
    adb_semaphore: threading.BoundedSemaphore | None = None,
) -> str:
    prior_summary_path = task_root / "cumulative_summary.md"
    prior_summary = (
        prior_summary_path.read_text(encoding="utf-8")
        if prior_summary_path.exists()
        else ""
    ).strip()
    prompt_template = _load_adb_summary_prompt()
    query = prompt_template.format_map({
        "task_name": task_name,
        "iteration": iteration,
        "previous_summary": prior_summary or "(none yet)",
    })
    if include_test_results:
        query += (
            "\n\nIMPORTANT: External verifier outcomes from this iteration are included below. "
            "Treat them as higher-priority evidence than the agent's self-assessment when "
            "you summarize what the next attempt should change.\n\n"
        )
        query += _build_seq_result_context(iteration, stats, verifier_outputs)
    trace_paths, trace_type = _collect_adb_trace_paths(trajectory_history, iteration)
    summary = _call_adb_summary(
        config, trace_paths, trace_type, query, adb_semaphore=adb_semaphore
    ).strip()
    if not summary:
        summary = "(ADB returned no summary text.)"

    summaries_dir = task_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    (summaries_dir / f"iteration_{iteration:03d}.md").write_text(summary + "\n", encoding="utf-8")
    prior_summary_path.write_text(summary + "\n", encoding="utf-8")
    return summary


def _write_task_summary(task_root: Path, summary: dict) -> None:
    with open(task_root / "seq_task_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _load_existing_task_summary(task_root: Path, start_iteration: int) -> dict:
    path = task_root / "seq_task_summary.json"
    if not path.exists() or start_iteration <= 1:
        return {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    rounds = [
        r for r in existing.get("rounds", [])
        if int(r.get("iteration", 0)) < start_iteration
    ]
    existing["rounds"] = rounds
    return existing


def _run_one_seq_task(*, config: dict, source_dir: Path, exp_dir: Path,
                      task_name: str, agent_config_filename: str,
                      start_iteration: int, max_iterations: int,
                      code_agent_patch: dict, n_concurrent_rollouts: int,
                      skip_eval: bool,
                      adb_semaphore: threading.BoundedSemaphore | None = None) -> dict:
    safe_task = _safe_task_dir_name(task_name)
    task_root = exp_dir / "tasks" / safe_task
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "task_name.txt").write_text(task_name + "\n", encoding="utf-8")

    workspace_dir = task_root / "workspace"
    is_new = init_workspace(source_dir, workspace_dir)
    if is_new:
        apply_code_agent_patch(workspace_dir, agent_config_filename, code_agent_patch)
    _ensure_base_system_prompt(task_root, workspace_dir)

    task_config = _single_task_config(config, task_name, n_concurrent_rollouts)
    k = int(task_config.get("harbor", {}).get("k", 1))
    include_test_results = _seq_exposes_test_results(task_config)
    existing = _load_existing_task_summary(task_root, start_iteration)
    rounds: list[dict] = existing.get("rounds", [])

    summary: dict = {
        "task_name": task_name,
        "safe_task_name": safe_task,
        "rollout_k": k,
        "max_iterations": max_iterations,
        "mode": "adb_summary_sequential_reflection",
        "workspace": "workspace",
        "rounds": rounds,
    }
    if rounds:
        summary.update({
            "ever_pass": any(r.get("any_pass") for r in rounds),
            "first_pass_iteration": next((r.get("iteration") for r in rounds if r.get("any_pass")), None),
            "final_iteration": rounds[-1],
        })
    _write_task_summary(task_root, summary)

    for iteration in range(start_iteration, max_iterations + 1):
        iter_start = time.monotonic()
        timing: dict[str, float] = {}
        iteration_dir = task_root / "runs" / f"iteration_{iteration:03d}"
        input_dir = iteration_dir / "input"
        benchmark_dir = input_dir / "benchmark"
        input_dir.mkdir(parents=True, exist_ok=True)
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        _write_workspace_prompt_with_summary(
            task_root,
            workspace_dir,
            include_previous_result_context=include_test_results,
        )
        prompt_snapshot = input_dir / "systemprompt.md"
        if not prompt_snapshot.exists():
            shutil.copy2(workspace_dir / "systemprompt.md", prompt_snapshot)

        eval_phase_start = time.monotonic()
        if skip_eval and iteration == start_iteration:
            from evolve import find_latest_job_dir

            job_dir = find_latest_job_dir(benchmark_dir)
            if job_dir is None:
                raise RuntimeError(f"--skip-eval requested but no job found for task {task_name}")
        else:
            job_dir = run_harbor(task_config, workspace_dir, agent_config_filename, benchmark_dir)
        trajectories_dir = _sanitize_trajectories_for_evolver(task_root, iteration, job_dir, task_name)
        verifier_outputs = _collect_seq_verifier_outputs(job_dir)
        eval_elapsed = time.monotonic() - eval_phase_start

        stats_phase_start = time.monotonic()
        stats = compute_stats(job_dir, k=k)
        eval_elapsed += time.monotonic() - stats_phase_start
        timing["eval_min"] = round(eval_elapsed / 60, 1)

        phase_start = time.monotonic()
        summary_text = _summarize_iteration(
            config=config,
            task_name=task_name,
            task_root=task_root,
            iteration=iteration,
            trajectory_history=trajectories_dir,
            stats=stats,
            verifier_outputs=verifier_outputs,
            include_test_results=include_test_results,
            adb_semaphore=adb_semaphore,
        )
        timing["summary_min"] = round((time.monotonic() - phase_start) / 60, 1)

        record = _round_record_for_task(task_name, iteration, stats, job_dir, task_root)
        timing["total_min"] = round((time.monotonic() - iter_start) / 60, 1)
        record["timing"] = timing
        record["summary_path"] = f"summaries/iteration_{iteration:03d}.md"
        rounds.append(record)
        _write_previous_result_context(
            task_root,
            iteration,
            stats,
            verifier_outputs,
            enabled=include_test_results,
        )

        summary.update({
            "ever_pass": any(r.get("any_pass") for r in rounds),
            "first_pass_iteration": next((r.get("iteration") for r in rounds if r.get("any_pass")), None),
            "final_iteration": rounds[-1],
            "latest_summary_path": "cumulative_summary.md",
            "latest_summary_preview": summary_text[:1000],
        })
        _write_task_summary(task_root, summary)

    return summary


def _write_seq_aggregate(exp_dir: Path, task_summaries: list[dict],
                         max_iterations: int, rollout_k: int,
                         target_task_count: int | None = None) -> dict:
    ok = [s for s in task_summaries if not s.get("error")]
    errors = [s for s in task_summaries if s.get("error")]
    total_target = int(target_task_count or len(task_summaries))
    per_iteration: list[dict] = []
    cumulative_by_iteration: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        round_records = []
        cumulative_pass_tasks = 0
        for task_summary in ok:
            rounds = task_summary.get("rounds", [])
            round_record = next((r for r in rounds if r.get("iteration") == iteration), None)
            if round_record:
                round_records.append(round_record)
            if any(r.get("iteration", 0) <= iteration and r.get("any_pass") for r in rounds):
                cumulative_pass_tasks += 1

        n_total = len(round_records)
        rollout_pass = sum(r.get("rollouts", {}).get("pass", 0) for r in round_records)
        rollout_fail = sum(r.get("rollouts", {}).get("fail", 0) for r in round_records)
        rollout_exception = sum(r.get("rollouts", {}).get("exception", 0) for r in round_records)
        rollout_total = sum(r.get("rollouts", {}).get("total", 0) for r in round_records)
        n_any_pass = sum(1 for r in round_records if r.get("any_pass"))
        n_all_pass = sum(1 for r in round_records if r.get("all_pass"))

        per_iteration.append({
            "iteration": iteration,
            "n_total": n_total,
            "n_any_pass": n_any_pass,
            "n_all_pass": n_all_pass,
            "any_pass_rate": (n_any_pass / n_total) if n_total else 0.0,
            "all_pass_rate": (n_all_pass / n_total) if n_total else 0.0,
            "rollouts": {
                "pass": rollout_pass,
                "fail": rollout_fail,
                "exception": rollout_exception,
                "total": rollout_total,
                "pass_rate": (rollout_pass / rollout_total) if rollout_total else 0.0,
            },
        })
        cumulative_by_iteration.append({
            "iteration": iteration,
            "k": iteration * rollout_k,
            "definition": "A task passes if any rollout up to and including this iteration passed.",
            "n_pass": cumulative_pass_tasks,
            "n_total": len(ok),
            "pass_rate": (cumulative_pass_tasks / len(ok)) if ok else 0.0,
        })

    ever_pass = [s for s in ok if s.get("ever_pass")]
    aggregate = {
        "mode": "adb_summary_sequential_reflection",
        "definition": (
            "No meta-agent code edits. ADB summarizes each task's own sanitized "
            "trajectories, and the task reuses that summary as context for later code-agent runs."
        ),
        "n_target_tasks": total_target,
        "n_reported_tasks": len(task_summaries),
        "n_completed": len(ok),
        "n_errors": len(errors),
        "max_iterations": max_iterations,
        "rollout_k": rollout_k,
        "evolution_pass_at_k": {
            "k": max_iterations * rollout_k,
            "n_pass": len(ever_pass),
            "n_total": len(ok),
            "pass_rate": (len(ever_pass) / len(ok)) if ok else 0.0,
        },
        "per_iteration": per_iteration,
        "cumulative_by_iteration": cumulative_by_iteration,
        "tasks": sorted(task_summaries, key=lambda s: s.get("task_name", "")),
    }

    with open(exp_dir / "seq_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    lines = [
        "# ADB Summary Sequential Reflection",
        "",
        f"- Tasks completed: {len(ok)}/{total_target}",
        f"- Iterations: {max_iterations}",
        f"- Rollout k per iteration: {rollout_k}",
        f"- Cumulative pass@{max_iterations * rollout_k}: {aggregate['evolution_pass_at_k']['pass_rate']:.1%}",
        "",
        "## Cumulative By Iteration",
        "",
    ]
    for item in cumulative_by_iteration:
        lines.append(f"- iter {item['iteration']} pass@{item['k']}: {item['pass_rate']:.1%} ({item['n_pass']}/{item['n_total']})")

    lines.extend(["", "## Per Task", ""])
    for item in aggregate["tasks"]:
        if item.get("error"):
            lines.append(f"- {item.get('task_name')}: ERROR - {item.get('error')}")
            continue
        final = item.get("final_iteration", {})
        lines.append(
            f"- {item['task_name']}: {'PASS' if item.get('ever_pass') else 'MISS'}, "
            f"first_pass={item.get('first_pass_iteration')}, "
            f"final_iter={final.get('iteration')}, final_any_pass={final.get('any_pass')}"
        )

    (exp_dir / "seq_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def run_seq_experiment(config: dict, config_path: str, *,
                       experiment_name: str | None = None,
                       start_iteration: int = 1,
                       skip_eval: bool = False) -> None:
    source_dir = resolve_source_dir(config)
    agent_config_filename = config["agent_config_filename"]
    max_iterations = int(config.get("max_iterations", 5))
    code_agent_patch = config.get("code_agent_patch", {})
    seq_cfg = _seq_cfg(config)

    exp_dir = create_seq_experiment_dir(config, config_path, experiment_name=experiment_name)
    task_names = _discover_per_task_names(config)
    rollout_k = int(config.get("harbor", {}).get("k", 1))
    n_task_workers = int(seq_cfg.get("n_concurrent_tasks", min(4, max(1, len(task_names)))))
    n_rollout_workers = int(
        seq_cfg.get(
            "n_concurrent_rollouts",
            max(1, min(rollout_k, int(config.get("harbor", {}).get("n_concurrent", rollout_k) or rollout_k))),
        )
    )
    adb_max_concurrent = max(
        1, int(config.get("agent_debugger", {}).get("max_concurrent", 16))
    )
    adb_semaphore = threading.BoundedSemaphore(adb_max_concurrent)

    print(f"\n{'=' * 60}")
    print("ADB summary sequential reflection")
    print(f"Experiment directory: {exp_dir.name}")
    print(f"Tasks: {len(task_names)}")
    print(f"Iterations per task: {max_iterations}")
    print(f"Rollout k per iteration: {rollout_k}")
    print(f"Concurrent task workers: {n_task_workers}")
    print(f"Concurrent rollouts per task: {n_rollout_workers}")
    print(f"Concurrent ADB processes: {adb_max_concurrent}")
    print(f"{'=' * 60}\n")

    _configure_adb_llm(config)

    results: list[dict] = []

    def worker(task_name: str) -> dict:
        try:
            return _run_one_seq_task(
                config=config,
                source_dir=source_dir,
                exp_dir=exp_dir,
                task_name=task_name,
                agent_config_filename=agent_config_filename,
                start_iteration=start_iteration,
                max_iterations=max_iterations,
                code_agent_patch=code_agent_patch,
                n_concurrent_rollouts=n_rollout_workers,
                skip_eval=skip_eval,
                adb_semaphore=adb_semaphore,
            )
        except Exception as exc:
            print(f"[seq] ERROR task={task_name}: {exc}", flush=True)
            return {
                "task_name": task_name,
                "safe_task_name": _safe_task_dir_name(task_name),
                "error": str(exc),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_task_workers) as pool:
        future_map = {pool.submit(worker, task_name): task_name for task_name in task_names}
        for future in concurrent.futures.as_completed(future_map):
            task_name = future_map[future]
            result = future.result()
            results.append(result)
            aggregate = _write_seq_aggregate(
                exp_dir,
                results,
                max_iterations=max_iterations,
                rollout_k=rollout_k,
                target_task_count=len(task_names),
            )
            status = "ERROR" if result.get("error") else ("PASS" if result.get("ever_pass") else "MISS")
            print(
                f"[seq] {task_name}: {status} ({len(results)}/{len(task_names)}) "
                f"pass@{max_iterations * rollout_k}={aggregate['evolution_pass_at_k']['pass_rate']:.1%}",
                flush=True,
            )

    aggregate = _write_seq_aggregate(
        exp_dir,
        results,
        max_iterations=max_iterations,
        rollout_k=rollout_k,
        target_task_count=len(task_names),
    )
    rate = aggregate["evolution_pass_at_k"]["pass_rate"]
    print(f"\n[seq] cumulative pass@{max_iterations * rollout_k}: {rate:.1%}")
    print(f"[seq] Summary: {exp_dir / 'seq_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run summary-only sequential per-task reflection")
    parser.add_argument("--config", required=True, help="Config yaml")
    parser.add_argument("--experiment", default=None, help="Reuse/write a specific experiments/ directory")
    parser.add_argument("--start-iteration", type=int, default=1, help="Start iteration for resume")
    parser.add_argument("--skip-eval", action="store_true", help="Reuse latest existing eval job at start iteration")
    args = parser.parse_args()

    config = load_config(args.config)
    run_seq_experiment(
        config,
        args.config,
        experiment_name=args.experiment,
        start_iteration=args.start_iteration,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()
