#!/usr/bin/env python3
"""Run code-agent-only pass@k evaluation without any evolution.

This runner intentionally stays separate from evolve.py's main loop:
it initializes one independent code-agent workspace per task, launches
Harbor with k rollouts for that task, and writes baseline pass@k summaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

from evolve import (
    EXPERIMENTS_DIR,
    PROJECT_DIR,
    _discover_per_task_names,
    _round_record_for_task,
    _safe_task_dir_name,
    _single_task_config,
    apply_code_agent_patch,
    compute_stats,
    deep_merge,
    init_workspace,
    load_config,
    resolve_source_dir,
    run_harbor,
)


def create_baseline_experiment_dir(config: dict, config_path: str,
                                   experiment_name: str | None = None,
                                   output_dir: str | None = None) -> Path:
    """Create a baseline experiment directory without copying evolve_agent."""
    if output_dir:
        exp_dir = Path(output_dir).expanduser()
        if not exp_dir.is_absolute():
            exp_dir = (PROJECT_DIR / exp_dir).resolve()
    elif experiment_name:
        exp_dir = EXPERIMENTS_DIR / experiment_name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        meta_name = config.get("_meta", {}).get("_name", "")
        dir_name = f"{timestamp}__{meta_name}" if meta_name else f"{timestamp}__code-agent-baseline"
        exp_dir = EXPERIMENTS_DIR / dir_name

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

    print(f"[baseline] Experiment directory: {exp_dir}")
    return exp_dir


def _baseline_cfg(config: dict) -> dict:
    """Return baseline settings with compatibility fallback to per_task_evolution."""
    fallback = config.get("per_task_evolution", {})
    return deep_merge(fallback, config.get("code_agent_baseline", {}))


def _pass_at_for_task(stats: dict, task_name: str, k: int, record: dict) -> dict[str, float]:
    """Return pass@i estimates for one task.

    With exactly k rollouts, pass@k is equivalent to "any rollout passed".
    pass@1 is the sampled per-rollout success fraction.
    """
    if k > 1:
        per_task = stats.get("pass_at_k", {}).get("per_task_pass_at", {}).get(task_name, {})
        return {str(i): float(per_task.get(i, 0.0)) for i in range(1, k + 1)}

    return {"1": 1.0 if record.get("any_pass") else 0.0}


def _write_task_summary(task_root: Path, summary: dict) -> None:
    with open(task_root / "baseline_task_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _load_completed_task_summary(exp_dir: Path, task_name: str, k: int) -> dict | None:
    """Load a completed baseline task summary that can be reused for resume."""
    safe_task = _safe_task_dir_name(task_name)
    summary_path = exp_dir / "tasks" / safe_task / "baseline_task_summary.json"
    if not summary_path.exists():
        return None

    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[baseline] Ignoring unreadable prior summary for {task_name}: {exc}", flush=True)
        return None

    if summary.get("task_name") != task_name:
        print(f"[baseline] Ignoring mismatched prior summary for {task_name}", flush=True)
        return None
    try:
        previous_k = int(summary.get("k", -1))
    except (TypeError, ValueError):
        previous_k = -1
    if previous_k != int(k):
        print(
            f"[baseline] Ignoring prior summary for {task_name}: "
            f"k={summary.get('k')} does not match current k={k}",
            flush=True,
        )
        return None
    if summary.get("error"):
        print(f"[baseline] Re-running prior errored task {task_name}", flush=True)
        return None
    if not summary.get("rollouts") or not summary.get("pass_at"):
        print(f"[baseline] Ignoring incomplete prior summary for {task_name}", flush=True)
        return None

    return summary


def _write_baseline_aggregate(exp_dir: Path, task_summaries: list[dict], k: int,
                              target_task_count: int | None = None) -> dict:
    ok = [s for s in task_summaries if not s.get("error")]
    errors = [s for s in task_summaries if s.get("error")]
    reported = len(task_summaries)
    target_total = int(target_task_count or reported)

    n_any_pass = sum(1 for s in ok if s.get("any_pass"))
    n_all_pass = sum(1 for s in ok if s.get("all_pass"))
    rollout_pass = sum(s.get("rollouts", {}).get("pass", 0) for s in ok)
    rollout_fail = sum(s.get("rollouts", {}).get("fail", 0) for s in ok)
    rollout_exception = sum(s.get("rollouts", {}).get("exception", 0) for s in ok)
    rollout_total = sum(s.get("rollouts", {}).get("total", 0) for s in ok)

    pass_at: dict[str, float] = {}
    eligible_counts: dict[str, int] = {}
    for i in range(1, k + 1):
        key = str(i)
        vals = [float(s.get("pass_at", {}).get(key, 0.0)) for s in ok if key in s.get("pass_at", {})]
        eligible_counts[key] = len(vals)
        pass_at[key] = sum(vals) / len(vals) if vals else 0.0

    aggregate = {
        "mode": "code_agent_baseline",
        "definition": "No evolution. Each task starts from the configured source code-agent prompt/workspace.",
        "k": k,
        "n_target_tasks": target_total,
        "n_reported_tasks": reported,
        "n_completed": len(ok),
        "n_errors": len(errors),
        "n_any_pass": n_any_pass,
        "n_all_pass": n_all_pass,
        "empirical_pass_at_k": (n_any_pass / len(ok)) if ok else 0.0,
        "all_pass_rate": (n_all_pass / len(ok)) if ok else 0.0,
        "pass_at": pass_at,
        "pass_at_eligible_counts": eligible_counts,
        "rollouts": {
            "pass": rollout_pass,
            "fail": rollout_fail,
            "exception": rollout_exception,
            "total": rollout_total,
            "trial_pass_rate": (rollout_pass / rollout_total) if rollout_total else 0.0,
        },
        "tasks": sorted(task_summaries, key=lambda s: s.get("task_name", "")),
    }

    with open(exp_dir / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    lines = [
        "# Code Agent Baseline Summary",
        "",
        f"- Mode: no evolution",
        f"- Tasks completed: {len(ok)}/{target_total}",
        f"- k: {k}",
        f"- Empirical pass@{k}: {aggregate['empirical_pass_at_k']:.1%}",
        f"- Trial pass rate: {aggregate['rollouts']['trial_pass_rate']:.1%}",
    ]
    if pass_at:
        rates = " | ".join(f"pass@{i}={pass_at[str(i)]:.1%}" for i in range(1, k + 1))
        lines.append(f"- Estimated pass@i: {rates}")
    if errors:
        lines.append(f"- Task errors: {len(errors)}")

    lines.extend(["", "## Per Task", ""])
    for item in aggregate["tasks"]:
        if item.get("error"):
            lines.append(f"- {item.get('task_name')}: ERROR - {item.get('error')}")
            continue
        status = "PASS" if item.get("any_pass") else "MISS"
        ro = item.get("rollouts", {})
        lines.append(
            f"- {item['task_name']}: {status}, "
            f"rollouts pass/fail/exception={ro.get('pass', 0)}/{ro.get('fail', 0)}/{ro.get('exception', 0)}, "
            f"job={item.get('job_dir')}"
        )

    (exp_dir / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def _run_one_task(*, config: dict, source_dir: Path, exp_dir: Path, task_name: str,
                  agent_config_filename: str, code_agent_patch: dict,
                  k: int, n_concurrent_rollouts: int) -> dict:
    safe_task = _safe_task_dir_name(task_name)
    task_root = exp_dir / "tasks" / safe_task
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "task_name.txt").write_text(task_name + "\n", encoding="utf-8")

    workspace_dir = task_root / "workspace"
    is_new = init_workspace(source_dir, workspace_dir)
    if is_new:
        apply_code_agent_patch(workspace_dir, agent_config_filename, code_agent_patch)

    task_config = _single_task_config(config, task_name, n_concurrent_rollouts)
    task_config["harbor"] = copy.deepcopy(task_config.get("harbor", {}))
    task_config["harbor"]["k"] = k
    task_config["harbor"]["n_concurrent"] = max(1, int(n_concurrent_rollouts))

    benchmark_dir = task_root / "runs" / "baseline" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    job_dir = run_harbor(task_config, workspace_dir, agent_config_filename, benchmark_dir)
    stats = compute_stats(job_dir, k=k)
    record = _round_record_for_task(task_name, 1, stats, job_dir, task_root)
    pass_at = _pass_at_for_task(stats, task_name, k, record)

    summary = {
        "task_name": task_name,
        "safe_task_name": safe_task,
        "k": k,
        "result": record["result"],
        "any_pass": record["any_pass"],
        "all_pass": record["all_pass"],
        "rollouts": record["rollouts"],
        "pass_at": pass_at,
        "job_dir": record["job_dir"],
        "timing": {
            "eval_min": round((time.monotonic() - started) / 60, 1),
        },
    }
    _write_task_summary(task_root, summary)
    return summary


def run_baseline(config: dict, config_path: str, *,
                 experiment_name: str | None = None,
                 output_dir: str | None = None,
                 k_override: int | None = None,
                 n_task_workers_override: int | None = None,
                 n_rollout_workers_override: int | None = None) -> None:
    source_dir = resolve_source_dir(config)
    agent_config_filename = config["agent_config_filename"]
    code_agent_patch = config.get("code_agent_patch", {})
    baseline_cfg = _baseline_cfg(config)

    exp_dir = create_baseline_experiment_dir(
        config,
        config_path,
        experiment_name=experiment_name,
        output_dir=output_dir,
    )
    task_names = _discover_per_task_names(config)
    k = int(k_override or config.get("harbor", {}).get("k", 1))
    n_task_workers = int(
        n_task_workers_override
        or baseline_cfg.get("n_concurrent_tasks", min(4, max(1, len(task_names))))
    )
    n_rollout_workers = int(
        n_rollout_workers_override
        or baseline_cfg.get(
            "n_concurrent_rollouts",
            max(1, min(k, int(config.get("harbor", {}).get("n_concurrent", k) or k))),
        )
    )

    print(f"\n{'=' * 60}")
    print("Code-agent-only baseline evaluation")
    print(f"Experiment directory: {exp_dir.name}")
    print(f"Tasks: {len(task_names)}")
    print(f"Rollouts per task (k): {k}")
    print(f"Concurrent task workers: {n_task_workers}")
    print(f"Concurrent rollouts per task: {n_rollout_workers}")
    print(f"Max trial concurrency estimate: {n_task_workers * n_rollout_workers}")
    print(f"{'=' * 60}\n")

    results_by_task: dict[str, dict] = {}
    pending_task_names: list[str] = []
    for task_name in task_names:
        existing = _load_completed_task_summary(exp_dir, task_name, k)
        if existing is None:
            pending_task_names.append(task_name)
        else:
            results_by_task[task_name] = existing

    if results_by_task:
        _write_baseline_aggregate(
            exp_dir,
            list(results_by_task.values()),
            k,
            target_task_count=len(task_names),
        )
        print(
            f"[baseline] Resuming with {len(results_by_task)} completed task(s); "
            f"{len(pending_task_names)} task(s) remaining.",
            flush=True,
        )

    def worker(task_name: str) -> dict:
        try:
            return _run_one_task(
                config=config,
                source_dir=source_dir,
                exp_dir=exp_dir,
                task_name=task_name,
                agent_config_filename=agent_config_filename,
                code_agent_patch=code_agent_patch,
                k=k,
                n_concurrent_rollouts=n_rollout_workers,
            )
        except Exception as exc:
            print(f"[baseline] ERROR task={task_name}: {exc}", flush=True)
            return {
                "task_name": task_name,
                "safe_task_name": _safe_task_dir_name(task_name),
                "error": str(exc),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_task_workers) as pool:
        future_map = {pool.submit(worker, task_name): task_name for task_name in pending_task_names}
        for future in concurrent.futures.as_completed(future_map):
            task_name = future_map[future]
            result = future.result()
            results_by_task[task_name] = result
            aggregate = _write_baseline_aggregate(
                exp_dir,
                list(results_by_task.values()),
                k,
                target_task_count=len(task_names),
            )
            status = "ERROR" if result.get("error") else ("PASS" if result.get("any_pass") else "MISS")
            print(
                f"[baseline] {task_name}: {status} ({len(results_by_task)}/{len(task_names)}) "
                f"pass@{k}={aggregate['empirical_pass_at_k']:.1%}",
                flush=True,
            )

    aggregate = _write_baseline_aggregate(
        exp_dir,
        list(results_by_task.values()),
        k,
        target_task_count=len(task_names),
    )
    print(f"\n[baseline] empirical pass@{k}: {aggregate['empirical_pass_at_k']:.1%}")
    print(f"[baseline] Summary: {exp_dir / 'baseline_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run code-agent-only pass@k baseline evaluation")
    parser.add_argument("--config", required=True, help="Experiment config yaml")
    parser.add_argument("--experiment", default=None, help="Reuse/write a specific experiments/ directory name")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write results to an explicit directory path instead of experiments/<name>",
    )
    parser.add_argument("--k", type=int, default=None, help="Override harbor.k rollouts per task")
    parser.add_argument("--n-concurrent-tasks", type=int, default=None, help="Override task-level parallelism")
    parser.add_argument("--n-concurrent-rollouts", type=int, default=None, help="Override rollout parallelism per task")
    args = parser.parse_args()

    config = load_config(args.config)
    run_baseline(
        config,
        args.config,
        experiment_name=args.experiment,
        output_dir=args.output_dir,
        k_override=args.k,
        n_task_workers_override=args.n_concurrent_tasks,
        n_rollout_workers_override=args.n_concurrent_rollouts,
    )


if __name__ == "__main__":
    main()
