#!/usr/bin/env python3
"""Blindly select one rollout from a pass@k baseline experiment.

The selector copies only code-agent trajectory files into a separate blind view,
then asks the evolve agent to choose the most promising rollout. Verifier,
reward, and test-output files are intentionally not copied into the view.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from evolve import (
    EVOLVE_AGENT_DIR,
    PROJECT_DIR,
    apply_agent_yaml_patch,
    build_evolve_agent_patch,
    load_config,
    run_evolve_agent,
)


SAFE_TRACE_FILES = (
    "nexau.txt",
    "nexau_in_memory_tracer.cleaned.json",
    "nexau_in_memory_tracer.json",
)

DEFAULT_OUTPUT_ROOT = "blind_rollout_selector"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _redact_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted: dict[str, Any] = {}
        for key, value in obj.items():
            if "api_key" in key.lower() or key.lower().endswith("token"):
                redacted[key] = "***REDACTED***" if value else value
            else:
                redacted[key] = _redact_secrets(value)
        return redacted
    if isinstance(obj, list):
        return [_redact_secrets(item) for item in obj]
    return obj


def _safe_task_name(task_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name).strip("_") or "task"


def _find_task_dirs(exp_dir: Path, selected_tasks: set[str] | None = None) -> list[Path]:
    tasks_root = exp_dir / "tasks"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"Expected baseline task directory: {tasks_root}")

    task_dirs = sorted(p for p in tasks_root.iterdir() if p.is_dir())
    if selected_tasks:
        task_dirs = [p for p in task_dirs if p.name in selected_tasks]
    return task_dirs


def _find_rollout_dirs(task_dir: Path, expected_k: int | None = None) -> list[Path]:
    """Find rollout directories by looking for agent trace files."""

    parents: dict[str, Path] = {}
    for file_name in SAFE_TRACE_FILES:
        for trace_path in task_dir.rglob(file_name):
            parts = {part.lower() for part in trace_path.parts}
            if DEFAULT_OUTPUT_ROOT in parts:
                continue
            parent = trace_path.parent
            parents[str(parent.resolve())] = parent

    rollout_dirs = sorted(parents.values(), key=lambda p: str(p))
    if expected_k and len(rollout_dirs) > expected_k:
        rollout_dirs = rollout_dirs[:expected_k]
    return rollout_dirs


def _copy_blind_rollout_view(rollout_dirs: list[Path], view_dir: Path) -> dict[str, str]:
    """Copy agent-only files into rollout_XX folders and return private path map."""

    if view_dir.exists():
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)

    path_map: dict[str, str] = {}
    for idx, rollout_dir in enumerate(rollout_dirs, start=1):
        label = f"rollout_{idx:02d}"
        dst = view_dir / label
        dst.mkdir(parents=True, exist_ok=True)
        path_map[label] = str(rollout_dir)

        copied = 0
        for file_name in SAFE_TRACE_FILES:
            src = rollout_dir / file_name
            if src.exists() and src.is_file():
                shutil.copy2(src, dst / file_name)
                copied += 1

        if copied == 0:
            (dst / "README.md").write_text(
                "No supported code-agent trace files were found for this rollout.\n",
                encoding="utf-8",
            )

    (view_dir / "README.md").write_text(
        "\n".join(
            [
                "# Blind Rollout View",
                "",
                "Each `rollout_XX` directory contains only code-agent trajectory files.",
                "Verifier, reward, pass/fail, and test-output files were deliberately omitted.",
                "Choose exactly one rollout based only on the visible agent behavior.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path_map


def _selector_prompt(task_name: str, k: int) -> str:
    return f"""You are a blind rollout selector for a Terminal-Bench code-agent baseline.

Task: {task_name}

You are working in a copied blind view containing {k} rollout directories named `rollout_01`, `rollout_02`, etc.

Rules:
- Do not use verifier output, reward files, pass/fail labels, test stdout, CTRF, or benchmark result files.
- The visible view should contain only code-agent traces such as `nexau.txt` and `nexau_in_memory_tracer.cleaned.json`.
- Read and compare all available rollout traces.
- Choose the single rollout that is most likely to be correct and complete.
- Prefer evidence from concrete commands run, files edited, final state, error handling, and whether the agent appears to have finished the task.
- If traces are incomplete, still choose the most promising rollout and explain the uncertainty.

Return only a JSON object with this schema:
{{
  "selected_rollout": "rollout_01",
  "confidence": 0.0,
  "rationale": "short explanation",
  "evidence": ["specific visible evidence"],
  "risks": ["specific uncertainty or concern"]
}}
"""


def _extract_json_object(raw: Any) -> dict[str, Any]:
    """Extract the selector JSON, unwrapping Nexau completion envelopes."""

    if isinstance(raw, dict):
        output = raw.get("output")
        if isinstance(output, dict) and isinstance(output.get("result"), str):
            inner = _extract_json_object(output["result"])
            if inner.get("selected_rollout") is not None:
                return inner
        if raw.get("selected_rollout") is not None:
            return raw
        return raw

    raw = (raw or "").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return _extract_json_object(parsed)
        if isinstance(parsed, str) and parsed != raw:
            return _extract_json_object(parsed)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {"selected_rollout": None, "raw_response": raw}
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return _extract_json_object(parsed)
        if isinstance(parsed, str):
            return _extract_json_object(parsed)
    except json.JSONDecodeError:
        pass
    return {"selected_rollout": None, "raw_response": raw}


def _normalize_rollout_label(value: Any, k: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    idx = int(match.group(1))
    if idx < 1 or idx > k:
        return None
    return f"rollout_{idx:02d}"


def _load_existing_selection(task_out: Path, task_name: str, k: int) -> dict[str, Any] | None:
    selection_path = task_out / "selection.json"
    if not selection_path.exists():
        return None

    try:
        selection = _read_json(selection_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[selector] Ignoring unreadable prior selection for {task_name}: {exc}", flush=True)
        return None

    if selection.get("task_name") != task_name:
        print(f"[selector] Ignoring mismatched prior selection for {task_name}", flush=True)
        return None
    if selection.get("error"):
        print(f"[selector] Re-running prior errored selection for {task_name}", flush=True)
        return None

    selected = _normalize_rollout_label(selection.get("selected_rollout"), k)
    available = selection.get("available_rollouts") or []
    expected_available = [f"rollout_{i:02d}" for i in range(1, k + 1)]
    if selected is None:
        print(f"[selector] Ignoring invalid prior selection for {task_name}", flush=True)
        return None
    if available and available != expected_available:
        print(f"[selector] Ignoring prior selection with stale rollout list for {task_name}", flush=True)
        return None
    if available and selected not in available:
        print(f"[selector] Ignoring prior selection outside available rollouts for {task_name}", flush=True)
        return None

    selection["selected_rollout"] = selected
    return selection


def _write_selector_summary(output_root: Path, baseline_dir: Path, expected_k: int,
                            results: list[dict[str, Any]]) -> None:
    summary = {
        "mode": "blind_rollout_selector",
        "definition": "Select one rollout using only code-agent trajectory files; verifier/reward/test outputs are hidden from the selector.",
        "baseline_experiment_dir": str(baseline_dir),
        "output_dir": str(output_root),
        "k": expected_k,
        "n_tasks": len(results),
        "n_selected": sum(1 for item in results if item.get("selected_rollout")),
        "tasks": sorted(results, key=lambda item: item.get("task_name", "")),
    }
    with open(output_root / "blind_rollout_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _write_error_selection(task_out: Path, task_name: str, error: str) -> dict[str, Any]:
    result = {
        "task_name": task_name,
        "error": error,
        "selected_rollout": None,
    }
    with open(task_out / "selection.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _provider_config(args: argparse.Namespace) -> dict[str, Any]:
    provider = (args.provider or "").lower()
    model = args.model
    base_url = args.base_url
    api_key = args.api_key or (os.environ.get(args.api_key_env) if args.api_key_env else None)

    if provider in {"gpt", "gpt54", "openai"}:
        model = model or "gpt-5.4"
        base_url = base_url or os.environ.get("GPT54_LLM_BASE_URL") or "https://api.openai.com/v1"
        api_key = api_key or os.environ.get("GPT54_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return {
            "llm": {"api_key": api_key, "base_url": base_url, "model": model},
            "evolve_agent": {
                "tool_call_mode": "openai",
                "llm_config": {
                    "api_key": api_key,
                    "base_url": base_url,
                    "model": model,
                    "api_type": "openai_responses",
                    "reasoning": {"effort": "high", "summary": "detailed"},
                    "stream": True,
                },
            },
        }

    if provider in {"claude", "anthropic", "opus46"}:
        model = model or "claude-opus-4-6"
        base_url = base_url or "https://api.anthropic.com"
        api_key = api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        return {
            "llm": {"api_key": api_key, "base_url": base_url, "model": model},
            "evolve_agent": {
                "tool_call_mode": "anthropic",
                "llm_config": {
                    "api_key": api_key,
                    "base_url": base_url,
                    "model": model,
                    "api_type": "anthropic_chat_completion",
                    "max_tokens": int(args.max_tokens),
                    "reasoning": None,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": args.effort},
                    "stream": True,
                },
            },
        }

    raise ValueError("--provider must be one of: gpt54, claude")


def _load_selector_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        config = load_config(args.config)
        if args.provider:
            override = _provider_config(args)
            config["llm"] = override["llm"]
            config["evolve_agent"] = override["evolve_agent"]
        return config
    return _provider_config(args)


def _prepare_selector_agent(selector_root: Path, config: dict[str, Any]) -> None:
    agent_dst = selector_root / "evolve_agent"
    if not agent_dst.exists():
        shutil.copytree(EVOLVE_AGENT_DIR, agent_dst)

    patch = build_evolve_agent_patch(config.get("evolve_agent", {}))
    apply_agent_yaml_patch(agent_dst / "evolve_agent.yaml", patch, label="blind_selector_evolve_agent")


def run_selector(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.experiment_dir).resolve()
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline experiment directory not found: {baseline_dir}")

    config = _load_selector_config(args)
    baseline_summary_path = baseline_dir / "baseline_summary.json"
    baseline_summary = _read_json(baseline_summary_path) if baseline_summary_path.exists() else {}
    expected_k = int(args.k or baseline_summary.get("k") or 5)

    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    selector_name = args.name or f"{timestamp}__{args.provider or 'config'}"
    output_root = Path(args.output_dir).resolve() if args.output_dir else baseline_dir / DEFAULT_OUTPUT_ROOT / selector_name
    output_root.mkdir(parents=True, exist_ok=True)
    private_root = output_root / "_private"
    private_root.mkdir(exist_ok=True)

    snapshot = {
        "baseline_experiment_dir": str(baseline_dir),
        "expected_k": expected_k,
        "selector_config": _redact_secrets(config),
    }
    with open(output_root / "selector_config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True)

    _prepare_selector_agent(output_root, config)

    selected_tasks = set(args.task or []) or None
    task_dirs = _find_task_dirs(baseline_dir, selected_tasks)
    if args.limit:
        task_dirs = task_dirs[: int(args.limit)]

    results_by_task: dict[str, dict[str, Any]] = {}
    private_map: dict[str, dict[str, str]] = {}

    for task_index, task_dir in enumerate(task_dirs, start=1):
        task_name = task_dir.name
        print(f"[selector] Task {task_index}/{len(task_dirs)}: {task_name}", flush=True)
        task_out = output_root / "tasks" / _safe_task_name(task_name)
        task_out.mkdir(parents=True, exist_ok=True)
        view_dir = task_out / "blind_view"

        rollout_dirs = _find_rollout_dirs(task_dir, expected_k)
        path_map = {f"rollout_{idx:02d}": str(rollout_dir) for idx, rollout_dir in enumerate(rollout_dirs, start=1)}
        private_map[task_name] = path_map

        existing = _load_existing_selection(task_out, task_name, len(rollout_dirs))
        if existing is not None:
            results_by_task[task_name] = existing
            print(f"[selector] Reusing prior selection for {task_name}: {existing['selected_rollout']}", flush=True)
            _write_selector_summary(output_root, baseline_dir, expected_k, list(results_by_task.values()))
            continue

        path_map = _copy_blind_rollout_view(rollout_dirs, view_dir)
        private_map[task_name] = path_map

        if not rollout_dirs:
            result = _write_error_selection(task_out, task_name, "No rollout trace directories found")
            results_by_task[task_name] = result
            _write_selector_summary(output_root, baseline_dir, expected_k, list(results_by_task.values()))
            continue

        prompt = _selector_prompt(task_name, len(rollout_dirs))
        (task_out / "selector_prompt.md").write_text(prompt, encoding="utf-8")
        selector_job_dir = task_out / "selector_job"
        selector_job_dir.mkdir(exist_ok=True)

        print(f"[selector] Running model selection for {task_name}", flush=True)
        try:
            raw_response = run_evolve_agent(
                config,
                output_root,
                iteration=1,
                query=prompt,
                job_dir=selector_job_dir,
                iteration_dir=task_out,
                work_dir=view_dir,
                log_dir=task_out / "selector_logs",
            )
        except Exception as exc:
            result = _write_error_selection(task_out, task_name, str(exc))
            results_by_task[task_name] = result
            _write_selector_summary(output_root, baseline_dir, expected_k, list(results_by_task.values()))
            print(f"[selector] ERROR {task_name}: {exc}", flush=True)
            continue

        parsed = _extract_json_object(raw_response)
        selected = _normalize_rollout_label(parsed.get("selected_rollout"), len(rollout_dirs))
        if selected is None:
            selected = "rollout_01"
            parsed["selection_normalization_warning"] = "Invalid selected_rollout; defaulted to rollout_01"

        result = {
            "task_name": task_name,
            "selected_rollout": selected,
            "available_rollouts": [f"rollout_{i:02d}" for i in range(1, len(rollout_dirs) + 1)],
            "selection": parsed,
            "raw_response": raw_response,
            "blind_view": str(view_dir),
        }
        results_by_task[task_name] = result
        with open(task_out / "selection.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _write_selector_summary(output_root, baseline_dir, expected_k, list(results_by_task.values()))
        print(f"[selector] Selected {task_name}: {selected}", flush=True)

    with open(private_root / "rollout_path_map.json", "w", encoding="utf-8") as f:
        json.dump(private_map, f, ensure_ascii=False, indent=2)

    results = list(results_by_task.values())
    _write_selector_summary(output_root, baseline_dir, expected_k, results)

    print(f"[selector] Wrote {len(results)} selections to {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, help="Baseline pass@k experiment directory")
    parser.add_argument("--config", default=None, help="Optional config file to reuse LLM/evolve-agent settings")
    parser.add_argument("--provider", default="gpt54", help="Selector provider shortcut: gpt54 or claude")
    parser.add_argument("--model", default=None, help="Override selector model")
    parser.add_argument("--base-url", default=None, help="Override selector base URL")
    parser.add_argument("--api-key", default=None, help="Direct API key override")
    parser.add_argument("--api-key-env", default=None, help="Environment variable containing the API key")
    parser.add_argument("--max-tokens", type=int, default=128000, help="Claude max_tokens when using Anthropic")
    parser.add_argument("--effort", default="high", help="Claude output_config.effort")
    parser.add_argument("--k", type=int, default=None, help="Expected rollouts per task; defaults to baseline_summary.k")
    parser.add_argument("--task", action="append", default=None, help="Restrict to a task name; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tasks")
    parser.add_argument("--output-dir", default=None, help="Output directory; default is under the baseline experiment")
    parser.add_argument("--name", default=None, help="Selector run name under the default output directory")
    args = parser.parse_args()
    run_selector(args)


if __name__ == "__main__":
    main()
