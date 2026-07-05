#!/usr/bin/env python3
"""Audit blind rollout-selector choices against baseline pass/fail results.

The selector itself must not see verifier or pass/fail data. This script is a
post-hoc audit: it loads the selector output and the baseline summary, then
checks whether each selected rollout actually passed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PASS_KEYS = (
    "passed",
    "pass",
    "success",
    "succeeded",
    "is_resolved",
    "resolved",
    "is_success",
)

STATUS_KEYS = (
    "status",
    "result",
    "outcome",
    "trial_status",
    "verdict",
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _extract_json_object(raw: Any) -> dict[str, Any]:
    """Extract JSON from plain text or Nexau's completion envelope."""

    if isinstance(raw, dict):
        output = raw.get("output")
        if isinstance(output, dict) and isinstance(output.get("result"), str):
            inner = _extract_json_object(output["result"])
            if inner.get("selected_rollout") is not None:
                return inner
        if raw.get("selected_rollout") is not None:
            return raw
        return raw

    raw = str(raw or "").strip()
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
        return {}
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return _extract_json_object(parsed)
        if isinstance(parsed, str):
            return _extract_json_object(parsed)
    except json.JSONDecodeError:
        pass
    return {}


def _resolve_selection_summary(path: str) -> Path:
    p = Path(path).resolve()
    if p.is_dir():
        p = p / "blind_rollout_selection_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Selection summary not found: {p}")
    return p


def _infer_baseline_summary(selection: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if p.is_dir():
            p = p / "baseline_summary.json"
    else:
        baseline_dir = selection.get("baseline_experiment_dir")
        if not baseline_dir:
            raise ValueError("Selection summary lacks baseline_experiment_dir; pass --baseline-summary")
        p = Path(str(baseline_dir)).resolve() / "baseline_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Baseline summary not found: {p}")
    return p


def _task_key(record: dict[str, Any]) -> str | None:
    for key in ("task_name", "safe_task_name", "name", "task"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _baseline_tasks(baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    raw_tasks = baseline.get("tasks")

    if isinstance(raw_tasks, list):
        for record in raw_tasks:
            if isinstance(record, dict):
                for key in ("task_name", "safe_task_name", "name", "task"):
                    value = record.get(key)
                    if isinstance(value, str) and value:
                        tasks[value] = record
    elif isinstance(raw_tasks, dict):
        for name, record in raw_tasks.items():
            if isinstance(record, dict):
                tasks[str(name)] = record
                task_name = _task_key(record)
                if task_name:
                    tasks[task_name] = record

    return tasks


def _rollout_index(label: Any) -> int | None:
    match = re.search(r"(\d+)", str(label or ""))
    if not match:
        return None
    return int(match.group(1))


def _selected_rollout(item: dict[str, Any]) -> str | None:
    """Return the actual selector choice, recovering old wrapped outputs."""

    selection = item.get("selection")
    if isinstance(selection, dict):
        parsed = _extract_json_object(selection)
        selected = parsed.get("selected_rollout")
        if selected:
            return str(selected)

        raw_selection = selection.get("raw_response")
        parsed = _extract_json_object(raw_selection)
        selected = parsed.get("selected_rollout")
        if selected:
            return str(selected)

    parsed = _extract_json_object(item.get("raw_response"))
    selected = parsed.get("selected_rollout")
    if selected:
        return str(selected)

    selected = item.get("selected_rollout")
    return str(selected) if selected else None


def _pass_from_scalar(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"pass", "passed", "success", "succeeded", "resolved", "true", "1"}:
            return True
        if lowered in {"fail", "failed", "failure", "error", "false", "0", "unresolved"}:
            return False
    return None


def _extract_pass(record: Any) -> bool | None:
    scalar = _pass_from_scalar(record)
    if scalar is not None:
        return scalar

    if not isinstance(record, dict):
        return None

    for key in PASS_KEYS:
        if key in record:
            scalar = _pass_from_scalar(record[key])
            if scalar is not None:
                return scalar

    for key in STATUS_KEYS:
        if key in record:
            scalar = _pass_from_scalar(record[key])
            if scalar is not None:
                return scalar

    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key in PASS_KEYS + ("score", "reward"):
            if key in metrics:
                scalar = _pass_from_scalar(metrics[key])
                if scalar is not None:
                    return scalar

    return None


def _rollout_candidates(task_record: dict[str, Any]) -> Any:
    for key in ("rollouts", "rollout_results", "trials", "trial_results", "attempts"):
        value = task_record.get(key)
        if isinstance(value, (dict, list)):
            return value
    return None


def _rollout_record(task_record: dict[str, Any], selected_label: str) -> Any:
    idx = _rollout_index(selected_label)
    rollouts = _rollout_candidates(task_record)

    if isinstance(rollouts, list):
        if idx is not None and 1 <= idx <= len(rollouts):
            return rollouts[idx - 1]
        return None

    if isinstance(rollouts, dict):
        keys = [
            selected_label,
            selected_label.replace("_0", "_"),
            selected_label.replace("rollout_", ""),
            f"rollout_{idx}" if idx is not None else None,
            str(idx) if idx is not None else None,
        ]
        for key in keys:
            if key is not None and key in rollouts:
                return rollouts[key]

        if idx is not None:
            numeric_items: list[tuple[int, Any]] = []
            for key, value in rollouts.items():
                parsed = _rollout_index(key)
                if parsed is not None:
                    numeric_items.append((parsed, value))
            numeric_items.sort(key=lambda item: item[0])
            if 1 <= idx <= len(numeric_items):
                return numeric_items[idx - 1][1]

    return None


def _rollout_passes(task_record: dict[str, Any]) -> list[bool | None]:
    rollouts = _rollout_candidates(task_record)
    values: list[Any]
    if isinstance(rollouts, list):
        values = rollouts
    elif isinstance(rollouts, dict):
        values = [
            value
            for _, value in sorted(
                rollouts.items(),
                key=lambda item: (_rollout_index(item[0]) is None, _rollout_index(item[0]) or 0, str(item[0])),
            )
        ]
    else:
        values = []
    return [_extract_pass(value) for value in values]


def _load_private_path_map(selection_summary: Path) -> dict[str, dict[str, str]]:
    path = selection_summary.parent / "_private" / "rollout_path_map.json"
    if not path.exists():
        return {}
    data = _read_json(path)
    return {
        str(task): {str(label): str(raw_path) for label, raw_path in labels.items()}
        for task, labels in data.items()
        if isinstance(labels, dict)
    }


def _path_candidates(raw_path: str) -> list[Path]:
    candidates = [Path(raw_path)]
    if raw_path.startswith("/mnt/c/"):
        candidates.append(Path("C:/") / raw_path[len("/mnt/c/"):])
    return candidates


def _selected_rollout_record_from_path(raw_path: str | None) -> dict[str, Any] | None:
    if not raw_path:
        return None
    for path in _path_candidates(raw_path):
        run_dir = path.parent if path.name == "agent" else path
        result_json = run_dir / "result.json"
        if result_json.exists():
            return _read_json(result_json)
    return None


def _extract_pass_from_result_json(result: dict[str, Any] | None) -> bool | None:
    if not result:
        return None
    if result.get("exception_info"):
        return False

    verifier = result.get("verifier_result")
    if isinstance(verifier, dict):
        rewards = verifier.get("rewards")
        if isinstance(rewards, dict) and "reward" in rewards:
            try:
                return float(rewards["reward"]) > 0
            except (TypeError, ValueError):
                return None

    return _extract_pass(result)


def audit(selection_summary: Path, baseline_summary: Path, output_path: Path | None = None) -> dict[str, Any]:
    selection = _read_json(selection_summary)
    baseline = _read_json(baseline_summary)
    tasks_by_name = _baseline_tasks(baseline)
    private_path_map = _load_private_path_map(selection_summary)

    task_results: list[dict[str, Any]] = []
    for item in selection.get("tasks", []):
        if not isinstance(item, dict):
            continue
        task_name = item.get("task_name")
        selected = _selected_rollout(item)
        task_record = tasks_by_name.get(str(task_name)) if task_name is not None else None

        raw_path = private_path_map.get(str(task_name), {}).get(str(selected))
        selected_result = _selected_rollout_record_from_path(raw_path)
        selected_pass = _extract_pass_from_result_json(selected_result)

        rollout_records = []
        for label in sorted(private_path_map.get(str(task_name), {}), key=_rollout_index):
            rollout_records.append(_selected_rollout_record_from_path(private_path_map[str(task_name)][label]))
        rollout_passes = [_extract_pass_from_result_json(record) for record in rollout_records]
        if not rollout_passes and task_record:
            rollout_record = _rollout_record(task_record, str(selected))
            selected_pass = _extract_pass(rollout_record)
            rollout_passes = _rollout_passes(task_record)
        any_pass = any(value is True for value in rollout_passes) if rollout_passes else None

        task_results.append(
            {
                "task_name": task_name,
                "selected_rollout": selected,
                "selected_pass": selected_pass,
                "task_any_pass": any_pass,
                "rollout_passes": rollout_passes,
                "selected_result_path": str(Path(raw_path).parent / "result.json") if raw_path else None,
                "audit_status": "ok" if selected_pass is not None else "missing_pass_label",
            }
        )

    audited = [item for item in task_results if item["selected_pass"] is not None]
    selected_pass_count = sum(1 for item in audited if item["selected_pass"] is True)
    passable = [item for item in task_results if item["task_any_pass"] is not None]
    oracle_pass_count = sum(1 for item in passable if item["task_any_pass"] is True)

    summary = {
        "mode": "blind_rollout_selection_audit",
        "selection_summary": str(selection_summary),
        "baseline_summary": str(baseline_summary),
        "n_tasks": len(task_results),
        "n_audited": len(audited),
        "selected_pass_count": selected_pass_count,
        "selected_pass_rate": selected_pass_count / len(audited) if audited else None,
        "oracle_any_pass_count": oracle_pass_count,
        "oracle_any_pass_rate": oracle_pass_count / len(passable) if passable else None,
        "capture_rate_among_passable": selected_pass_count / oracle_pass_count if oracle_pass_count else None,
        "tasks": task_results,
    }

    if output_path is None:
        output_path = selection_summary.parent / "selected_rollout_audit_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["output_path"] = str(output_path)
    return summary


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "selection_summary",
        help="Path to blind_rollout_selection_summary.json or its parent selector directory",
    )
    parser.add_argument("--baseline-summary", default=None, help="Optional baseline_summary.json or experiment dir")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    selection_summary = _resolve_selection_summary(args.selection_summary)
    selection = _read_json(selection_summary)
    baseline_summary = _infer_baseline_summary(selection, args.baseline_summary)
    output_path = Path(args.output).resolve() if args.output else None

    summary = audit(selection_summary, baseline_summary, output_path)
    print(f"[audit] selected pass: {summary['selected_pass_count']}/{summary['n_audited']} ({_format_rate(summary['selected_pass_rate'])})")
    print(f"[audit] oracle any-pass: {summary['oracle_any_pass_count']}/{summary['n_tasks']} ({_format_rate(summary['oracle_any_pass_rate'])})")
    print(f"[audit] capture among passable: {_format_rate(summary['capture_rate_among_passable'])}")
    print(f"[audit] wrote: {summary['output_path']}")


if __name__ == "__main__":
    main()
