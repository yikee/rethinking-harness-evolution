#!/usr/bin/env python3
"""Evaluate train-run iteration harness snapshots on held-out test/val splits.

For each iteration (or the final workspace), runs harbor with the task lists from
exp-simple-code-cc-w-test.yaml and exp-simple-code-cc-w-val.yaml against the
harness snapshot that was evaluated on the train split in that iteration.

Usage:
    uv run python scripts/eval_held_out_iters.py --experiment 2026-06-26__10-37-00__claude-opus-4.6

    # Test split only, skip runs that already finished
    uv run python scripts/eval_held_out_iters.py \\
        --experiment 2026-06-26__10-37-00__claude-opus-4.6 \\
        --splits test --skip-existing

    # Also eval the final evolved workspace after the last iteration
    uv run python scripts/eval_held_out_iters.py \\
        --experiment 2026-06-26__10-37-00__claude-opus-4.6 \\
        --splits test val final
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from evolve_ahe import (  # noqa: E402
    EXPERIMENTS_DIR,
    compute_stats,
    find_latest_job_dir,
    load_config,
    run_harbor,
)

DEFAULT_TEST_CONFIG = PROJECT_DIR / "configs/experiments/exp-simple-code-cc-w-test.yaml"
DEFAULT_VAL_CONFIG = PROJECT_DIR / "configs/experiments/exp-simple-code-cc-w-val.yaml"
SCORES_YAML = "held_out_scores.yaml"
SCORES_MD = "held_out_scores.md"


def resolve_experiment_dir(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path.resolve()
    candidate = EXPERIMENTS_DIR / name
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Experiment directory not found: {name}")


def list_iteration_dirs(exp_dir: Path) -> list[tuple[int, Path]]:
    runs_dir = exp_dir / "runs"
    if not runs_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            num = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((num, child))
    return out


def iteration_workspace(iter_dir: Path) -> Path | None:
    ws = iter_dir / "input" / "workspace"
    return ws if ws.is_dir() else None


def split_output_dir(exp_dir: Path, split: str, iteration: int | None) -> Path:
    if split == "final":
        return exp_dir / "held_out" / "final"
    assert iteration is not None
    return exp_dir / "runs" / f"iteration_{iteration:03d}" / "input" / "held_out" / split


def eval_already_done(out_dir: Path) -> bool:
    marker = out_dir / "eval_done.json"
    if marker.is_file():
        return True
    job_dir = find_latest_job_dir(out_dir)
    if job_dir is None:
        return False
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return result.get("finished_at") is not None


def write_eval_marker(out_dir: Path, entry: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "eval_done.json"
    marker.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


def run_split_eval(
    config: dict,
    workspace: Path,
    out_dir: Path,
    *,
    skip_existing: bool,
) -> dict:
    agent_cfg = config["agent_config_filename"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and eval_already_done(out_dir):
        job_dir = find_latest_job_dir(out_dir)
        if job_dir is None:
            raise RuntimeError(f"--skip-existing set but no job dir in {out_dir}")
        print(f"[held-out] Reusing existing results: {job_dir.name}")
    else:
        job_dir = run_harbor(config, workspace, agent_cfg, out_dir)

    k = int(config.get("harbor", {}).get("k", 1))
    stats = compute_stats(job_dir, k=k)
    entry = {
        "pass_rate": round(stats["pass_rate"], 4),
        "n_pass": stats["n_pass"],
        "n_fail": stats["n_fail"],
        "n_exception": stats.get("n_exception", 0),
        "n_total": stats["n_total"],
        "k": k,
        "job_dir": str(job_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_eval_marker(out_dir, entry)
    return entry


def load_scores(exp_dir: Path) -> dict:
    path = exp_dir / SCORES_YAML
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {"experiment": exp_dir.name, "scores": []}
    data.setdefault("scores", [])
    return data


def upsert_score(data: dict, record: dict) -> None:
    scores = data["scores"]
    key = (record.get("split"), record.get("iteration"))
    for i, existing in enumerate(scores):
        if (existing.get("split"), existing.get("iteration")) == key:
            scores[i] = record
            return
    scores.append(record)
    scores.sort(key=lambda s: (s.get("split", ""), s.get("iteration") or 9999))


def regenerate_scores_md(exp_dir: Path, data: dict) -> None:
    lines = [
        f"# {data.get('experiment', exp_dir.name)} Held-Out Scores\n",
        "| Split | Iter | Pass Rate | Pass | Fail | Exc | Total | Job | Time |",
        "|-------|------|-----------|------|------|-----|-------|-----|------|",
    ]
    for s in data.get("scores", []):
        split = s.get("split", "")
        iteration = s.get("iteration")
        iter_label = str(iteration) if iteration is not None else "final"
        ts = (s.get("timestamp") or "")[:16].replace("T", " ")
        job_name = Path(s.get("job_dir", "")).name
        lines.append(
            f"| {split} | {iter_label} | {s.get('pass_rate', 0):.1%} | "
            f"{s.get('n_pass', 0)} | {s.get('n_fail', 0)} | {s.get('n_exception', 0)} | "
            f"{s.get('n_total', 0)} | {job_name} | {ts} |"
        )
    (exp_dir / SCORES_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_scores(exp_dir: Path, data: dict) -> None:
    with open(exp_dir / SCORES_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    regenerate_scores_md(exp_dir, data)


def parse_iterations(raw: str | None, available: list[int]) -> list[int]:
    if raw is None:
        return available
    selected: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            selected.update(range(int(start_s), int(end_s) + 1))
        else:
            selected.add(int(part))
    return sorted(i for i in available if i in selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory name under experiments/ or absolute path",
    )
    parser.add_argument(
        "--test-config",
        default=str(DEFAULT_TEST_CONFIG),
        help="Config overlay for the 34-task test split",
    )
    parser.add_argument(
        "--val-config",
        default=str(DEFAULT_VAL_CONFIG),
        help="Config overlay for the 10-task val split",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "val"],
        choices=["test", "val", "final"],
        help="Which held-out splits to evaluate (default: test val)",
    )
    parser.add_argument(
        "--iterations",
        default=None,
        help="Comma-separated iteration numbers and/or ranges, e.g. 1,3-5 (default: all found)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip harbor runs when eval_done.json or a finished job already exists",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env", override=True)

    exp_dir = resolve_experiment_dir(args.experiment)
    test_config = load_config(args.test_config)
    val_config = load_config(args.val_config)
    split_configs = {"test": test_config, "val": val_config}

    iter_dirs = list_iteration_dirs(exp_dir)
    available_iters = [n for n, _ in iter_dirs]
    selected_iters = parse_iterations(args.iterations, available_iters)

    if not selected_iters and "final" not in args.splits:
        print(f"[held-out] No iteration directories found under {exp_dir / 'runs'}")
        sys.exit(1)

    scores_data = load_scores(exp_dir)
    scores_data["experiment"] = exp_dir.name

    for iteration, iter_dir in iter_dirs:
        if iteration not in selected_iters:
            continue
        workspace = iteration_workspace(iter_dir)
        if workspace is None:
            print(f"[held-out] Skipping iteration {iteration}: no input/workspace snapshot")
            continue

        for split in args.splits:
            if split == "final":
                continue
            config = copy.deepcopy(split_configs[split])
            out_dir = split_output_dir(exp_dir, split, iteration)
            print(f"\n[held-out] iteration {iteration} / {split} ({len(config.get('task_names', []))} tasks)")
            result = run_split_eval(config, workspace, out_dir, skip_existing=args.skip_existing)
            record = {"split": split, "iteration": iteration, **result}
            record["job_dir"] = str(Path(result["job_dir"]).relative_to(exp_dir))
            upsert_score(scores_data, record)
            save_scores(exp_dir, scores_data)
            print(
                f"[held-out] iteration {iteration} / {split}: "
                f"{result['n_pass']}/{result['n_total']} = {result['pass_rate']:.1%}"
            )

    if "final" in args.splits:
        final_ws = exp_dir / "workspace"
        if not final_ws.is_dir():
            print("[held-out] Skipping final: workspace/ not found")
        else:
            for split in ("test", "val"):
                config = copy.deepcopy(split_configs[split])
                out_dir = split_output_dir(exp_dir, "final", None) / split
                print(f"\n[held-out] final / {split} ({len(config.get('task_names', []))} tasks)")
                result = run_split_eval(config, final_ws, out_dir, skip_existing=args.skip_existing)
                record = {"split": split, "iteration": None, **result}
                record["job_dir"] = str(Path(result["job_dir"]).relative_to(exp_dir))
                upsert_score(scores_data, record)
                save_scores(exp_dir, scores_data)
                print(
                    f"[held-out] final / {split}: "
                    f"{result['n_pass']}/{result['n_total']} = {result['pass_rate']:.1%}"
                )

    print(f"\n[held-out] Wrote {exp_dir / SCORES_YAML} and {exp_dir / SCORES_MD}")


if __name__ == "__main__":
    main()
