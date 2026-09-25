"""--evolve: agent-maintained skills, sandboxed analysis tools and declarative hooks.

Everything lives on the host inside the run's visible directory; the player only
reaches it through the public game tools (`read_skill`, `write_skill`,
`write_tool`, `run_tool`, `write_hooks`), at any point of the run, exactly like
GUIDE.md / WORKING.md. Both runtimes (Claude Code and Codex) share this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
MAX_SKILLS = 32
MAX_SKILL_CHARS = 16 * 1024
MAX_TOOLS = 16
MAX_TOOL_CODE_CHARS = 32 * 1024
MAX_TOOL_DESCRIPTION_CHARS = 400
MAX_TOOL_ARGS_CHARS = 8 * 1024
MAX_TOOL_OUTPUT_CHARS = 8 * 1024
TOOL_TIMEOUT_SECONDS = 10
MAX_HOOKS = 20
MAX_HOOK_NOTE_CHARS = 500
GRID_SIZE = 64
HOOK_EVENTS = ("before_play", "after_play")
HOOK_CONDITIONS = frozenset(
    {
        "action_in",
        "click_in_region",
        "frame_unchanged_steps_at_least",
        "same_action_repeated_at_least",
        "level_steps_at_least",
    }
)
ACTION_NAMES = (
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
)
EVOLVE_TOOL_NAMES = frozenset(
    {
        "read_skill",
        "write_skill",
        "write_tool",
        "run_tool",
        "write_hooks",
    }
)
SANDBOX_IMAGES = ("arc3-codex-player:0.1", "arc3-claude-player:0.1")

EVOLVE_INSTRUCTIONS = """## Self-improvement: skills, analysis tools and hooks

You maintain three kinds of reusable assets that persist across levels:

- Skills: short Markdown procedures stored as `skills/<name>.md`; the first line is a one-line description. The skill index is listed below; read one with `read_skill`, create, replace or delete one with `write_skill`.
- Analysis tools: Python functions you write with `write_tool`. The code must define `analyze(frames, args)`, where `frames` is the list of 64x64 integer color grids (values 0-15) archived for one turn (the last grid is that turn's final visual) and `args` is the JSON object you pass in; return any JSON-serializable value. Run one with `run_tool`; it executes in an isolated sandbox (pure Python standard library, no network, 10 s limit) and does not count as a game action.
- Hooks: declarative rules stored in `hooks.json`, replaced as a whole with `write_hooks`. A rule is {"name", "event": "before_play" | "after_play", "when": {...}, "note": "text", "block": false}. Conditions in `when` must all hold: `action_in` (list of action names), `click_in_region` {"x0","y0","x1","y1"} in 64x64 grid coordinates, `frame_unchanged_steps_at_least`, `same_action_repeated_at_least`, `level_steps_at_least`. A matching rule appends its note to the play result; a `before_play` rule with "block": true stops the action before it reaches the game (repeat the same action immediately to override the block).

Update these assets whenever you learn something reusable, the same way you maintain GUIDE.md and WORKING.md: after a level is completed (`level_boundary: true` in the play result), review what worked in that level and capture it as a skill, a tool or a hook so the next levels need fewer actions. Changes take effect immediately; hooks are evaluated on the very next `play`, and the skill/tool/hook index below is refreshed whenever a new thread starts."""


@dataclass(frozen=True)
class HookContext:
    action: str
    environment_x: int | None
    environment_y: int | None
    frame_unchanged_steps: int
    same_action_repeated: int
    level_steps: int


@dataclass(frozen=True)
class HookOutcome:
    notes: tuple[str, ...] = ()
    blocked: bool = False
    blocking_rule: str | None = None
    matched: tuple[str, ...] = ()


def validate_name(value: object, kind: str) -> str:
    if not isinstance(value, str) or not NAME_PATTERN.match(value):
        raise ValueError(
            f"Invalid {kind} name; use 1-40 lowercase letters, digits, '-' or '_'."
        )
    return value


def validate_hooks(rules: object) -> list[dict[str, Any]]:
    """Return a normalized copy of `rules` or raise ValueError."""
    if not isinstance(rules, list):
        raise ValueError("hooks must be a list of rule objects.")
    if len(rules) > MAX_HOOKS:
        raise ValueError(f"At most {MAX_HOOKS} hook rules are allowed.")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Rule {index} must be an object.")
        unexpected = set(rule) - {"name", "event", "when", "note", "block"}
        if unexpected:
            raise ValueError(
                f"Rule {index} has unexpected keys: {', '.join(sorted(unexpected))}."
            )
        name = validate_name(rule.get("name"), "hook")
        if name in names:
            raise ValueError(f"Rule {index}: duplicate hook name {name!r}.")
        names.add(name)
        event = rule.get("event")
        if event not in HOOK_EVENTS:
            raise ValueError(
                f"Rule {index}: event must be one of {', '.join(HOOK_EVENTS)}."
            )
        when = rule.get("when")
        if not isinstance(when, dict) or not when:
            raise ValueError(f"Rule {index}: when must be a non-empty object.")
        unknown = set(when) - HOOK_CONDITIONS
        if unknown:
            raise ValueError(
                f"Rule {index}: unknown conditions: {', '.join(sorted(unknown))}."
            )
        conditions: dict[str, Any] = {}
        if "action_in" in when:
            actions = when["action_in"]
            if (
                not isinstance(actions, list)
                or not actions
                or any(action not in ACTION_NAMES for action in actions)
            ):
                raise ValueError(
                    f"Rule {index}: action_in must list valid action names."
                )
            conditions["action_in"] = list(dict.fromkeys(actions))
        if "click_in_region" in when:
            region = when["click_in_region"]
            if not isinstance(region, dict) or set(region) != {"x0", "y0", "x1", "y1"}:
                raise ValueError(
                    f"Rule {index}: click_in_region needs x0, y0, x1, y1."
                )
            values = {key: region[key] for key in ("x0", "y0", "x1", "y1")}
            if any(
                type(value) is not int or not 0 <= value < GRID_SIZE
                for value in values.values()
            ):
                raise ValueError(
                    f"Rule {index}: click_in_region coordinates must be integers "
                    f"in 0..{GRID_SIZE - 1}."
                )
            if values["x0"] > values["x1"] or values["y0"] > values["y1"]:
                raise ValueError(
                    f"Rule {index}: click_in_region needs x0 <= x1 and y0 <= y1."
                )
            conditions["click_in_region"] = values
        for key in (
            "frame_unchanged_steps_at_least",
            "same_action_repeated_at_least",
            "level_steps_at_least",
        ):
            if key in when:
                value = when[key]
                if type(value) is not int or value < 1 or value > 100000:
                    raise ValueError(
                        f"Rule {index}: {key} must be a positive integer."
                    )
                conditions[key] = value
        note = rule.get("note")
        if (
            not isinstance(note, str)
            or not note.strip()
            or len(note) > MAX_HOOK_NOTE_CHARS
        ):
            raise ValueError(
                f"Rule {index}: note must be 1..{MAX_HOOK_NOTE_CHARS} characters."
            )
        block = rule.get("block", False)
        if type(block) is not bool:
            raise ValueError(f"Rule {index}: block must be true or false.")
        if block and event != "before_play":
            raise ValueError(f"Rule {index}: only before_play rules can block.")
        normalized.append(
            {
                "name": name,
                "event": event,
                "when": conditions,
                "note": note.strip(),
                "block": block,
            }
        )
    return normalized


def rule_matches(rule: dict[str, Any], context: HookContext) -> bool:
    when = rule["when"]
    actions = when.get("action_in")
    if actions is not None and context.action not in actions:
        return False
    region = when.get("click_in_region")
    if region is not None:
        if context.environment_x is None or context.environment_y is None:
            return False
        if not (
            region["x0"] <= context.environment_x <= region["x1"]
            and region["y0"] <= context.environment_y <= region["y1"]
        ):
            return False
    minimum = when.get("frame_unchanged_steps_at_least")
    if minimum is not None and context.frame_unchanged_steps < minimum:
        return False
    minimum = when.get("same_action_repeated_at_least")
    if minimum is not None and context.same_action_repeated < minimum:
        return False
    minimum = when.get("level_steps_at_least")
    if minimum is not None and context.level_steps < minimum:
        return False
    return True


def evaluate_hooks(
    rules: list[dict[str, Any]],
    event: str,
    context: HookContext,
) -> HookOutcome:
    notes: list[str] = []
    matched: list[str] = []
    blocked = False
    blocking_rule = None
    for rule in rules:
        if rule["event"] != event or not rule_matches(rule, context):
            continue
        matched.append(rule["name"])
        notes.append(f"[{rule['name']}] {rule['note']}")
        if rule.get("block") and event == "before_play" and not blocked:
            blocked = True
            blocking_rule = rule["name"]
    return HookOutcome(
        notes=tuple(notes),
        blocked=blocked,
        blocking_rule=blocking_rule,
        matched=tuple(matched),
    )


def action_key(action: str, environment_x: int | None, environment_y: int | None) -> str:
    if action == "ACTION6" and environment_x is not None and environment_y is not None:
        return f"ACTION6@{environment_x},{environment_y}"
    return action


def frame_digest(frame: Any) -> str:
    """Stable digest of one 64x64 grid (numpy array or nested lists)."""
    if hasattr(frame, "tolist"):
        frame = frame.tolist()
    return hashlib.sha1(
        json.dumps(frame, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class HookState:
    """Per-run counters that the hook conditions are evaluated against."""

    def __init__(self, initial_frame_digest: str | None = None) -> None:
        self.last_action_key: str | None = None
        self.same_action_repeated = 0
        self.frame_unchanged_steps = 0
        self.last_frame_digest = initial_frame_digest
        self.last_blocked_key: str | None = None

    def proposed_repeats(self, key: str) -> int:
        """How many times in a row `key` would have been played, including now."""
        if key == self.last_action_key:
            return self.same_action_repeated + 1
        return 1

    def override_allowed(self, key: str) -> bool:
        return self.last_blocked_key == key

    def record_block(self, key: str) -> None:
        self.last_blocked_key = key

    def record_step(self, key: str, new_frame_digest: str | None) -> None:
        self.last_blocked_key = None
        if key == self.last_action_key:
            self.same_action_repeated += 1
        else:
            self.last_action_key = key
            self.same_action_repeated = 1
        if new_frame_digest is not None and new_frame_digest == self.last_frame_digest:
            self.frame_unchanged_steps += 1
        else:
            self.frame_unchanged_steps = 0
        self.last_frame_digest = new_frame_digest

    def reset_level(self) -> None:
        self.frame_unchanged_steps = 0
        self.same_action_repeated = 0
        self.last_action_key = None
        self.last_blocked_key = None


SANDBOX_RUNNER = r'''
import importlib.util
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    mode = payload.get("mode", "run")
    try:
        tool_path = Path(__file__).with_name("tool.py")
        spec = importlib.util.spec_from_file_location("evolve_tool", str(tool_path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        analyze = getattr(module, "analyze", None)
        if not callable(analyze):
            raise TypeError("the tool code must define analyze(frames, args)")
        if mode == "check":
            result = None
        else:
            result = analyze(payload.get("frames", []), payload.get("args", {}))
        text = json.dumps({"ok": True, "result": result}, default=str)
    except BaseException as exc:  # noqa: BLE001 - report everything to the model
        text = json.dumps(
            {
                "ok": False,
                "error": "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip(),
                "traceback": traceback.format_exc()[-2000:],
            }
        )
    sys.stdout.write(text)
    sys.stdout.flush()


main()
'''


def default_sandbox_image() -> str:
    for image in SANDBOX_IMAGES:
        probe = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode == 0:
            return image
    raise RuntimeError(
        "No sandbox image is available; build arc3-codex-player:0.1 or "
        "arc3-claude-player:0.1."
    )


class ToolSandbox:
    """Run agent-authored analysis functions in a throwaway, network-less container.

    `local=True` runs the same runner script with the host interpreter instead
    of docker; it exists for tests and must not be used for real runs.
    """

    def __init__(
        self,
        work_dir: Path,
        *,
        image: str | None = None,
        timeout_seconds: int = TOOL_TIMEOUT_SECONDS,
        local: bool = False,
    ) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.local = local
        self._image = image

    @property
    def image(self) -> str:
        if self._image is None:
            self._image = default_sandbox_image()
        return self._image

    def check(self, code: str) -> str | None:
        """Import the code once; return an error message or None."""
        outcome = self._execute(code, {"mode": "check"})
        if outcome.get("ok") is True:
            return None
        return str(outcome.get("error") or "the tool code could not be loaded")

    def run(self, code: str, frames: list[Any], args: dict[str, Any]) -> dict[str, Any]:
        return self._execute(code, {"mode": "run", "frames": frames, "args": args})

    def _execute(self, code: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        stage = Path(tempfile.mkdtemp(prefix="tool_", dir=self.work_dir))
        try:
            os.chmod(stage, 0o755)
            (stage / "tool.py").write_text(code, encoding="utf-8")
            (stage / "_runner.py").write_text(SANDBOX_RUNNER, encoding="utf-8")
            for name in ("tool.py", "_runner.py"):
                os.chmod(stage / name, 0o644)
            stdin = json.dumps(payload, separators=(",", ":"))
            if self.local:
                command = [sys.executable, "-I", "-B", str(stage / "_runner.py")]
                name = None
            else:
                name = f"arc3-evolve-{uuid.uuid4().hex[:12]}"
                command = self._docker_command(stage, name)
            try:
                completed = subprocess.run(
                    command,
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds + (5 if self.local else 30),
                )
            except subprocess.TimeoutExpired:
                if name is not None:
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        capture_output=True,
                        timeout=30,
                    )
                return {
                    "ok": False,
                    "error": f"the tool exceeded the {self.timeout_seconds} s limit",
                    "seconds": round(time.monotonic() - started, 2),
                }
            seconds = round(time.monotonic() - started, 2)
            if completed.returncode == 124:
                return {
                    "ok": False,
                    "error": f"the tool exceeded the {self.timeout_seconds} s limit",
                    "seconds": seconds,
                }
            stdout = completed.stdout.strip()
            try:
                outcome = json.loads(stdout) if stdout else None
            except json.JSONDecodeError:
                outcome = None
            if not isinstance(outcome, dict):
                detail = (completed.stderr or completed.stdout or "").strip()[-1000:]
                return {
                    "ok": False,
                    "error": (
                        f"the sandbox exited with code {completed.returncode}"
                        + (f": {detail}" if detail else "")
                    ),
                    "seconds": seconds,
                }
            outcome["seconds"] = seconds
            return outcome
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _docker_command(self, stage: Path, name: str) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--pull=never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-v",
            f"{stage}:/sandbox:ro",
            "-w",
            "/sandbox",
            self.image,
            "timeout",
            str(self.timeout_seconds),
            "python3",
            "-I",
            "-B",
            "/sandbox/_runner.py",
        ]


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def skill_description(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:160]
    return ""


class EvolveStore:
    """Host-side storage for skills, tools and hooks plus their tool handlers."""

    def __init__(
        self,
        root: Path,
        *,
        log_path: Path,
        sandbox: ToolSandbox,
    ) -> None:
        self.root = root
        self.skills_dir = root / "skills"
        self.tools_dir = root / "tools"
        self.tools_index_path = self.tools_dir / "index.json"
        self.hooks_path = root / "hooks.json"
        self.log_path = log_path
        self.sandbox = sandbox
        self._hooks_cache: list[dict[str, Any]] | None = None

    def initialize(self) -> None:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        if not self.tools_index_path.exists():
            _atomic_write(self.tools_index_path, "{}\n")
        if not self.hooks_path.exists():
            _atomic_write(self.hooks_path, "[]\n")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # -- logging -----------------------------------------------------------
    def log(self, event: str, **fields: Any) -> None:
        record = {"time": time.time(), "event": event, **fields}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")

    # -- skills ------------------------------------------------------------
    def list_skills(self) -> list[dict[str, str]]:
        skills = []
        for path in sorted(self.skills_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8", errors="replace")
            skills.append({"name": path.stem, "description": skill_description(content)})
        return skills

    def read_skill(self, name: str) -> str | None:
        path = self.skills_dir / f"{validate_name(name, 'skill')}.md"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    def write_skill(self, name: str, content: str | None) -> str:
        name = validate_name(name, "skill")
        path = self.skills_dir / f"{name}.md"
        if content is None:
            if not path.exists():
                raise ValueError(f"Skill {name!r} does not exist.")
            path.unlink()
            return f"Skill {name!r} deleted."
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Skill content must be non-empty text.")
        if len(content) > MAX_SKILL_CHARS:
            raise ValueError(f"Skill content must be at most {MAX_SKILL_CHARS} characters.")
        if not path.exists() and len(list(self.skills_dir.glob("*.md"))) >= MAX_SKILLS:
            raise ValueError(f"At most {MAX_SKILLS} skills are allowed; delete one first.")
        created = not path.exists()
        _atomic_write(path, content.strip() + "\n")
        return f"Skill {name!r} {'created' if created else 'updated'}."

    # -- tools -------------------------------------------------------------
    def _read_tool_index(self) -> dict[str, dict[str, Any]]:
        try:
            value = json.loads(self.tools_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def list_tools(self) -> list[dict[str, Any]]:
        index = self._read_tool_index()
        return [
            {"name": name, "description": str(entry.get("description", ""))}
            for name, entry in sorted(index.items())
        ]

    def tool_code(self, name: str) -> str | None:
        path = self.tools_dir / f"{validate_name(name, 'tool')}.py"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    def write_tool(self, name: str, description: str | None, code: str | None) -> str:
        name = validate_name(name, "tool")
        path = self.tools_dir / f"{name}.py"
        index = self._read_tool_index()
        if code is None:
            if name not in index and not path.exists():
                raise ValueError(f"Tool {name!r} does not exist.")
            path.unlink(missing_ok=True)
            index.pop(name, None)
            _atomic_write(
                self.tools_index_path,
                json.dumps(index, indent=2, sort_keys=True) + "\n",
            )
            return f"Tool {name!r} deleted."
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Tool code must be non-empty Python source.")
        if len(code) > MAX_TOOL_CODE_CHARS:
            raise ValueError(f"Tool code must be at most {MAX_TOOL_CODE_CHARS} characters.")
        if (
            not isinstance(description, str)
            or not description.strip()
            or len(description) > MAX_TOOL_DESCRIPTION_CHARS
        ):
            raise ValueError(
                f"Tool description must be 1..{MAX_TOOL_DESCRIPTION_CHARS} characters."
            )
        if name not in index and len(index) >= MAX_TOOLS:
            raise ValueError(f"At most {MAX_TOOLS} tools are allowed; delete one first.")
        try:
            compile(code, f"{name}.py", "exec")
        except SyntaxError as exc:
            raise ValueError(f"Tool code has a syntax error: {exc}") from None
        if "def analyze" not in code:
            raise ValueError("Tool code must define analyze(frames, args).")
        error = self.sandbox.check(code)
        if error is not None:
            raise ValueError(f"Tool code failed to load in the sandbox: {error}")
        created = name not in index
        _atomic_write(path, code.rstrip() + "\n")
        index[name] = {
            "description": description.strip(),
            "updated_at": time.time(),
            "chars": len(code),
        }
        _atomic_write(
            self.tools_index_path,
            json.dumps(index, indent=2, sort_keys=True) + "\n",
        )
        return f"Tool {name!r} {'created' if created else 'updated'}."

    def run_tool(
        self,
        name: str,
        frames: list[Any],
        args: dict[str, Any],
    ) -> dict[str, Any]:
        code = self.tool_code(name)
        if code is None:
            raise ValueError(f"Tool {name!r} does not exist.")
        return self.sandbox.run(code, frames, args)

    # -- hooks -------------------------------------------------------------
    def read_hooks(self) -> list[dict[str, Any]]:
        if self._hooks_cache is not None:
            return self._hooks_cache
        try:
            value = json.loads(self.hooks_path.read_text(encoding="utf-8"))
            rules = validate_hooks(value)
        except (OSError, ValueError):
            rules = []
        self._hooks_cache = rules
        return rules

    def write_hooks(self, rules: object) -> str:
        normalized = validate_hooks(rules)
        _atomic_write(self.hooks_path, json.dumps(normalized, indent=2) + "\n")
        self._hooks_cache = normalized
        return f"hooks.json replaced with {len(normalized)} rule(s)."

    # -- prompt index ------------------------------------------------------
    def render_index(self) -> str:
        lines = ["## Current skills, analysis tools and hooks", ""]
        skills = self.list_skills()
        lines.append("Skills (read_skill):")
        if skills:
            lines.extend(f"- {s['name']}: {s['description']}" for s in skills)
        else:
            lines.append("- (none yet)")
        lines.append("")
        tools = self.list_tools()
        lines.append("Analysis tools (run_tool):")
        if tools:
            lines.extend(f"- {t['name']}: {t['description']}" for t in tools)
        else:
            lines.append("- (none yet)")
        lines.append("")
        hooks = self.read_hooks()
        lines.append("Hooks (active rules):")
        if hooks:
            for rule in hooks:
                effect = "block" if rule.get("block") else "note"
                lines.append(
                    f"- {rule['name']} [{rule['event']}] when "
                    f"{json.dumps(rule['when'], separators=(',', ':'))} -> "
                    f"{effect}: {rule['note']}"
                )
        else:
            lines.append("- (none yet)")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "skills": self.list_skills(),
            "tools": self.list_tools(),
            "hooks": self.read_hooks(),
        }

    # -- tool handlers shared by both dispatchers -------------------------
    def execute(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        frames_for_turn: Callable[[int | None], tuple[int, list[Any]]],
    ) -> tuple[str, bool]:
        """Run one evolve tool; returns (text, success)."""
        try:
            if tool == "read_skill":
                if set(arguments) != {"name"}:
                    raise ValueError("read_skill takes exactly one argument: name.")
                content = self.read_skill(arguments["name"])
                if content is None:
                    return (
                        f"Skill {arguments['name']!r} does not exist. "
                        f"Available: {', '.join(s['name'] for s in self.list_skills()) or 'none'}.",
                        False,
                    )
                return content, True
            if tool == "write_skill":
                if set(arguments) != {"name", "content"}:
                    raise ValueError("write_skill takes name and content (null deletes).")
                message = self.write_skill(arguments["name"], arguments["content"])
                self.log(
                    "write_skill",
                    name=arguments["name"],
                    deleted=arguments["content"] is None,
                    chars=len(arguments["content"] or ""),
                )
                return message, True
            if tool == "write_tool":
                if set(arguments) - {"name", "description", "code"} or "name" not in arguments or "code" not in arguments:
                    raise ValueError(
                        "write_tool takes name, description and code (null code deletes)."
                    )
                message = self.write_tool(
                    arguments["name"],
                    arguments.get("description"),
                    arguments["code"],
                )
                self.log(
                    "write_tool",
                    name=arguments["name"],
                    deleted=arguments["code"] is None,
                    chars=len(arguments["code"] or ""),
                )
                return message, True
            if tool == "run_tool":
                if set(arguments) - {"name", "turn", "args"} or "name" not in arguments:
                    raise ValueError("run_tool takes name, optional turn and optional args.")
                name = validate_name(arguments["name"], "tool")
                turn = arguments.get("turn")
                if turn is not None and (type(turn) is not int or turn < 0):
                    raise ValueError("turn must be a non-negative integer.")
                args = arguments.get("args", {})
                if args is None:
                    args = {}
                if not isinstance(args, dict):
                    raise ValueError("args must be a JSON object.")
                if len(json.dumps(args)) > MAX_TOOL_ARGS_CHARS:
                    raise ValueError(f"args must serialize to at most {MAX_TOOL_ARGS_CHARS} characters.")
                if self.tool_code(name) is None:
                    return (
                        f"Tool {name!r} does not exist. Available: "
                        f"{', '.join(t['name'] for t in self.list_tools()) or 'none'}.",
                        False,
                    )
                resolved_turn, frames = frames_for_turn(turn)
                outcome = self.run_tool(name, frames, args)
                self.log(
                    "run_tool",
                    name=name,
                    turn=resolved_turn,
                    ok=outcome.get("ok"),
                    seconds=outcome.get("seconds"),
                )
                if outcome.get("ok") is True:
                    result = outcome.get("result")
                    text = json.dumps(
                        {
                            "tool": name,
                            "turn": resolved_turn,
                            "frames": len(frames),
                            "seconds": outcome.get("seconds"),
                            "result": result,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                    return truncate(text), True
                error = str(outcome.get("error") or "unknown error")
                trace = outcome.get("traceback")
                text = json.dumps(
                    {
                        "tool": name,
                        "turn": resolved_turn,
                        "error": error,
                        **({"traceback": trace} if isinstance(trace, str) else {}),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                return truncate(text), False
            if tool == "write_hooks":
                if set(arguments) != {"rules"}:
                    raise ValueError("write_hooks takes exactly one argument: rules.")
                message = self.write_hooks(arguments["rules"])
                self.log("write_hooks", count=len(self.read_hooks()))
                return message, True
        except ValueError as exc:
            return str(exc), False
        return "Unknown tool.", False


def _read_only(idempotent: bool = True) -> dict[str, bool]:
    return {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": idempotent,
        "openWorldHint": False,
    }


def _writes(destructive: bool = False) -> dict[str, bool]:
    return {
        "readOnlyHint": False,
        "destructiveHint": destructive,
        "idempotentHint": True,
        "openWorldHint": False,
    }


_NAME_SCHEMA = {
    "type": "string",
    "pattern": NAME_PATTERN.pattern,
    "minLength": 1,
    "maxLength": 40,
}

_HOOK_RULE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "event", "when", "note"],
    "properties": {
        "name": _NAME_SCHEMA,
        "event": {"type": "string", "enum": list(HOOK_EVENTS)},
        "when": {
            "type": "object",
            "additionalProperties": False,
            "minProperties": 1,
            "properties": {
                "action_in": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": list(ACTION_NAMES)},
                },
                "click_in_region": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x0", "y0", "x1", "y1"],
                    "properties": {
                        key: {"type": "integer", "minimum": 0, "maximum": GRID_SIZE - 1}
                        for key in ("x0", "y0", "x1", "y1")
                    },
                },
                "frame_unchanged_steps_at_least": {"type": "integer", "minimum": 1},
                "same_action_repeated_at_least": {"type": "integer", "minimum": 1},
                "level_steps_at_least": {"type": "integer", "minimum": 1},
            },
        },
        "note": {"type": "string", "minLength": 1, "maxLength": MAX_HOOK_NOTE_CHARS},
        "block": {"type": "boolean", "default": False},
    },
}


def build_evolve_tools() -> tuple[dict[str, Any], ...]:
    """Public contracts of the --evolve tools (shared by both runtimes)."""
    return (
        {
            "name": "read_skill",
            "description": (
                "Read one of your skills (`skills/<name>.md`). The skill index with "
                "one-line descriptions is in the system prompt."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {"name": _NAME_SCHEMA},
            },
            "annotations": _read_only(),
        },
        {
            "name": "write_skill",
            "description": (
                "Create or replace a skill: a short, reusable Markdown procedure for "
                "this game whose first line is a one-line description. Set content to "
                f"null to delete the skill. At most {MAX_SKILLS} skills of "
                f"{MAX_SKILL_CHARS} characters each."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "content"],
                "properties": {
                    "name": _NAME_SCHEMA,
                    "content": {
                        "type": ["string", "null"],
                        "minLength": 1,
                        "maxLength": MAX_SKILL_CHARS,
                    },
                },
            },
            "annotations": _writes(destructive=True),
        },
        {
            "name": "write_tool",
            "description": (
                "Create or replace an analysis tool: Python source that defines "
                "`analyze(frames, args)`. `frames` is the list of 64x64 integer color "
                "grids (values 0-15) archived for one turn, last grid = final visual; "
                "`args` is the JSON object passed to run_tool; return any "
                "JSON-serializable value. Only the Python standard library is "
                "available (no numpy, no files, no network). The code is loaded once "
                "in the sandbox on save and any error is returned. Set code to null to "
                f"delete the tool. At most {MAX_TOOLS} tools."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "description", "code"],
                "properties": {
                    "name": _NAME_SCHEMA,
                    "description": {
                        "type": ["string", "null"],
                        "minLength": 1,
                        "maxLength": MAX_TOOL_DESCRIPTION_CHARS,
                    },
                    "code": {
                        "type": ["string", "null"],
                        "minLength": 1,
                        "maxLength": MAX_TOOL_CODE_CHARS,
                    },
                },
            },
            "annotations": _writes(destructive=True),
        },
        {
            "name": "run_tool",
            "description": (
                "Run one of your analysis tools on the archived grids of a turn "
                "(omit turn for the current visual). Executes in an isolated sandbox "
                f"with a {TOOL_TIMEOUT_SECONDS} s limit; the JSON result is returned as "
                f"text (at most {MAX_TOOL_OUTPUT_CHARS} characters). This does not "
                "change the game and does not count as a game action."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {
                    "name": _NAME_SCHEMA,
                    "turn": {"type": "integer", "minimum": 0, "maximum": 999999},
                    "args": {"type": "object"},
                },
            },
            "annotations": _read_only(idempotent=True),
        },
        {
            "name": "write_hooks",
            "description": (
                "Replace all hook rules (`hooks.json`). Each rule fires on "
                "before_play or after_play when every condition in `when` holds: "
                "action_in (action names), click_in_region {x0,y0,x1,y1} in 64x64 grid "
                "coordinates, frame_unchanged_steps_at_least (consecutive steps whose "
                "final visual did not change), same_action_repeated_at_least "
                "(consecutive identical actions, including the one being played), "
                "level_steps_at_least (actions taken in the current level). A matching "
                "rule appends its note to the play result; a before_play rule with "
                "block=true stops the action before it reaches the game (repeat the "
                f"same action immediately to override). At most {MAX_HOOKS} rules; "
                "pass an empty list to remove all hooks."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rules"],
                "properties": {
                    "rules": {
                        "type": "array",
                        "maxItems": MAX_HOOKS,
                        "items": _HOOK_RULE_SCHEMA,
                    }
                },
            },
            "annotations": _writes(destructive=True),
        },
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
