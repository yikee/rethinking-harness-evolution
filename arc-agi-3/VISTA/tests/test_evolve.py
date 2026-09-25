"""--evolve: skills / analysis tools / hooks store, hook engine and sandbox."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vista_arc3.shared.evolve import (
    EVOLVE_TOOL_NAMES,
    MAX_HOOKS,
    MAX_SKILLS,
    MAX_TOOL_OUTPUT_CHARS,
    SANDBOX_IMAGES,
    EvolveStore,
    HookContext,
    HookState,
    ToolSandbox,
    action_key,
    build_evolve_tools,
    evaluate_hooks,
    frame_digest,
    validate_hooks,
)


def make_store(tmp_path: Path, **sandbox_kwargs) -> EvolveStore:
    sandbox = ToolSandbox(tmp_path / "sandbox", local=True, **sandbox_kwargs)
    store = EvolveStore(
        tmp_path / "player",
        log_path=tmp_path / "private" / "evolve_log.jsonl",
        sandbox=sandbox,
    )
    store.initialize()
    return store


def frames_for_turn(turn):
    resolved = 3 if turn is None else turn
    return resolved, [[[resolved] * 64 for _ in range(64)]]


GOOD_TOOL = """
def analyze(frames, args):
    grid = frames[-1]
    return {"height": len(grid), "width": len(grid[0]), "corner": grid[0][0], "args": args}
"""


# -- tool contracts ----------------------------------------------------------


def test_evolve_tool_contracts_cover_exactly_the_evolve_tool_names() -> None:
    tools = build_evolve_tools()
    assert {tool["name"] for tool in tools} == EVOLVE_TOOL_NAMES
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["inputSchema"]["additionalProperties"] is False
        assert set(tool["annotations"]) == {
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        }
    # The player edits skills / tools / hooks through these tools at any time;
    # there is no separate reflection turn and no end_reflection tool.
    assert "end_reflection" not in EVOLVE_TOOL_NAMES


# -- hooks -------------------------------------------------------------------


def test_validate_hooks_normalizes_and_rejects_bad_rules() -> None:
    rules = validate_hooks(
        [
            {
                "name": "warn-repeat",
                "event": "after_play",
                "when": {"same_action_repeated_at_least": 3},
                "note": " Same action three times. ",
            },
            {
                "name": "no-border",
                "event": "before_play",
                "when": {
                    "action_in": ["ACTION6", "ACTION6"],
                    "click_in_region": {"x0": 0, "y0": 0, "x1": 63, "y1": 1},
                },
                "note": "Top rows are decoration.",
                "block": True,
            },
        ]
    )
    assert rules[0]["note"] == "Same action three times."
    assert rules[0]["block"] is False
    assert rules[1]["when"]["action_in"] == ["ACTION6"]

    bad = [
        ("not a list", "must be a list"),
        ([{"name": "Bad Name", "event": "after_play", "when": {"level_steps_at_least": 1}, "note": "x"}], "Invalid hook name"),
        ([{"name": "a", "event": "on_win", "when": {"level_steps_at_least": 1}, "note": "x"}], "event must be"),
        ([{"name": "a", "event": "after_play", "when": {}, "note": "x"}], "non-empty"),
        ([{"name": "a", "event": "after_play", "when": {"weather": "rain"}, "note": "x"}], "unknown conditions"),
        ([{"name": "a", "event": "after_play", "when": {"action_in": ["JUMP"]}, "note": "x"}], "valid action names"),
        ([{"name": "a", "event": "after_play", "when": {"click_in_region": {"x0": 5, "y0": 0, "x1": 1, "y1": 1}}, "note": "x"}], "x0 <= x1"),
        ([{"name": "a", "event": "after_play", "when": {"click_in_region": {"x0": 0, "y0": 0, "x1": 64, "y1": 1}}, "note": "x"}], "0..63"),
        ([{"name": "a", "event": "after_play", "when": {"level_steps_at_least": 0}, "note": "x"}], "positive integer"),
        ([{"name": "a", "event": "after_play", "when": {"level_steps_at_least": 1}, "note": ""}], "note must be"),
        ([{"name": "a", "event": "after_play", "when": {"level_steps_at_least": 1}, "note": "x", "block": True}], "only before_play"),
        ([{"name": "a", "event": "after_play", "when": {"level_steps_at_least": 1}, "note": "x", "extra": 1}], "unexpected keys"),
        (
            [
                {"name": "a", "event": "after_play", "when": {"level_steps_at_least": 1}, "note": "x"},
                {"name": "a", "event": "after_play", "when": {"level_steps_at_least": 1}, "note": "y"},
            ],
            "duplicate",
        ),
        (
            [
                {"name": f"r{i}", "event": "after_play", "when": {"level_steps_at_least": 1}, "note": "x"}
                for i in range(MAX_HOOKS + 1)
            ],
            f"At most {MAX_HOOKS}",
        ),
    ]
    for rules, message in bad:
        with pytest.raises(ValueError, match=message):
            validate_hooks(rules)


def test_evaluate_hooks_matches_all_conditions_and_blocks_only_before_play() -> None:
    rules = validate_hooks(
        [
            {
                "name": "no-border",
                "event": "before_play",
                "when": {
                    "action_in": ["ACTION6"],
                    "click_in_region": {"x0": 0, "y0": 0, "x1": 63, "y1": 1},
                },
                "note": "Top rows are decoration.",
                "block": True,
            },
            {
                "name": "stuck",
                "event": "after_play",
                "when": {"frame_unchanged_steps_at_least": 2},
                "note": "Nothing changed twice; try another action.",
            },
        ]
    )
    inside = HookContext("ACTION6", 10, 1, 0, 1, 4)
    outside = HookContext("ACTION6", 10, 30, 0, 1, 4)
    other_action = HookContext("ACTION1", None, None, 0, 1, 4)

    before = evaluate_hooks(rules, "before_play", inside)
    assert before.blocked is True and before.blocking_rule == "no-border"
    assert before.notes == ("[no-border] Top rows are decoration.",)
    assert evaluate_hooks(rules, "before_play", outside).blocked is False
    assert evaluate_hooks(rules, "before_play", other_action).matched == ()

    # The blocking rule is a before_play rule only; after_play sees the other one.
    after = evaluate_hooks(rules, "after_play", HookContext("ACTION1", None, None, 2, 1, 5))
    assert after.blocked is False
    assert after.matched == ("stuck",)
    assert evaluate_hooks(rules, "after_play", HookContext("ACTION1", None, None, 1, 1, 5)).matched == ()


def test_hook_state_counts_repeats_unchanged_frames_and_overrides() -> None:
    state = HookState(frame_digest([[0]]))
    key = action_key("ACTION6", 3, 4)
    assert key == "ACTION6@3,4"
    assert action_key("ACTION1", None, None) == "ACTION1"

    assert state.proposed_repeats(key) == 1
    state.record_step(key, frame_digest([[0]]))  # frame did not change
    assert state.same_action_repeated == 1
    assert state.frame_unchanged_steps == 1
    assert state.proposed_repeats(key) == 2
    state.record_step(key, frame_digest([[1]]))  # frame changed
    assert state.same_action_repeated == 2
    assert state.frame_unchanged_steps == 0
    state.record_step("ACTION1", frame_digest([[1]]))
    assert state.same_action_repeated == 1
    assert state.frame_unchanged_steps == 1

    # A blocked action may be repeated once to override; anything else re-arms.
    assert state.override_allowed(key) is False
    state.record_block(key)
    assert state.override_allowed(key) is True
    assert state.override_allowed("ACTION1") is False
    state.record_step(key, None)
    assert state.override_allowed(key) is False

    state.reset_level()
    assert (state.same_action_repeated, state.frame_unchanged_steps) == (0, 0)


# -- store: skills -----------------------------------------------------------


def test_store_initializes_empty_and_round_trips_skills(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert (store.root / "skills").is_dir()
    assert json.loads((store.root / "tools" / "index.json").read_text()) == {}
    assert json.loads((store.root / "hooks.json").read_text()) == []
    assert store.list_skills() == [] and store.list_tools() == [] and store.read_hooks() == []
    index = store.render_index()
    assert index.count("(none yet)") == 3

    text, ok = store.execute(
        "write_skill",
        {"name": "open-doors", "content": "# Open doors\n\nClick the key, then the door.\n"},
        frames_for_turn=frames_for_turn,
    )
    assert ok and "created" in text
    assert store.list_skills() == [{"name": "open-doors", "description": "Open doors"}]
    assert "- open-doors: Open doors" in store.render_index()
    text, ok = store.execute(
        "read_skill", {"name": "open-doors"}, frames_for_turn=frames_for_turn
    )
    assert ok and "Click the key" in text
    text, ok = store.execute(
        "read_skill", {"name": "missing"}, frames_for_turn=frames_for_turn
    )
    assert not ok and "does not exist" in text and "open-doors" in text
    text, ok = store.execute(
        "write_skill", {"name": "open-doors", "content": None}, frames_for_turn=frames_for_turn
    )
    assert ok and "deleted" in text
    assert store.list_skills() == []
    text, ok = store.execute(
        "write_skill", {"name": "Bad Name", "content": "x"}, frames_for_turn=frames_for_turn
    )
    assert not ok and "Invalid skill name" in text

    log = [json.loads(line) for line in store.log_path.read_text().splitlines()]
    assert [entry["event"] for entry in log] == ["write_skill", "write_skill"]
    assert log[1]["deleted"] is True


def test_store_enforces_skill_limits(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for index in range(MAX_SKILLS):
        store.write_skill(f"s{index}", f"skill {index}")
    with pytest.raises(ValueError, match=f"At most {MAX_SKILLS}"):
        store.write_skill("one-too-many", "x")
    store.write_skill("s0", "replacing an existing skill is fine")
    with pytest.raises(ValueError, match="at most"):
        store.write_skill("s1", "x" * (16 * 1024 + 1))
    with pytest.raises(ValueError, match="non-empty"):
        store.write_skill("s1", "   ")


# -- store: tools and the sandbox -------------------------------------------


def test_write_tool_checks_the_code_in_the_sandbox_and_run_tool_executes_it(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    text, ok = store.execute(
        "write_tool",
        {"name": "shape", "description": "Grid shape and top-left color.", "code": GOOD_TOOL},
        frames_for_turn=frames_for_turn,
    )
    assert ok and "created" in text
    assert store.list_tools() == [{"name": "shape", "description": "Grid shape and top-left color."}]
    assert (store.root / "tools" / "shape.py").is_file()

    text, ok = store.execute(
        "run_tool",
        {"name": "shape", "args": {"k": 1}},
        frames_for_turn=frames_for_turn,
    )
    assert ok, text
    payload = json.loads(text)
    assert payload["turn"] == 3 and payload["frames"] == 1
    assert payload["result"] == {"height": 64, "width": 64, "corner": 3, "args": {"k": 1}}

    text, ok = store.execute(
        "run_tool", {"name": "shape", "turn": 1}, frames_for_turn=frames_for_turn
    )
    assert ok and json.loads(text)["result"]["corner"] == 1

    # Errors from the sandbox are returned to the model, not raised.
    text, ok = store.execute(
        "write_tool",
        {"name": "broken", "description": "x", "code": "def analyze(frames, args):\n    return 1 +\n"},
        frames_for_turn=frames_for_turn,
    )
    assert not ok and "syntax error" in text
    text, ok = store.execute(
        "write_tool",
        {"name": "noanalyze", "description": "x", "code": "VALUE = 1\n"},
        frames_for_turn=frames_for_turn,
    )
    assert not ok and "must define analyze" in text
    text, ok = store.execute(
        "write_tool",
        {"name": "importfail", "description": "x", "code": "import no_such_module_xyz\n\ndef analyze(frames, args):\n    return 1\n"},
        frames_for_turn=frames_for_turn,
    )
    assert not ok and "failed to load in the sandbox" in text and "no_such_module_xyz" in text
    assert [tool["name"] for tool in store.list_tools()] == ["shape"]

    store.write_tool("raises", "x", "def analyze(frames, args):\n    raise KeyError('nope')\n")
    text, ok = store.execute(
        "run_tool", {"name": "raises"}, frames_for_turn=frames_for_turn
    )
    assert not ok
    assert json.loads(text)["error"] == "KeyError: 'nope'"

    text, ok = store.execute(
        "run_tool", {"name": "unknown"}, frames_for_turn=frames_for_turn
    )
    assert not ok and "does not exist" in text and "shape" in text

    text, ok = store.execute(
        "write_tool", {"name": "raises", "description": None, "code": None}, frames_for_turn=frames_for_turn
    )
    assert ok and "deleted" in text
    assert not (store.root / "tools" / "raises.py").exists()
    events = [json.loads(line)["event"] for line in store.log_path.read_text().splitlines()]
    # Only successful tool-channel writes and every run_tool are logged.
    assert events.count("write_tool") == 2 and events.count("run_tool") == 3


def test_run_tool_output_is_truncated(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.write_tool("big", "x", "def analyze(frames, args):\n    return 'y' * 20000\n")
    text, ok = store.execute(
        "run_tool", {"name": "big"}, frames_for_turn=frames_for_turn
    )
    assert ok
    assert len(text) < MAX_TOOL_OUTPUT_CHARS + 64
    assert text.endswith("chars]")


def test_sandbox_timeout_is_reported_not_raised(tmp_path: Path) -> None:
    store = make_store(tmp_path, timeout_seconds=1)
    store.write_tool("spin", "x", "def analyze(frames, args):\n    while True:\n        pass\n")
    outcome = store.run_tool("spin", [[[0]]], {})
    assert outcome["ok"] is False
    assert "exceeded the 1 s limit" in outcome["error"]


def _docker_sandbox_image() -> str | None:
    if shutil.which("docker") is None:
        return None
    for image in SANDBOX_IMAGES:
        probe = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=30
        )
        if probe.returncode == 0:
            return image
    return None


@pytest.mark.skipif(_docker_sandbox_image() is None, reason="no docker sandbox image")
def test_docker_sandbox_runs_the_tool_without_network(tmp_path: Path) -> None:
    sandbox = ToolSandbox(tmp_path / "sandbox", image=_docker_sandbox_image())
    assert sandbox.check(GOOD_TOOL) is None
    outcome = sandbox.run(GOOD_TOOL, [[[7] * 64 for _ in range(64)]], {"a": 1})
    assert outcome["ok"] is True, outcome
    assert outcome["result"]["corner"] == 7
    offline = sandbox.run(
        "import socket\n\ndef analyze(frames, args):\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n    return 'online'\n",
        [[[0]]],
        {},
    )
    assert offline["ok"] is False
    readonly = sandbox.run(
        "def analyze(frames, args):\n    open('/sandbox/x', 'w').write('x')\n    return 1\n",
        [[[0]]],
        {},
    )
    assert readonly["ok"] is False
    command = sandbox._docker_command(tmp_path, "probe")
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--read-only" in command and "--cap-drop" in command


# -- store: hooks -----------------------------------------------------------


def test_write_hooks_validates_and_updates_the_index(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    text, ok = store.execute(
        "write_hooks",
        {"rules": [{"name": "a", "event": "after_play", "when": {"level_steps_at_least": 50}, "note": "Slow level."}]},
        frames_for_turn=frames_for_turn,
    )
    assert ok and "1 rule" in text
    assert store.read_hooks()[0]["name"] == "a"
    assert "- a [after_play] when" in store.render_index()
    text, ok = store.execute(
        "write_hooks",
        {"rules": [{"name": "a", "event": "after_play", "when": {}, "note": "x"}]},
        frames_for_turn=frames_for_turn,
    )
    assert not ok and "non-empty" in text
    assert store.read_hooks()[0]["name"] == "a"  # unchanged on error
    text, ok = store.execute(
        "write_hooks", {"rules": []}, frames_for_turn=frames_for_turn
    )
    assert ok and store.read_hooks() == []

    # A second store over the same directory reads the persisted state.
    again = EvolveStore(store.root, log_path=store.log_path, sandbox=store.sandbox)
    assert again.read_hooks() == []


def test_execute_rejects_unknown_tools(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    text, ok = store.execute("bogus", {}, frames_for_turn=frames_for_turn)
    assert not ok and text == "Unknown tool."
