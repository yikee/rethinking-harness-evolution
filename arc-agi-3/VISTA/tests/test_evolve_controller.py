"""--evolve on both controllers: hooks and the skills / tools / hooks tools."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import GameAction, GameState

from vista_arc3.shared.evolve import EVOLVE_TOOL_NAMES, EvolveStore, ToolSandbox

RUNTIMES = ("claude", "codex")


def modules(runtime: str):
    return (
        importlib.import_module(f"vista_arc3.{runtime}.controller"),
        importlib.import_module(f"vista_arc3.{runtime}.dispatcher"),
        importlib.import_module(f"vista_arc3.{runtime}.tools"),
        importlib.import_module(f"vista_arc3.{runtime}.recovery"),
    )


class FakeEnvironment:
    def __init__(self, observations, actions=None):
        self.observations = list(observations)
        self.action_space = actions or [GameAction.ACTION1, GameAction.ACTION6]
        self.calls = []

    def step(self, action, data=None, reasoning=None):
        self.calls.append({"action": action, "data": data})
        return self.observations.pop(0)

    def reset(self):
        self.calls.append({"action": GameAction.RESET, "data": None})
        return self.observations.pop(0)


def observation(
    value: int = 0,
    *,
    state: GameState = GameState.NOT_FINISHED,
    levels_completed: int = 0,
    win_levels: int = 3,
):
    return SimpleNamespace(
        game_id="private-game-id",
        guid="private-guid",
        state=state,
        levels_completed=levels_completed,
        win_levels=win_levels,
        available_actions=[1, 6],
        action_input=None,
        full_reset=False,
        frame=[np.full((64, 64), value, dtype=np.uint8)],
    )


def make_evolve(tmp_path: Path, runtime: str):
    controller_module, dispatcher_module, _, _ = modules(runtime)
    visible = tmp_path / "visible"
    private = tmp_path / "private"
    frames = visible / "screenshots"
    frames.mkdir(parents=True)
    private.mkdir()
    store = EvolveStore(
        visible,
        log_path=private / "evolve_log.jsonl",
        sandbox=ToolSandbox(private / "sandbox", local=True),
    )
    store.initialize()
    guide = visible / "GUIDE.md"
    guide.write_text("No reliable model yet.\n", encoding="utf-8")

    def build(next_observations, *, notes: bool = True, initial=None):
        env = FakeEnvironment(next_observations)
        controller = controller_module.GameController(
            env=env,
            initial_observation=initial or observation(0),
            private_dir=private,
            frames_dir=frames,
            max_steps=50,
            max_invalid_retries=2,
            observation_mode="vision",
            render_scale=8,
            retry_boundary_marker=private / "retry_boundary.pending",
            reset_starts_fresh_session=False,
            working_path=visible / "WORKING.md" if notes else None,
            guide_path=guide if notes else None,
            evolve=store,
        )
        dispatcher = dispatcher_module.GameToolDispatcher(
            controller=controller,
            guide_path=guide,
            working_path=visible / "WORKING.md",
            notes_enabled=notes,
            evolve=store,
        )
        return controller, dispatcher, env

    return store, build


def play(dispatcher, call_id, **arguments):
    return dispatcher.execute("play", arguments, str(call_id))


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_assets_are_edited_during_play_and_level_completion_does_not_interrupt(
    tmp_path: Path, runtime: str
) -> None:
    """Like GUIDE.md / WORKING.md: the evolve tools work at any point of the run."""
    _, dispatcher_module, _, _ = modules(runtime)
    store, build = make_evolve(tmp_path, runtime)
    controller, dispatcher, env = build(
        [
            observation(1, levels_completed=0),
            observation(2, levels_completed=1),
            observation(3, levels_completed=1),
        ]
    )

    # Mid-level edits take effect immediately.
    written = dispatcher.execute(
        "write_skill",
        {"name": "level-1", "content": "# Level 1\nClick the bright cell.\n"},
        "1",
    )
    assert written.success and not written.interrupt_after
    assert "- level-1: Level 1" in store.render_index()
    first = play(dispatcher, 2, action="ACTION1")
    assert first.success and not first.interrupt_after

    # Completing a level only flags level_boundary; the turn goes on as before.
    crossed = play(dispatcher, 3, action="ACTION6", x=8, y=8)
    metadata = json.loads(crossed.text)
    assert metadata["level_boundary"] is True
    assert "reflection_boundary" not in metadata
    assert not crossed.interrupt_after and crossed.boundary_reason is None
    assert not (controller.private_dir / "reflection.pending").exists()
    assert not hasattr(controller, "reflection_pending")

    # Hooks written right after the level apply to the very next play.
    written = dispatcher.execute(
        "write_hooks",
        {
            "rules": [
                {
                    "name": "warn",
                    "event": "after_play",
                    "when": {"action_in": ["ACTION1"]},
                    "note": "Hook is live.",
                }
            ]
        },
        "4",
    )
    assert written.success
    following = play(dispatcher, 5, action="ACTION1")
    assert following.success and not following.interrupt_after
    assert json.loads(following.text)["hook_notes"] == ["[warn] Hook is live."]
    assert len(env.calls) == 3

    events = [json.loads(line)["event"] for line in store.log_path.read_text().splitlines()]
    assert events == ["write_skill", "write_hooks", "hooks_matched"]
    # There is no reflection tool anywhere in the dispatcher.
    assert dispatcher.execute("end_reflection", {}, "6").text == "Unknown tool."


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_winning_the_last_level_is_terminal_without_extra_flags(
    tmp_path: Path, runtime: str
) -> None:
    _, build = make_evolve(tmp_path, runtime)
    controller, dispatcher, _ = build(
        [observation(1, state=GameState.WIN, levels_completed=3)],
        initial=observation(0, levels_completed=2),
    )
    final = play(dispatcher, 1, action="ACTION1")
    metadata = json.loads(final.text)
    assert metadata["terminal"] is True
    assert "reflection_boundary" not in metadata and "level_boundary" not in metadata
    assert not final.interrupt_after
    assert controller.terminal


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_hooks_block_note_and_can_be_overridden(tmp_path: Path, runtime: str) -> None:
    store, build = make_evolve(tmp_path, runtime)
    store.write_hooks(
        [
            {
                "name": "no-top",
                "event": "before_play",
                "when": {
                    "action_in": ["ACTION6"],
                    "click_in_region": {"x0": 0, "y0": 0, "x1": 63, "y1": 3},
                },
                "note": "The top rows are decoration.",
                "block": True,
            },
            {
                "name": "stuck",
                "event": "after_play",
                "when": {"frame_unchanged_steps_at_least": 2},
                "note": "Two actions without a visible change.",
            },
            {
                "name": "spam",
                "event": "before_play",
                "when": {"same_action_repeated_at_least": 3},
                "note": "Third identical action in a row.",
            },
        ]
    )
    controller, dispatcher, env = build(
        [observation(1), observation(1), observation(1), observation(1), observation(1)]
    )

    # y=8 px -> grid row 1: inside the blocked region.
    blocked = play(dispatcher, 1, action="ACTION6", x=100, y=8)
    metadata = json.loads(blocked.text)
    assert not blocked.success and not blocked.interrupt_after
    assert metadata["hook_blocked"] is True
    assert metadata["hook_notes"] == ["[no-top] The top rows are decoration."]
    assert "Blocked by hook 'no-top'" in metadata["error"]
    assert env.calls == [] and controller.step_index == 0
    assert controller.attempt_index == 0  # not an invalid action
    assert controller.total_attempts == 1

    # The same action again overrides the block and is executed.
    overridden = play(dispatcher, 2, action="ACTION6", x=100, y=8)
    metadata = json.loads(overridden.text)
    assert overridden.success and metadata["action_applied"] is True
    assert metadata["hook_notes"][0] == "[no-top] The top rows are decoration."
    assert metadata["hook_notes"][1].startswith("hook_overridden: no-top")
    assert len(env.calls) == 1

    # A different action re-arms the block.
    other = play(dispatcher, 3, action="ACTION1")
    assert other.success and "hook_notes" not in json.loads(other.text)
    blocked_again = play(dispatcher, 4, action="ACTION6", x=100, y=8)
    assert json.loads(blocked_again.text)["hook_blocked"] is True
    # Outside the region the click is fine (frame unchanged -> after_play note).
    outside = play(dispatcher, 5, action="ACTION6", x=100, y=400)
    metadata = json.loads(outside.text)
    assert outside.success
    assert metadata["hook_notes"] == ["[stuck] Two actions without a visible change."]
    # Repeats: the third identical action in a row triggers the before_play note.
    play(dispatcher, 6, action="ACTION6", x=100, y=400)
    third = play(dispatcher, 7, action="ACTION6", x=100, y=400)
    notes = json.loads(third.text)["hook_notes"]
    assert "[spam] Third identical action in a row." in notes
    assert "[stuck] Two actions without a visible change." in notes

    log = [json.loads(line) for line in store.log_path.read_text().splitlines()]
    kinds = [entry["event"] for entry in log]
    assert kinds.count("hook_blocked") == 2 and kinds.count("hook_overridden") == 1
    assert "hooks_matched" in kinds
    action_log = [
        json.loads(line)
        for line in (controller.private_dir / "action_log.jsonl").read_text().splitlines()
    ]
    assert action_log[0]["hook_blocked"] == "no-top" and "result" not in action_log[0]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_evolve_combines_with_no_guide_working(tmp_path: Path, runtime: str) -> None:
    _, dispatcher_module, tools_module, _ = modules(runtime)
    names = {
        tool["name"]
        for tool in tools_module.build_tools(
            include_notes=False, include_evolve=True, include_compact_checkpoint=False
        )
    }
    assert EVOLVE_TOOL_NAMES <= names
    assert not names & tools_module.NOTE_TOOL_NAMES
    plain = {tool["name"] for tool in tools_module.build_tools(include_compact_checkpoint=False)}
    assert not plain & EVOLVE_TOOL_NAMES

    store, build = make_evolve(tmp_path, runtime)
    controller, dispatcher, _ = build(
        [observation(1, levels_completed=1)], notes=False
    )
    assert dispatcher.execute("read_guide", {}, "1").text == "Unknown tool."
    assert dispatcher.execute("write_skill", {"name": "s", "content": "# S\nx"}, "2").success
    crossed = play(dispatcher, 3, action="ACTION1")
    assert crossed.success and not crossed.interrupt_after
    assert json.loads(crossed.text)["level_boundary"] is True
    assert dispatcher.execute("write_guide", {"content": "x"}, "4").text == "Unknown tool."
    assert dispatcher.execute("read_skill", {"name": "s"}, "5").success

    # Without the store the evolve tools do not exist at all.
    bare = dispatcher_module.GameToolDispatcher(
        controller=controller,
        guide_path=tmp_path / "GUIDE.md",
        working_path=tmp_path / "WORKING.md",
        notes_enabled=False,
    )
    assert bare.execute("write_skill", {"name": "s", "content": "x"}, "6").text == "Unknown tool."
    assert bare.execute("end_reflection", {}, "7").text == "Unknown tool."


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_run_tool_reads_the_archived_grids_of_any_turn(tmp_path: Path, runtime: str) -> None:
    store, build = make_evolve(tmp_path, runtime)
    controller, dispatcher, _ = build([observation(5), observation(9)])
    play(dispatcher, 1, action="ACTION1")
    play(dispatcher, 2, action="ACTION1")
    store.write_tool(
        "corner", "Top-left color.", "def analyze(frames, args):\n    return frames[-1][0][0]\n"
    )
    assert controller.frames_for_turn(None)[0] == 2
    with pytest.raises(ValueError, match="turn must be in 0..2"):
        controller.frames_for_turn(3)
    current = dispatcher.execute("run_tool", {"name": "corner"}, "3")
    assert current.success and json.loads(current.text)["result"] == 9
    past = dispatcher.execute("run_tool", {"name": "corner", "turn": 1}, "4")
    assert past.success and json.loads(past.text)["result"] == 5
    initial = dispatcher.execute("run_tool", {"name": "corner", "turn": 0}, "5")
    assert initial.success and json.loads(initial.text)["result"] == 0
    bad = dispatcher.execute("run_tool", {"name": "corner", "turn": 7}, "6")
    assert not bad.success and "turn must be in 0..2" in bad.text
    # run_tool never touches the game.
    assert controller.step_index == 2 and controller.total_attempts == 2
