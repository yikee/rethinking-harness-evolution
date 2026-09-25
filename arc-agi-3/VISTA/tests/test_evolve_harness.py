"""--evolve wiring: runners expose the tools and index; run_game prepares the assets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import GameState

import vista_arc3.claude.harness as claude_harness
import vista_arc3.codex.harness as codex_harness
import vista_arc3.shared.evolve as evolve_module
from vista_arc3.claude.runner import (
    CLAUDE_PERMISSION_MODE,
    PINNED_CLAUDE_VERSION,
    ClaudeCodeRunner,
    ClaudeResult,
    validate_instruction_envelope,
)
from vista_arc3.codex.runner import DockerCodexRunner, prepare_codex_home
from vista_arc3.shared.evolve import (
    EVOLVE_INSTRUCTIONS,
    EVOLVE_TOOL_NAMES,
    EvolveStore,
    ToolSandbox,
)


def make_store(tmp_path: Path, visible: Path) -> EvolveStore:
    store = EvolveStore(
        visible,
        log_path=tmp_path / "private" / "evolve_log.jsonl",
        sandbox=ToolSandbox(tmp_path / "private" / "sandbox", local=True),
    )
    store.initialize()
    return store


class StubController:
    terminal = False
    step_index = 0
    reset_starts_fresh_session = False
    reset_requires_retry_state = False
    compact_checkpoint_marker = None
    compact_checkpoint_ready = None
    compact_restore_marker = None
    retry_boundary_pending = False

    def initial_image_paths(self):
        return []

    def frames_for_turn(self, turn):
        return 0, [[[0] * 64 for _ in range(64)]]


# -- runners -----------------------------------------------------------------


def test_claude_runner_exposes_evolve_tools_and_a_live_index(tmp_path: Path) -> None:
    visible = tmp_path / "visible"
    (visible / "screenshots").mkdir(parents=True)
    (visible / "AGENTS.md").write_text("Play the game.\n", encoding="utf-8")
    (visible / "GUIDE.md").write_text("No model.\n", encoding="utf-8")
    store = make_store(tmp_path, visible)
    runner = ClaudeCodeRunner(
        visible_dir=visible,
        claude_config_dir=tmp_path / "config",
        io_dir=tmp_path / "io",
        controller=StubController(),
        effort="max",
        oauth_token="test-oauth-token",
        evolve=store,
    )
    assert runner.dispatcher.evolve is store
    mcp = runner.claude_config_dir / "mcp.json"
    mcp.write_text("{}", encoding="utf-8")

    def allowed():
        command = runner._docker_command(
            mcp_config_path=mcp,
            settings_path=runner.claude_config_dir / "settings.json",
            bridge_host="host.docker.internal",
            bridge_port=43123,
            session_id="00000000-0000-4000-8000-000000000000",
            resume=False,
        )
        return command, command[command.index("--allowedTools") + 1].split(",")

    command, allowed_tools = allowed()
    assert {f"mcp__game__{name}" for name in EVOLVE_TOOL_NAMES} <= set(allowed_tools)
    assert "mcp__game__read_guide" in allowed_tools
    system_prompt = command[command.index("--system-prompt") + 1]
    assert system_prompt.startswith("Play the game.")
    assert "## Current skills, analysis tools and hooks" in system_prompt
    assert "(none yet)" in system_prompt

    # The index is regenerated for every segment from the current store state.
    store.write_skill("corners", "# Corner scan\nLook at the four corners first.")
    command, _ = allowed()
    assert "- corners: Corner scan" in command[command.index("--system-prompt") + 1]

    envelope = {
        "tools": sorted(allowed_tools),
        "skills": [],
        "plugins": [],
        "slash_commands": [],
        "permissionMode": CLAUDE_PERMISSION_MODE,
        "claude_code_version": PINNED_CLAUDE_VERSION,
        "mcp_servers": [{"name": "game", "status": "connected"}],
        "model": "claude-opus-5",
    }
    validate_instruction_envelope(envelope, display_size=512, include_evolve=True)
    with pytest.raises(RuntimeError, match="unexpected tool surface"):
        validate_instruction_envelope(envelope, display_size=512)


def test_claude_mcp_host_lists_evolve_tools_only_when_enabled(tmp_path: Path) -> None:
    from vista_arc3.claude.mcp_host import ClaudeGameMcpHost

    def listed(evolve_enabled: bool) -> set[str]:
        host = ClaudeGameMcpHost(
            bind_host="127.0.0.1",
            port=0,
            token="t",
            dispatcher=SimpleNamespace(controller=StubController()),
            display_size=512,
            log_path=tmp_path / f"bridge-{evolve_enabled}.jsonl",
            evolve_enabled=evolve_enabled,
        )
        response = host._handle({"token": "t", "method": "tools/list"})
        return {tool["name"] for tool in response["result"]["tools"]}

    assert EVOLVE_TOOL_NAMES <= listed(True)
    assert not listed(False) & EVOLVE_TOOL_NAMES


def test_codex_runner_exposes_evolve_tools_and_a_live_index(tmp_path: Path) -> None:
    visible = tmp_path / "visible"
    visible.mkdir()
    (visible / "GUIDE.md").write_text("Initial model\n", encoding="utf-8")
    (visible / "AGENTS.md").write_text("Game instructions\n", encoding="utf-8")
    (visible / "screenshots").mkdir()
    (visible / ".codex").mkdir()
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")
    codex_home = prepare_codex_home(tmp_path / "codex-home")
    (codex_home / "hooks.json").write_text('{"hooks": {}}\n', encoding="utf-8")
    store = make_store(tmp_path, visible)
    runner = DockerCodexRunner(
        visible_dir=visible,
        codex_home=codex_home,
        auth_file=auth_file,
        io_dir=tmp_path / "io",
        controller=StubController(),
        display_size=512,
        evolve=store,
    )
    assert runner.tool_dispatcher.evolve is store
    tools = {tool["name"] for tool in runner._dynamic_tools()[0]["tools"]}
    assert EVOLVE_TOOL_NAMES <= tools and "save_compact_checkpoint" in tools

    instructions = runner._thread_parameters(100_000)["developerInstructions"]
    assert instructions.startswith("Game instructions")
    assert "(none yet)" in instructions
    store.write_hooks(
        [{"name": "r", "event": "after_play", "when": {"level_steps_at_least": 9}, "note": "Slow."}]
    )
    assert "- r [after_play]" in runner._thread_parameters(100_000)["developerInstructions"]

    # The evolve tools reach the shared store through the dispatcher.
    execution = runner._execute_tool("read_skill", {"name": "nope"}, "call-1")
    assert execution.success is False and "does not exist" in execution.text
    execution = runner._execute_tool("end_reflection", {}, "call-2")
    assert execution.success is False and execution.text == "Unknown tool."


# -- harness loops -----------------------------------------------------------


def claude_result(tmp_path: Path, session: str | None, returncode: int, **overrides) -> ClaudeResult:
    fields = dict(
        segment=0,
        returncode=returncode,
        process_returncode=returncode,
        final_message="",
        session_id=session,
        resolved_models=("claude-opus-4-6",),
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        input_path=tmp_path / "input",
        command_path=tmp_path / "command",
        bridge_log_path=tmp_path / "bridge",
        compact_events_path=tmp_path / "compact",
        usage=None,
        api_error_status=None,
        context_boundary=None,
        compact_boundaries=(),
        decision_lease_failure=None,
        unconfirmed_play_calls=0,
    )
    fields.update(overrides)
    return ClaudeResult(**fields)


def test_level_completion_is_not_a_harness_event() -> None:
    """Skills / tools / hooks are edited through game tools during play; there is
    no reflection turn, so neither runner nor harness has any reflection hook."""
    for runner_class in (ClaudeCodeRunner, DockerCodexRunner):
        assert not hasattr(runner_class, "resume_reflection")
        assert not hasattr(runner_class, "run_level_recovery")
    for harness in (claude_harness, codex_harness):
        assert not hasattr(harness, "REFLECTION_MARKER")
    assert "end_reflection" not in EVOLVE_TOOL_NAMES


# -- run_game setup ----------------------------------------------------------


def _winning_observation():
    return SimpleNamespace(
        game_id="s5i5",
        guid="guid",
        state=GameState.WIN,
        levels_completed=8,
        win_levels=8,
        available_actions=[1, 6],
        action_input=None,
        full_reset=False,
        frame=[np.zeros((64, 64), dtype=np.uint8)],
    )


def _fake_arcade(observation):
    env = SimpleNamespace(
        observation_space=observation,
        action_space=[],
        _session=SimpleNamespace(cookies={}),
    )
    arc = SimpleNamespace(
        get_environments=lambda: ["s5i5"],
        make=lambda *args, **kwargs: env,
        _session=SimpleNamespace(cookies={}),
    )
    return arc, env


def test_codex_run_game_with_evolve_prepares_empty_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_auth = tmp_path / "api-auth.json"
    monkeypatch.setenv("ARC3_CODEX_API_KEY_AUTH_FILE", str(api_auth))
    monkeypatch.delenv("ARC3_CODEX_AUTH", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-test")
    monkeypatch.setattr(evolve_module, "default_sandbox_image", lambda: "sandbox:test")
    arc, _ = _fake_arcade(_winning_observation())
    monkeypatch.setattr(codex_harness, "Arcade", lambda **kwargs: arc)
    monkeypatch.setattr(codex_harness, "open_evaluation_scorecard", lambda a: "card-1")
    monkeypatch.setattr(
        codex_harness, "finalize_scorecard", lambda **kwargs: ({"scorecard_closed": True}, None)
    )
    captured: dict = {}

    def fake_build_runner(args, visible_dir, codex_home, auth_file, io_dir, controller,
                          reasoning_effort, compact_thresholds, *, notes_enabled=True, evolve=None):
        captured["evolve"] = evolve
        captured["notes_enabled"] = notes_enabled
        return SimpleNamespace()

    def fake_loop(*, runner, controller, prompt, results, **kwargs):
        captured["controller"] = controller
        results.append(SimpleNamespace(session_id="s", returncode=0, usage=None, segment=0))
        return results

    monkeypatch.setattr(codex_harness, "build_runner", fake_build_runner)
    monkeypatch.setattr(codex_harness, "run_task_with_compact_checkpoints", fake_loop)

    run_dir = tmp_path / "evolve"
    outcome = codex_harness.run_game(
        codex_harness.build_parser().parse_args(["--game-id", "s5i5", "--evolve", "--no-guide-working"]),
        run_dir=run_dir,
    )
    assert outcome.failure is None
    visible = run_dir / "codex_visible"
    assert (visible / "skills").is_dir() and not any((visible / "skills").iterdir())
    assert json.loads((visible / "tools" / "index.json").read_text()) == {}
    assert json.loads((visible / "hooks.json").read_text()) == []
    assert not (visible / "GUIDE.md").exists()
    agents = (visible / "AGENTS.md").read_text(encoding="utf-8")
    assert agents.rstrip().endswith(EVOLVE_INSTRUCTIONS.strip())
    assert "GUIDE.md" not in agents.split("## Self-improvement")[0]
    store = captured["evolve"]
    assert isinstance(store, EvolveStore) and store.root == visible
    assert store.log_path == run_dir / "private" / "evolve_log.jsonl"
    controller = captured["controller"]
    assert controller.evolve is store
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evolve_enabled"] is True
    assert manifest["guide_working_enabled"] is False
    assert manifest["evolve_sandbox_image"] == "sandbox:test"

    run_dir2 = tmp_path / "plain"
    outcome = codex_harness.run_game(
        codex_harness.build_parser().parse_args(["--game-id", "s5i5"]), run_dir=run_dir2
    )
    assert outcome.failure is None
    assert captured["evolve"] is None
    assert captured["controller"].evolve is None
    assert not (run_dir2 / "codex_visible" / "skills").exists()
    assert "## Self-improvement" not in (run_dir2 / "codex_visible" / "AGENTS.md").read_text()
    manifest2 = json.loads((run_dir2 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest2["evolve_enabled"] is False
    assert "evolve_sandbox_image" not in manifest2


def test_claude_run_game_with_evolve_prepares_empty_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ARC3_CLAUDE_AUTH", raising=False)
    monkeypatch.setattr(evolve_module, "default_sandbox_image", lambda: "sandbox:test")
    arc, _ = _fake_arcade(_winning_observation())
    monkeypatch.setattr(claude_harness, "Arcade", lambda **kwargs: arc)
    monkeypatch.setattr(claude_harness, "open_evaluation_scorecard", lambda a: "card-1")
    monkeypatch.setattr(
        claude_harness, "finalize_scorecard", lambda **kwargs: ({"scorecard_closed": True}, None)
    )
    monkeypatch.setattr(claude_harness, "validate_claude_runtime", lambda **kwargs: "claude:test")
    monkeypatch.setattr(claude_harness, "_safe_claude_version", lambda: "test")
    monkeypatch.setattr(
        claude_harness,
        "run_runtime_preflight",
        lambda **kwargs: (SimpleNamespace(), "claude-opus-5"),
    )
    captured: dict = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured["runner_kwargs"] = kwargs

    def fake_loop(*, runner, controller, prompt, results, **kwargs):
        captured["controller"] = controller
        results.append(
            claude_result(tmp_path, "s", 0)
        )
        return results

    monkeypatch.setattr(claude_harness, "ClaudeCodeRunner", FakeRunner)
    monkeypatch.setattr(claude_harness, "run_task_with_native_context", fake_loop)

    run_dir = tmp_path / "evolve"
    outcome = claude_harness.run_game(
        claude_harness.build_parser().parse_args(["--game-id", "s5i5", "--evolve"]),
        run_dir=run_dir,
    )
    assert outcome.failure is None, outcome.failure
    visible = run_dir / "player"
    assert (visible / "skills").is_dir()
    assert json.loads((visible / "hooks.json").read_text()) == []
    assert (visible / "GUIDE.md").is_file()
    agents = (visible / "AGENTS.md").read_text(encoding="utf-8")
    assert "GUIDE.md" in agents and agents.rstrip().endswith(EVOLVE_INSTRUCTIONS.strip())
    store = captured["runner_kwargs"]["evolve"]
    assert isinstance(store, EvolveStore) and store.root == visible
    controller = captured["controller"]
    assert controller.evolve is store
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evolve_enabled"] is True and manifest["evolve_sandbox_image"] == "sandbox:test"


def test_batch_passes_evolve_to_every_game() -> None:
    from vista_arc3.batch import build_parser, config_from_args, game_args

    parser = build_parser()
    config = config_from_args(parser.parse_args(["--evolve"]))
    assert config.evolve is True
    assert game_args(config, "s5i5").evolve is True
    plain = config_from_args(parser.parse_args([]))
    assert plain.evolve is False and game_args(plain, "s5i5").evolve is False
