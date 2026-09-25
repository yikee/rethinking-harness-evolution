import json
from pathlib import Path

import pytest
from PIL import Image

from vista_arc3.claude import runner as claude_runner
from vista_arc3.claude.runner import (
    CLAUDE_AUTO_COMPACT_PERCENT,
    CLAUDE_AUTO_COMPACT_WINDOW,
    CLAUDE_PERMISSION_MODE,
    CONTAINER_CLAUDE_BIN,
    PINNED_CLAUDE_VERSION,
    ClaudeCodeRunner,
    ClaudePreflightController,
    ClaudeResult,
    DecisionLeaseWatchdog,
    acknowledge_completed_play_tools,
    api_error_status_from_message,
    claude_process_environment,
    credential_handoff_is_safe,
    decision_lease_failure_from_result,
    default_claude_image,
    record_pending_play_tool_uses,
    resolve_claude_credential,
    resolved_model_id,
    sanitize_stream_message,
    stream_user_message,
    validate_claude_runtime,
    validate_instruction_envelope,
)
from vista_arc3.claude.tools import build_tools


class StubController:
    compact_restore_marker = None
    compact_checkpoint_marker = None
    retry_boundary_pending = False
    reset_starts_fresh_session = False

    @staticmethod
    def initial_image_paths():
        return ()

    @staticmethod
    def handle(request):
        return {"ok": False, "metadata": {"error": "not used"}}


def test_preflight_uses_the_formal_same_session_reset_contract() -> None:
    assert ClaudePreflightController.reset_starts_fresh_session is False


def test_default_claude_image_uses_the_configured_or_default_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARC3_CLAUDE_IMAGE", raising=False)
    assert default_claude_image() == "arc3-claude-player:0.1"

    monkeypatch.setenv("ARC3_CLAUDE_IMAGE", "custom-claude:dev")
    assert default_claude_image() == "custom-claude:dev"


def test_validate_runtime_checks_the_docker_image_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if command[:5] == ["docker", "run", "--rm", "--pull=never", "--entrypoint"]:
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "2.1.220 (Claude Code)\n", "stderr": ""},
            )()
        raise AssertionError(command)

    monkeypatch.setattr(claude_runner.subprocess, "run", fake_run)
    assert (
        validate_claude_runtime("custom-claude:dev", oauth_token="token")
        == "custom-claude:dev"
    )
    assert calls[0] == ["docker", "image", "inspect", "custom-claude:dev"]
    assert calls[1][:7] == [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--entrypoint",
        CONTAINER_CLAUDE_BIN,
        "custom-claude:dev",
    ]


def test_api_error_status_accepts_claude_result_and_stream_spellings() -> None:
    assert api_error_status_from_message({"api_error_status": 500}) == 500
    assert api_error_status_from_message({"apiErrorStatus": 529}) == 529
    assert api_error_status_from_message({"api_error_status": "500"}) is None


def test_result_errors_identify_output_limits_and_runtime_failures() -> None:
    assert decision_lease_failure_from_result(
        {
            "type": "result",
            "is_error": True,
            "terminal_reason": "api_error",
            "result": (
                "API Error: Claude's response exceeded the 64000 output token "
                "maximum."
            ),
        }
    ) == "output_token_limit"
    assert decision_lease_failure_from_result(
        {
            "type": "result",
            "is_error": True,
            "subtype": "error_during_execution",
        }
    ) == "runtime_exit"
    assert decision_lease_failure_from_result(
        {"type": "result", "is_error": False, "subtype": "success"}
    ) is None


def test_decision_lease_resets_after_tools_and_waits_for_in_flight_calls() -> None:
    lease = DecisionLeaseWatchdog(timeout_seconds=30, started_at=100)

    assert lease.expired(
        now=131,
        completed_tool_calls=0,
        tool_calls_in_flight=1,
        last_tool_completed_at=None,
    ) is False
    assert lease.expired(
        now=140,
        completed_tool_calls=1,
        tool_calls_in_flight=0,
        last_tool_completed_at=139,
    ) is False
    assert lease.expired(
        now=170,
        completed_tool_calls=1,
        tool_calls_in_flight=0,
        last_tool_completed_at=139,
    ) is True


def test_credential_handoff_waits_for_live_results_but_recovers_known_state() -> None:
    assert credential_handoff_is_safe(
        process_running=True,
        tool_calls_in_flight=0,
        pending_play_tool_ids={"play-1"},
        unconfirmed_play_calls=0,
    ) is False
    assert credential_handoff_is_safe(
        process_running=False,
        tool_calls_in_flight=0,
        pending_play_tool_ids={"play-1"},
        unconfirmed_play_calls=0,
    ) is True
    assert credential_handoff_is_safe(
        process_running=False,
        tool_calls_in_flight=0,
        pending_play_tool_ids=set(),
        unconfirmed_play_calls=1,
    ) is False


def make_runner(
    tmp_path: Path,
    *,
    effort: str = "max",
    compact_window: int | None = None,
    compact_percent: int | None = None,
    notes_enabled: bool = True,
    auth_mode: str | None = None,
) -> ClaudeCodeRunner:
    visible = tmp_path / "visible"
    (visible / "screenshots").mkdir(parents=True)
    (visible / "AGENTS.md").write_text("Play the game.\n", encoding="utf-8")
    if notes_enabled:
        (visible / "GUIDE.md").write_text("No model.\n", encoding="utf-8")
    controller = StubController()
    controller.compact_checkpoint_marker = tmp_path / "io" / "checkpoint.requested"
    controller.compact_checkpoint_ready = tmp_path / "io" / "checkpoint.ready"
    controller.compact_restore_marker = tmp_path / "io" / "restore.pending"
    compact_options = {}
    if compact_window is not None or compact_percent is not None:
        compact_options = {
            "auto_compact_window": compact_window,
            "auto_compact_percent": compact_percent,
        }
    return ClaudeCodeRunner(
        visible_dir=visible,
        claude_config_dir=tmp_path / "config",
        io_dir=tmp_path / "io",
        controller=controller,
        effort=effort,
        oauth_token="test-oauth-token",
        auth_mode=auth_mode,
        notes_enabled=notes_enabled,
        **compact_options,
    )


def test_no_guide_working_runner_drops_note_tools_and_checkpoint_hook(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path, notes_enabled=False)
    assert runner.dispatcher.notes_enabled is False
    mcp = runner.claude_config_dir / "mcp.json"
    settings = runner.claude_config_dir / "settings.json"
    mcp.write_text("{}", encoding="utf-8")

    command = runner._docker_command(
        mcp_config_path=mcp,
        settings_path=settings,
        bridge_host="host.docker.internal",
        bridge_port=43123,
        session_id="00000000-0000-4000-8000-000000000000",
        resume=False,
    )
    allowed_tools = command[command.index("--allowedTools") + 1].split(",")
    assert set(allowed_tools) == {
        "mcp__game__play",
        "mcp__game__inspect",
        "mcp__game__read_pixels",
        "mcp__game__history",
    }
    assert not any(tool.endswith(("_guide", "_working")) for tool in allowed_tools)

    # The pre-compact hook still logs compaction events but no longer blocks on
    # a WORKING.md checkpoint the player cannot write.
    runner._write_settings(settings, "compact-events.jsonl")
    parsed_settings = json.loads(settings.read_text(encoding="utf-8"))
    hook_command = parsed_settings["hooks"]["PreCompact"][0]["hooks"][0]["command"]
    assert "ARC3_COMPACT_EVENT_LOG=/claude-io/compact-events.jsonl" in hook_command
    assert "ARC3_COMPACT_CHECKPOINT_REQUEST" not in hook_command
    assert "ARC3_COMPACT_CHECKPOINT_READY" not in hook_command

    # Envelope validation follows the reduced tool surface in both directions.
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
    validate_instruction_envelope(envelope, display_size=512, include_notes=False)
    with pytest.raises(RuntimeError, match="unexpected tool surface"):
        validate_instruction_envelope(envelope, display_size=512)
    full_envelope = {
        **envelope,
        "tools": sorted(
            f'mcp__game__{tool["name"]}' for tool in build_tools(512)
        ),
    }
    with pytest.raises(RuntimeError, match="unexpected tool surface"):
        validate_instruction_envelope(
            full_envelope, display_size=512, include_notes=False
        )


def test_child_environment_contains_only_the_assigned_claude_credential() -> None:
    child = claude_process_environment(
        "assigned-token",
        {
            "PATH": "/usr/bin",
            "CLAUDE_CODE_OAUTH_TOKEN": "primary-token",
            "CLAUDE_CODE_OAUTH_TOKEN_2": "second-token",
            "CLAUDE_CODE_OAUTH_TOKEN_3": "third-token",
            "ANTHROPIC_API_KEY": "api-key",
            "ARC_API_KEY": "arc-key",
        },
    )

    assert child == {
        "PATH": "/usr/bin",
        "CLAUDE_CODE_OAUTH_TOKEN": "assigned-token",
    }


def test_api_key_mode_exports_only_anthropic_api_key(tmp_path: Path) -> None:
    child = claude_process_environment(
        "sk-ant-api-assigned",
        {
            "PATH": "/usr/bin",
            "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
            "ANTHROPIC_API_KEY": "stale-key",
        },
        auth_mode="api-key",
    )
    assert child == {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-ant-api-assigned"}

    with pytest.raises(ValueError, match="auth mode"):
        claude_process_environment("x", {}, auth_mode="chatgpt")

    # The container only receives the variable for the chosen billing mode.
    def docker_command(runner):
        mcp = runner.claude_config_dir / "mcp.json"
        mcp.write_text("{}", encoding="utf-8")
        return runner._docker_command(
            mcp_config_path=mcp,
            settings_path=runner.claude_config_dir / "settings.json",
            bridge_host="host.docker.internal",
            bridge_port=43123,
            session_id="00000000-0000-4000-8000-000000000000",
            resume=False,
        )

    runner = make_runner(tmp_path, auth_mode="api-key")
    assert runner.auth_mode == "api-key"
    command = docker_command(runner)
    assert "ANTHROPIC_API_KEY" in command
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in command
    default_runner = make_runner(tmp_path / "oauth")
    assert default_runner.auth_mode == "oauth"
    assert "CLAUDE_CODE_OAUTH_TOKEN" in docker_command(default_runner)


def test_resolve_claude_credential_prefers_the_api_key_in_auto_mode() -> None:
    both = {"ANTHROPIC_API_KEY": "sk-ant-api", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat"}
    assert resolve_claude_credential(environ=both) == ("api-key", "sk-ant-api")
    assert resolve_claude_credential("oauth", environ=both) == ("oauth", "sk-ant-oat")
    assert resolve_claude_credential(
        environ={"CLAUDE_CODE_OAUTH_TOKEN": " sk-ant-oat "}
    ) == ("oauth", "sk-ant-oat")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        resolve_claude_credential("api-key", environ={"CLAUDE_CODE_OAUTH_TOKEN": "t"})
    with pytest.raises(RuntimeError, match="CLAUDE_CODE_OAUTH_TOKEN is not set"):
        resolve_claude_credential(environ={})
    with pytest.raises(ValueError, match="auth mode"):
        resolve_claude_credential("chatgpt", environ=both)


def test_nonterminal_continuation_resumes_without_reinjecting_visuals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(tmp_path)
    captured = {}
    expected = object()

    def capture_segment(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(runner, "_run_segment", capture_segment)

    actual = runner.resume_nonterminal_task("same-session", "Continue.")

    assert actual is expected
    assert captured == {
        "prompt": "Continue.",
        "initial_images": (),
        "resume_session_id": "same-session",
    }


def test_docker_command_has_only_the_game_mcp_surface(
    tmp_path: Path,
) -> None:
    runner = make_runner(tmp_path)
    mcp = runner.claude_config_dir / "mcp.json"
    settings = runner.claude_config_dir / "settings.json"
    mcp.write_text("{}", encoding="utf-8")
    settings.write_text("{}", encoding="utf-8")
    command = runner._docker_command(
        mcp_config_path=mcp,
        settings_path=settings,
        bridge_host="host.docker.internal",
        bridge_port=43123,
        session_id="00000000-0000-4000-8000-000000000000",
        resume=False,
    )

    assert command[:3] == ["docker", "run", "-i"]
    assert ["--memory", "16g"] == command[
        command.index("--memory") : command.index("--memory") + 2
    ]
    assert ["--cpus", "4"] == command[
        command.index("--cpus") : command.index("--cpus") + 2
    ]
    assert CONTAINER_CLAUDE_BIN in command
    assert "--strict-mcp-config" in command
    assert command[command.index("--tools") + 1] == ""
    allowed_tools = command[command.index("--allowedTools") + 1].split(",")
    assert set(allowed_tools) == {
        f'mcp__game__{tool["name"]}'
        for tool in build_tools(512, include_compact_checkpoint=False)
    }
    assert "mcp__game__save_compact_checkpoint" not in allowed_tools

    resumed_command = runner._docker_command(
        mcp_config_path=mcp,
        settings_path=settings,
        bridge_host="host.docker.internal",
        bridge_port=43123,
        session_id="00000000-0000-4000-8000-000000000001",
        resume=True,
    )
    resumed_tools = resumed_command[
        resumed_command.index("--allowedTools") + 1
    ].split(",")
    assert "mcp__game__save_compact_checkpoint" not in resumed_tools
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--prompt-suggestions") + 1] == "false"
    assert command[command.index("--effort") + 1] == "max"
    # Partial messages are only requested by the opt-in stall watchdog.
    assert "--include-partial-messages" not in command
    runner.stream_stall_timeout = 900
    watched = runner._docker_command(
        mcp_config_path=tmp_path / "config" / "mcp.json",
        settings_path=tmp_path / "config" / "settings.json",
        bridge_host="host.docker.internal",
        bridge_port=43123,
        session_id="00000000-0000-4000-8000-000000000001",
        resume=False,
        include_partial_messages=runner.stream_stall_timeout is not None,
    )
    assert "--include-partial-messages" in watched
    assert "--disable-slash-commands" in command
    assert "--no-chrome" in command
    assert ["--add-host", "host.docker.internal:host-gateway"] == command[
        command.index("--add-host") : command.index("--add-host") + 2
    ]
    assert "--safe-mode" not in command
    assert "seccomp=unconfined" not in command
    assert "apparmor=unconfined" not in command
    assert "CLAUDE_CODE_OAUTH_TOKEN" in command
    assert "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1" in command
    assert "CLAUDE_CODE_DISABLE_CLAUDE_MDS=1" in command
    assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1" in command
    assert "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1" in command
    assert (
        f"CLAUDE_CODE_AUTO_COMPACT_WINDOW={CLAUDE_AUTO_COMPACT_WINDOW}"
        in command
    )
    assert (
        f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE={CLAUDE_AUTO_COMPACT_PERCENT}"
        in command
    )
    assert not any("ARC_API_KEY" in value for value in command)
    assert not any(str(runner.visible_dir) in value for value in command)
    assert not any(str(runner.guide_path) in value for value in command)
    assert not any("/run/arc3-game" in value for value in command)


def test_runner_rejects_unknown_effort(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported Claude effort"):
        make_runner(tmp_path, effort="ultra")


def test_auto_compact_threshold_can_be_lowered_for_canaries(tmp_path: Path) -> None:
    runner = make_runner(
        tmp_path,
        compact_window=64_000,
        compact_percent=50,
    )

    assert runner._auto_compact_environment() == [
        "-e",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW=64000",
        "-e",
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50",
    ]


def test_checkpoint_uses_the_normal_auto_compact_threshold(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)

    assert runner._auto_compact_environment() == [
        "-e",
        f"CLAUDE_CODE_AUTO_COMPACT_WINDOW={CLAUDE_AUTO_COMPACT_WINDOW}",
        "-e",
        f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE={CLAUDE_AUTO_COMPACT_PERCENT}",
    ]


def test_stream_message_embeds_visual_but_private_log_only_keeps_digest(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "visual.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image_path)

    message = stream_user_message("Current state.", [image_path])
    sanitized = sanitize_stream_message(message)

    source = message["message"]["content"][1]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert isinstance(source["data"], str)
    logged = sanitized["message"]["content"][1]["source"]["data"]
    assert set(logged) == {"encoded_chars", "sha256"}
    assert json.dumps(logged) not in json.dumps(message)


def test_mcp_and_hook_commands_remove_parent_credentials(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    mcp = runner.claude_config_dir / "mcp.json"
    settings = runner.claude_config_dir / "settings.json"

    runner._write_mcp_config(
        mcp,
        "bridge-token",
        bridge_host="host.docker.internal",
        bridge_port=43123,
    )
    runner._write_settings(settings, "compact-events.jsonl")

    game = json.loads(mcp.read_text(encoding="utf-8"))["mcpServers"]["game"]
    assert game["command"] == "/usr/bin/env"
    assert game["env"] == {
        "ARC3_GAME_HOST": "host.docker.internal",
        "ARC3_GAME_PORT": "43123",
        "ARC3_GAME_TOKEN": "bridge-token",
    }
    for name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ARC_API_KEY"):
        index = game["args"].index(name)
        assert game["args"][index - 1] == "-u"
    parsed_settings = json.loads(settings.read_text(encoding="utf-8"))
    assert parsed_settings["autoCompactEnabled"] is True
    assert set(parsed_settings["hooks"]) == {
        "PreCompact",
        "PostCompact",
        "Stop",
    }
    hook_command = parsed_settings["hooks"][
        "PreCompact"
    ][0]["hooks"][0]["command"]
    assert parsed_settings["hooks"]["Stop"][0]["hooks"][0]["command"] == (
        hook_command
    )
    for name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ARC_API_KEY"):
        assert f"-u {name}" in hook_command
    assert "ARC3_COMPACT_EVENT_LOG=/claude-io/compact-events.jsonl" in hook_command
    assert (
        "ARC3_COMPACT_CHECKPOINT_REQUEST=/claude-io/checkpoint.requested"
        in hook_command
    )
    assert "ARC3_COMPACT_CHECKPOINT_READY=/claude-io/checkpoint.ready" in hook_command


def test_clean_instruction_envelope_and_concrete_model_id(tmp_path: Path) -> None:
    envelope = {
        "tools": [
            f'mcp__game__{tool["name"]}'
            for tool in build_tools(512, include_compact_checkpoint=False)
        ],
        "mcp_servers": [{"name": "game", "status": "connected"}],
        "model": "claude-opus-4-6",
        "permissionMode": CLAUDE_PERMISSION_MODE,
        "claude_code_version": "2.1.220",
        "slash_commands": [],
        "skills": [],
        "plugins": [],
    }
    validate_instruction_envelope(envelope, display_size=512)
    result = ClaudeResult(
        segment=0,
        returncode=0,
        process_returncode=0,
        final_message="READY",
        session_id="session",
        resolved_models=("claude-opus-4-6",),
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        input_path=tmp_path / "input",
        command_path=tmp_path / "command",
        bridge_log_path=tmp_path / "bridge",
        compact_events_path=tmp_path / "compact",
        usage=None,
        init_envelope=envelope,
    )
    assert resolved_model_id(result) == "claude-opus-4-6"


def test_instruction_envelope_tool_order_is_not_semantic() -> None:
    envelope = {
        "tools": sorted(
            f'mcp__game__{tool["name"]}'
            for tool in build_tools(512, include_compact_checkpoint=False)
        ),
        "mcp_servers": [{"name": "game", "status": "connected"}],
        "model": "claude-opus-4-6",
        "permissionMode": CLAUDE_PERMISSION_MODE,
        "claude_code_version": "2.1.220",
        "slash_commands": [],
        "skills": [],
        "plugins": [],
    }

    validate_instruction_envelope(envelope, display_size=512)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("permissionMode", "dontAsk"),
        ("claude_code_version", "2.1.221"),
        ("skills", ["unexpected"]),
        ("plugins", ["unexpected"]),
        ("slash_commands", ["unexpected"]),
    ],
)
def test_instruction_envelope_rejects_runtime_drift(field: str, value) -> None:
    envelope = {
        "tools": [
            f'mcp__game__{tool["name"]}'
            for tool in build_tools(512, include_compact_checkpoint=False)
        ],
        "mcp_servers": [{"name": "game", "status": "connected"}],
        "model": "claude-opus-4-6",
        "permissionMode": CLAUDE_PERMISSION_MODE,
        "claude_code_version": "2.1.220",
        "slash_commands": [],
        "skills": [],
        "plugins": [],
    }
    envelope[field] = value

    with pytest.raises(RuntimeError):
        validate_instruction_envelope(envelope, display_size=512)


def test_action_cycle_reopens_only_for_the_matching_play_result() -> None:
    pending: set[str] = set()
    acknowledgements = []
    record_pending_play_tool_uses(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "play-1",
                    "name": "mcp__game__play",
                },
                {
                    "type": "tool_use",
                    "id": "inspect-1",
                    "name": "mcp__game__inspect",
                },
            ]
        },
        pending,
    )
    acknowledge_completed_play_tools(
        {
            "content": [
                {"type": "tool_result", "tool_use_id": "inspect-1"}
            ]
        },
        pending,
        lambda: acknowledgements.append(True),
    )
    assert pending == {"play-1"}
    assert acknowledgements == []

    acknowledge_completed_play_tools(
        {
            "content": [
                {"type": "tool_result", "tool_use_id": "play-1"}
            ]
        },
        pending,
        lambda: acknowledgements.append(True),
    )
    assert pending == set()
    assert acknowledgements == [True]
