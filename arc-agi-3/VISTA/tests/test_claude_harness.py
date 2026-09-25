from pathlib import Path
from types import SimpleNamespace

import pytest

from vista_arc3.claude import harness
from vista_arc3.claude.harness import (
    MAX_DECISION_LEASE_RECOVERIES_PER_STEP,
    MAX_STALL_RESUMES_PER_STEP,
    NONTERMINAL_CONTINUATION_PROMPT,
    build_parser,
    run_game,
    run_task_with_native_context,
    stall_timeout_seconds,
)
from vista_arc3.claude.runner import (
    STALL_RESUME_PROMPT,
    ClaudeResult,
    resolve_claude_credential,
)


def test_claude_cli_matches_the_evaluation_choice_surface() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert options == {
        "--game-id",
        "--max-steps",
        "--operation-mode",
        "--model",
        "--effort",
        "--auth",
        "--timeout",
        "--max-invalid-retries",
        "--stall-timeout",
        "--no-guide-working",
        "--evolve",
    }
    defaults = parser.parse_args(["--game-id", "test-game"])
    # --evolve (skills / tools / hooks) is opt-in.
    assert defaults.evolve is False
    assert parser.parse_args(["--game-id", "test-game", "--evolve"]).evolve is True
    assert defaults.model == "opus"
    assert defaults.effort == "xhigh"
    # Billing is chosen at run time (auto -> api-key when ANTHROPIC_API_KEY is set).
    assert defaults.auth is None
    assert parser.parse_args(
        ["--game-id", "test-game", "--auth", "api-key"]
    ).auth == "api-key"
    # GUIDE.md / WORKING.md are on by default; --no-guide-working opts out.
    assert defaults.guide_working is True
    assert (
        parser.parse_args(["--game-id", "test-game", "--no-guide-working"])
        .guide_working
        is False
    )
    # The watchdog is opt-in so the raw harness is unchanged by default.
    assert defaults.stall_timeout == 0
    assert stall_timeout_seconds(defaults) is None
    assert stall_timeout_seconds(
        parser.parse_args(["--game-id", "test-game", "--stall-timeout", "15"])
    ) == 15 * 60
    assert parser.parse_args(
        ["--game-id", "test-game", "--effort", "max"]
    ).effort == "max"
    assert defaults.max_steps == 2_000
    assert defaults.max_invalid_retries == 15
    assert defaults.observation_mode == "vision"
    assert defaults.render_grid is True
    assert defaults.operation_mode == "online"
    assert parser.parse_args(
        ["--game-id", "test-game", "--operation-mode", "offline"]
    ).operation_mode == "offline"


@pytest.mark.parametrize(
    ("mode", "has_arc_key", "has_claude_token", "error"),
    [
        ("offline", False, True, None),
        ("online", True, True, None),
        ("online", False, True, "ARC_API_KEY is not set"),
        ("offline", False, False, "CLAUDE_CODE_OAUTH_TOKEN is not set"),
    ],
)
def test_claude_cli_credential_requirements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    has_arc_key: bool,
    has_claude_token: bool,
    error: str | None,
) -> None:
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ARC3_CLAUDE_AUTH", raising=False)
    for name, present in (
        ("ARC_API_KEY", has_arc_key),
        ("CLAUDE_CODE_OAUTH_TOKEN", has_claude_token),
    ):
        if present:
            monkeypatch.setenv(name, "test-credential")
        else:
            monkeypatch.delenv(name, raising=False)
    calls = []

    def fake_run(args):
        calls.append(args)
        return SimpleNamespace(
            run_dir=tmp_path / "run",
            scorecard_id="test-card",
            session_id="test-session",
            failure=None,
        )

    monkeypatch.setattr(harness, "run_game", fake_run)
    argv = ["--game-id", "test-game", "--operation-mode", mode]
    if error is not None:
        with pytest.raises(SystemExit) as exc:
            harness.main(argv)
        assert exc.value.code == 2
        assert error in capsys.readouterr().err
        assert calls == []
    else:
        assert harness.main(argv) == 0
        assert len(calls) == 1
        assert calls[0].operation_mode == mode


@pytest.mark.parametrize(
    ("argv_auth", "env", "expected_mode", "error"),
    [
        # auto: the API key wins when present, otherwise the subscription token
        (None, {"ANTHROPIC_API_KEY": "sk-ant-api-x", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-x"}, "api-key", None),
        (None, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-x"}, "oauth", None),
        (None, {"ANTHROPIC_API_KEY": "sk-ant-api-x"}, "api-key", None),
        # explicit choices
        ("oauth", {"ANTHROPIC_API_KEY": "sk-ant-api-x", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-x"}, "oauth", None),
        ("api-key", {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-x"}, None, "ANTHROPIC_API_KEY is not set"),
        ("oauth", {"ANTHROPIC_API_KEY": "sk-ant-api-x"}, None, "CLAUDE_CODE_OAUTH_TOKEN is not set"),
        (None, {}, None, "CLAUDE_CODE_OAUTH_TOKEN is not set"),
    ],
)
def test_claude_cli_picks_the_billing_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv_auth: str | None,
    env: dict[str, str],
    expected_mode: str | None,
    error: str | None,
) -> None:
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ARC3_CLAUDE_AUTH"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ARC_API_KEY", "arc")
    calls = []

    def fake_run(args):
        calls.append(args)
        return SimpleNamespace(
            run_dir=tmp_path / "run", scorecard_id="c", session_id="s", failure=None
        )

    monkeypatch.setattr(harness, "run_game", fake_run)
    argv = ["--game-id", "test-game"] + (["--auth", argv_auth] if argv_auth else [])
    if error is not None:
        with pytest.raises(SystemExit):
            harness.main(argv)
        assert error in capsys.readouterr().err
        assert calls == []
        return
    assert harness.main(argv) == 0
    mode, secret = resolve_claude_credential(calls[0].auth)
    assert mode == expected_mode
    assert secret == env["ANTHROPIC_API_KEY" if mode == "api-key" else "CLAUDE_CODE_OAUTH_TOKEN"]


def test_only_a_coordinator_owned_run_may_reuse_a_preflight_model_id() -> None:
    args = build_parser().parse_args(["--game-id", "test-game"])

    with pytest.raises(ValueError, match="coordinator-owned"):
        run_game(args, preflight_model_id="claude-opus-5")


def result(
    tmp_path: Path,
    session: str | None,
    returncode: int,
    *,
    api_error_status: int | None = None,
    context_boundary: str | None = None,
    compact_boundaries: tuple[dict, ...] = (),
    decision_lease_failure: str | None = None,
    unconfirmed_play_calls: int = 0,
) -> ClaudeResult:
    return ClaudeResult(
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
        api_error_status=api_error_status,
        context_boundary=context_boundary,
        compact_boundaries=compact_boundaries,
        decision_lease_failure=decision_lease_failure,
        unconfirmed_play_calls=unconfirmed_play_calls,
    )


class RetryController:
    def __init__(self) -> None:
        self.terminal = False
        self.retry_boundary_pending = False
        self.retry_boundary_step = None
        self.step_index = 1

    def begin_retry_recovery(self):
        return SimpleNamespace(boundary_step=1)


class RetryRunner:
    def __init__(self, tmp_path: Path, controller: RetryController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.resumed = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        self.controller.retry_boundary_pending = True
        self.controller.retry_boundary_step = 1
        return result(self.tmp_path, "old-session", 0)

    def run_retry_task(self, recovery) -> ClaudeResult:
        self.controller.retry_boundary_pending = False
        self.controller.retry_boundary_step = None
        return result(self.tmp_path, "new-session", 1)

    def resume_retry_task(self, session_id: str, recovery) -> ClaudeResult:
        self.resumed += 1
        self.controller.terminal = True
        return result(self.tmp_path, session_id, 0)


def test_retry_handoff_resumes_the_same_fresh_session_after_runtime_failure(
    tmp_path: Path,
) -> None:
    controller = RetryController()
    runner = RetryRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert [item.session_id for item in results] == [
        "old-session",
        "new-session",
        "new-session",
    ]
    assert runner.resumed == 1


class FreshRetryRunner:
    def __init__(self, tmp_path: Path, controller: RetryController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.fresh_attempts = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        self.controller.retry_boundary_pending = True
        self.controller.retry_boundary_step = 1
        return result(self.tmp_path, "failed-attempt", 0)

    def run_retry_task(self, recovery) -> ClaudeResult:
        self.fresh_attempts += 1
        if self.fresh_attempts == 1:
            return result(self.tmp_path, "uninitialized-fresh-session", 1)
        self.controller.retry_boundary_pending = False
        self.controller.retry_boundary_step = None
        self.controller.terminal = True
        return result(self.tmp_path, "successful-fresh-session", 0)


def test_retry_handoff_can_retry_a_failed_fresh_session_start(tmp_path: Path) -> None:
    controller = RetryController()
    runner = FreshRetryRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.fresh_attempts == 2
    assert [item.session_id for item in results] == [
        "failed-attempt",
        "uninitialized-fresh-session",
        "successful-fresh-session",
    ]


class RuntimeRecoveryController:
    def __init__(self) -> None:
        self.terminal = False
        self.retry_boundary_pending = False
        self.step_index = 4
        self.recoveries = 0

    def begin_runtime_recovery(self):
        self.recoveries += 1
        return SimpleNamespace(boundary_step=self.step_index)


class RuntimeRecoveryRunner:
    def __init__(self, tmp_path: Path, controller: RuntimeRecoveryController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.fresh_attempts = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(
            self.tmp_path,
            "failed-session",
            1,
            decision_lease_failure="runtime_exit",
        )

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.fresh_attempts += 1
        if self.fresh_attempts == 1:
            return result(self.tmp_path, None, 1)
        self.controller.terminal = True
        return result(self.tmp_path, "fresh-session", 0)


def test_runtime_failure_retries_only_with_fresh_sessions_after_preinit_failure(
    tmp_path: Path,
) -> None:
    controller = RuntimeRecoveryController()
    runner = RuntimeRecoveryRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.fresh_attempts == 2
    assert controller.recoveries == 2
    assert [item.session_id for item in results] == [
        "failed-session",
        None,
        "fresh-session",
    ]


class CredentialRelayRunner:
    def __init__(self, tmp_path: Path, controller: RuntimeRecoveryController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.oauth_tokens = ["account-one"]
        self.recoveries = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(
            self.tmp_path,
            "old-account-session",
            0,
            context_boundary="credential_rate_limit",
            decision_lease_failure="credential_rate_limit",
        )

    def replace_oauth_token(self, oauth_token: str) -> None:
        self.oauth_tokens.append(oauth_token)

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.recoveries += 1
        assert recovery.boundary_step == 4
        self.controller.terminal = True
        return result(self.tmp_path, "new-account-session", 0)


def test_rate_limit_relays_the_same_game_state_to_a_fresh_account_session(
    tmp_path: Path,
) -> None:
    controller = RuntimeRecoveryController()
    runner = CredentialRelayRunner(tmp_path, controller)
    relays = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=[],
        credential_relay=lambda: relays.append("account-two") or "account-two",
    )

    assert relays == ["account-two"]
    assert runner.oauth_tokens == ["account-one", "account-two"]
    assert runner.recoveries == 1
    assert controller.recoveries == 1
    assert controller.step_index == 4


def test_rate_limit_without_a_credential_pool_fails_before_recovery(
    tmp_path: Path,
) -> None:
    controller = RuntimeRecoveryController()
    runner = CredentialRelayRunner(tmp_path, controller)

    with pytest.raises(RuntimeError, match="without a relay"):
        run_task_with_native_context(
            runner=runner,
            controller=controller,
            prompt="test",
            results=[],
        )

    assert runner.recoveries == 0
    assert controller.recoveries == 0


class ContinuationController:
    def __init__(self) -> None:
        self.terminal = False
        self.retry_boundary_pending = False
        self.step_index = 12
        self.levels_completed = 0
        self.runtime_recoveries = 0

    def initial_metadata(self):
        return {"progress": {"completed": self.levels_completed}}

    def begin_runtime_recovery(self):
        self.runtime_recoveries += 1
        return SimpleNamespace(boundary_step=self.step_index)


class CompactController:
    def __init__(self, tmp_path: Path) -> None:
        self.terminal = False
        self.retry_boundary_pending = False
        self.step_index = 7
        self.compact_checkpoint_marker = tmp_path / "checkpoint.requested"
        self.compact_checkpoint_ready = tmp_path / "checkpoint.ready"
        self.compact_restore_marker = tmp_path / "restore.pending"
        self.recovery = SimpleNamespace(boundary_step=self.step_index, image_paths=())
        self.checkpoint_reasons = []

    def request_compact_checkpoint(self, *, reason: str) -> None:
        self.checkpoint_reasons.append(reason)
        self.compact_checkpoint_marker.touch()

    def begin_compact_recovery(self):
        self.compact_restore_marker.touch()
        return self.recovery

    def discard_compact_recovery(self) -> None:
        pass


class CompactRunner:
    def __init__(self, tmp_path: Path, controller: CompactController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.calls = []

    def run_task(self, prompt: str) -> ClaudeResult:
        self.calls.append("run")
        self.controller.request_compact_checkpoint(reason="provider_image_limit")
        self.controller.compact_checkpoint_ready.touch()
        return result(
            self.tmp_path,
            "same-session",
            0,
            context_boundary="compact_checkpoint_saved",
        )

    def resume_native_compact_trigger(self, session_id: str) -> ClaudeResult:
        raise AssertionError("native compact must not be used")

    def run_compact_recovery(self, recovery) -> ClaudeResult:
        assert recovery is self.controller.recovery
        self.calls.append("fresh-recovery")
        self.controller.compact_checkpoint_marker.unlink()
        self.controller.compact_checkpoint_ready.unlink()
        self.controller.compact_restore_marker.unlink()
        self.controller.terminal = True
        return result(self.tmp_path, "fresh-session", 0)


def test_image_boundary_checkpoints_then_recovers_in_a_fresh_session(
    tmp_path: Path,
) -> None:
    controller = CompactController(tmp_path)
    runner = CompactRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.calls == ["run", "fresh-recovery"]
    assert controller.checkpoint_reasons == ["provider_image_limit"]
    assert controller.step_index == 7
    assert [item.session_id for item in results] == [
        "same-session",
        "fresh-session",
    ]


class CredentialAwareCompactRunner(CompactRunner):
    def __init__(self, tmp_path: Path, controller: CompactController) -> None:
        super().__init__(tmp_path, controller)
        self.oauth_tokens = ["account-one"]

    def replace_oauth_token(self, oauth_token: str) -> None:
        self.oauth_tokens.append(oauth_token)

    def run_compact_recovery(self, recovery) -> ClaudeResult:
        assert self.oauth_tokens == ["account-one", "account-two"]
        return super().run_compact_recovery(recovery)


def test_quota_relay_preserves_an_existing_compact_recovery_boundary(
    tmp_path: Path,
) -> None:
    controller = CompactController(tmp_path)
    runner = CredentialAwareCompactRunner(tmp_path, controller)

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=[],
        credential_relay=lambda: "account-two",
        credential_relay_needed=lambda: len(runner.oauth_tokens) == 1,
    )

    assert runner.calls == ["run", "fresh-recovery"]
    assert runner.oauth_tokens == ["account-one", "account-two"]


class TokenCompactRunner(CompactRunner):
    def run_task(self, prompt: str) -> ClaudeResult:
        self.calls.append("run")
        self.controller.compact_checkpoint_marker.touch()
        self.controller.compact_checkpoint_ready.touch()
        return result(
            self.tmp_path,
            "same-session",
            0,
            context_boundary="compact_checkpoint_saved",
        )


def test_token_boundary_uses_the_same_checkpoint_handoff(tmp_path: Path) -> None:
    controller = CompactController(tmp_path)
    runner = TokenCompactRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.calls == ["run", "fresh-recovery"]
    assert controller.checkpoint_reasons == []
    assert [item.session_id for item in results] == [
        "same-session",
        "fresh-session",
    ]


class IncompleteCompactRunner(CompactRunner):
    def run_task(self, prompt: str) -> ClaudeResult:
        self.calls.append("run")
        self.controller.compact_checkpoint_marker.touch()
        return result(self.tmp_path, "same-session", 0)


def test_checkpoint_never_resumes_an_old_session_without_working(
    tmp_path: Path,
) -> None:
    controller = CompactController(tmp_path)
    runner = IncompleteCompactRunner(tmp_path, controller)

    with pytest.raises(
        RuntimeError,
        match="ended before writing WORKING.md",
    ):
        run_task_with_native_context(
            runner=runner,
            controller=controller,
            prompt="test",
            results=[],
        )

    assert runner.calls == ["run"]


class ContinuationRunner:
    def __init__(self, tmp_path: Path, controller: ContinuationController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.fresh_continuations = 0
        self.same_session_prompts = []

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(self.tmp_path, "continuing-session", 0)

    def resume_nonterminal_task(
        self,
        session_id: str,
        prompt: str,
    ) -> ClaudeResult:
        self.same_session_prompts.append(prompt)
        self.controller.step_index += 1
        self.controller.terminal = True
        return result(self.tmp_path, session_id, 0)

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.fresh_continuations += 1
        raise AssertionError("The first normal completion must resume in-session")


def test_nonterminal_completion_first_resumes_the_same_session(
    tmp_path: Path,
) -> None:
    controller = ContinuationController()
    runner = ContinuationRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.fresh_continuations == 0
    assert runner.same_session_prompts == [NONTERMINAL_CONTINUATION_PROMPT]
    assert results[0].decision_lease_failure == "nonterminal_completion"
    assert [item.session_id for item in results] == [
        "continuing-session",
        "continuing-session",
    ]


class ActionBetweenStopsRunner:
    def __init__(self, tmp_path: Path, controller: ContinuationController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.same_session_continuations = 0
        self.fresh_continuations = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(self.tmp_path, "stopped-session", 0)

    def resume_nonterminal_task(
        self,
        session_id: str,
        prompt: str,
    ) -> ClaudeResult:
        assert prompt == NONTERMINAL_CONTINUATION_PROMPT
        self.same_session_continuations += 1
        self.controller.step_index += 8
        return result(self.tmp_path, session_id, 0)

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.fresh_continuations += 1
        self.controller.terminal = True
        return result(self.tmp_path, "fresh-session", 0)


def test_actions_do_not_renew_same_session_continuation(
    tmp_path: Path,
) -> None:
    controller = ContinuationController()
    runner = ActionBetweenStopsRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.same_session_continuations == 1
    assert runner.fresh_continuations == 1
    assert [item.session_id for item in results] == [
        "stopped-session",
        "stopped-session",
        "fresh-session",
    ]


class LevelChangeBetweenStopsRunner(ActionBetweenStopsRunner):
    def resume_nonterminal_task(
        self,
        session_id: str,
        prompt: str,
    ) -> ClaudeResult:
        assert prompt == NONTERMINAL_CONTINUATION_PROMPT
        self.same_session_continuations += 1
        if self.same_session_continuations == 1:
            self.controller.levels_completed += 1
        else:
            self.controller.terminal = True
        return result(self.tmp_path, session_id, 0)

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        raise AssertionError("A new level must renew the same-session continuation")


def test_level_change_renews_same_session_continuation(tmp_path: Path) -> None:
    controller = ContinuationController()
    runner = LevelChangeBetweenStopsRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.same_session_continuations == 2
    assert [item.session_id for item in results] == [
        "stopped-session",
        "stopped-session",
        "stopped-session",
    ]


class StoppedRunner:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.fresh_continuations = 0
        self.same_session_continuations = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(self.tmp_path, "stopped-session", 0)

    def resume_nonterminal_task(
        self,
        session_id: str,
        prompt: str,
    ) -> ClaudeResult:
        assert prompt == NONTERMINAL_CONTINUATION_PROMPT
        self.same_session_continuations += 1
        return result(self.tmp_path, session_id, 0)

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.fresh_continuations += 1
        return result(
            self.tmp_path,
            f"fresh-player-{self.fresh_continuations}",
            0,
        )


def test_nonterminal_completion_relays_through_bounded_fresh_players(
    tmp_path: Path,
) -> None:
    controller = ContinuationController()
    runner = StoppedRunner(tmp_path)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert (
        runner.fresh_continuations
        == MAX_DECISION_LEASE_RECOVERIES_PER_STEP
    )
    assert (
        controller.runtime_recoveries
        == MAX_DECISION_LEASE_RECOVERIES_PER_STEP
    )
    assert runner.same_session_continuations == (
        MAX_DECISION_LEASE_RECOVERIES_PER_STEP + 1
    )
    assert len(results) == 2 * (
        MAX_DECISION_LEASE_RECOVERIES_PER_STEP + 1
    )
    assert [item.session_id for item in results] == [
        "stopped-session",
        "stopped-session",
        "fresh-player-1",
        "fresh-player-1",
        "fresh-player-2",
        "fresh-player-2",
        "fresh-player-3",
        "fresh-player-3",
        "fresh-player-4",
        "fresh-player-4",
        "fresh-player-5",
        "fresh-player-5",
    ]


class SuccessfulRelayRunner(StoppedRunner):
    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        continued = super().run_fresh_runtime_recovery(recovery)
        if self.fresh_continuations == 3:
            self.controller.step_index += 1
            self.controller.terminal = True
        return continued


def test_a_later_fresh_player_can_continue_the_game(tmp_path: Path) -> None:
    controller = ContinuationController()
    runner = SuccessfulRelayRunner(tmp_path)
    runner.controller = controller
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.fresh_continuations == 3
    assert runner.same_session_continuations == 3
    assert controller.runtime_recoveries == 3
    assert controller.step_index == 13
    assert controller.terminal is True


class LeaseFailureRunner:
    def __init__(self, tmp_path: Path, controller, failure: str) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.failure = failure
        self.fresh = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(
            self.tmp_path,
            "failed-lease",
            1,
            decision_lease_failure=self.failure,
        )

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.fresh += 1
        self.controller.terminal = True
        return result(self.tmp_path, "fresh-lease", 0)


@pytest.mark.parametrize(
    "failure",
    ["output_token_limit", "decision_timeout", "runtime_exit"],
)
def test_each_failed_decision_lease_uses_the_same_fresh_handoff(
    tmp_path: Path,
    failure: str,
) -> None:
    controller = RuntimeRecoveryController()
    runner = LeaseFailureRunner(tmp_path, controller, failure)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.fresh == 1
    assert [item.session_id for item in results] == [
        "failed-lease",
        "fresh-lease",
    ]


class InvalidActionBoundaryRunner:
    def __init__(self, tmp_path: Path, controller: RuntimeRecoveryController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.fresh = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(
            self.tmp_path,
            "invalid-action-session",
            0,
            context_boundary="invalid_action_limit",
            decision_lease_failure="invalid_action_limit",
        )

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.fresh += 1
        self.controller.terminal = True
        return result(self.tmp_path, "fresh-session", 0)


def test_invalid_action_limit_recovers_in_a_fresh_session(tmp_path: Path) -> None:
    controller = RuntimeRecoveryController()
    runner = InvalidActionBoundaryRunner(tmp_path, controller)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.fresh == 1
    assert controller.recoveries == 1
    assert [item.session_id for item in results] == [
        "invalid-action-session",
        "fresh-session",
    ]
    assert results[0].decision_lease_failure == "invalid_action_limit"


class StallController:
    def __init__(self) -> None:
        self.terminal = False
        self.retry_boundary_pending = False
        self.step_index = 30
        self.recoveries = 0

    def initial_metadata(self):
        return {"progress": {"completed": 2}}

    def begin_runtime_recovery(self):
        self.recoveries += 1
        return SimpleNamespace(boundary_step=self.step_index, image_paths=())


class StallRunner:
    """Stalls `stalls` times in a row at the same game state, then finishes."""

    def __init__(self, tmp_path: Path, controller: StallController, stalls: int) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.remaining_stalls = stalls
        self.calls: list[tuple[str, str | None]] = []

    def _stalled(self, session: str) -> ClaudeResult:
        self.remaining_stalls -= 1
        return result(
            self.tmp_path,
            session,
            1,
            decision_lease_failure="stream_stall",
        )

    def run_task(self, prompt: str) -> ClaudeResult:
        self.calls.append(("run", None))
        return self._stalled("stalled-session")

    def resume_nonterminal_task(self, session_id: str, prompt: str) -> ClaudeResult:
        assert prompt == STALL_RESUME_PROMPT
        self.calls.append(("resume", session_id))
        if self.remaining_stalls > 0:
            return self._stalled(session_id)
        self.controller.terminal = True
        return result(self.tmp_path, session_id, 0)

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.calls.append(("fresh", None))
        self.controller.terminal = True
        return result(self.tmp_path, "fresh-session", 0)


def test_stream_stall_resumes_the_same_session(tmp_path: Path) -> None:
    controller = StallController()
    runner = StallRunner(tmp_path, controller, stalls=2)
    results: list[ClaudeResult] = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
    )

    assert runner.calls == [
        ("run", None),
        ("resume", "stalled-session"),
        ("resume", "stalled-session"),
    ]
    assert controller.recoveries == 0
    assert [item.session_id for item in results] == ["stalled-session"] * 3


def test_repeated_stalls_at_one_state_fall_back_to_fresh_recovery(
    tmp_path: Path,
) -> None:
    controller = StallController()
    runner = StallRunner(
        tmp_path,
        controller,
        stalls=MAX_STALL_RESUMES_PER_STEP + 5,
    )

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=[],
    )

    resumes = [call for call in runner.calls if call[0] == "resume"]
    assert len(resumes) == MAX_STALL_RESUMES_PER_STEP
    assert runner.calls[-1] == ("fresh", None)
    assert controller.recoveries == 1


def _wire_fake_claude_run(monkeypatch, tmp_path: Path, *, terminal_state) -> dict:
    """Stub everything outside the harness so run_game exercises its own wiring."""

    import numpy as np
    from arcengine import GameState

    calls: dict = {
        "runners": [],
        "controllers": [],
        "loop_kwargs": [],
        "loop_calls": 0,
    }

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.visible_dir = kwargs["visible_dir"]
            calls["runners"].append(self)

    def fake_loop(*, runner, controller, prompt, results, **kwargs):
        calls["loop_calls"] += 1
        calls["controllers"].append(controller)
        calls["loop_kwargs"].append(kwargs)
        controller.step_index = calls.get("steps", 0)
        results.append(result(tmp_path, "session", 0))
        return results

    initial = SimpleNamespace(
        game_id="g",
        guid="guid",
        state=terminal_state or GameState.WIN,
        levels_completed=8,
        win_levels=8,
        available_actions=[1, 6],
        action_input=None,
        full_reset=False,
        frame=[np.zeros((64, 64), dtype=np.uint8)],
    )
    env = SimpleNamespace(observation_space=initial, action_space=[])
    arc = SimpleNamespace(get_environments=lambda: [SimpleNamespace(game_id="g")])
    monkeypatch.setattr(harness, "validate_claude_runtime", lambda **kwargs: "image")
    monkeypatch.setattr(
        harness,
        "run_runtime_preflight",
        lambda **kwargs: (None, "claude-opus-5"),
    )
    monkeypatch.setattr(harness, "Arcade", lambda **kwargs: arc)
    monkeypatch.setattr(harness, "configure_arc_http", lambda *args, **kwargs: None)
    monkeypatch.setattr(harness, "open_evaluation_scorecard", lambda arc: "card")
    monkeypatch.setattr(harness, "share_online_cookie_jar", lambda *args: None)
    monkeypatch.setattr(
        harness,
        "finalize_scorecard",
        lambda **kwargs: ({"scorecard_closed": True, "scorecard_summary": {"score": 1}}, None),
    )
    monkeypatch.setattr(harness, "ClaudeCodeRunner", FakeRunner)
    monkeypatch.setattr(harness, "run_task_with_native_context", fake_loop)
    monkeypatch.setattr(harness, "_safe_claude_version", lambda: "test")
    calls["environment_factory"] = lambda game_id, scorecard_id, recordings_dir: env
    return calls


def test_run_game_wires_the_stall_watchdog_only_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    import json as json_module

    calls = _wire_fake_claude_run(monkeypatch, tmp_path, terminal_state=None)

    args = build_parser().parse_args(["--game-id", "g", "--stall-timeout", "15"])
    outcome = run_game(
        args,
        run_dir=tmp_path / "watched_run",
        environment_factory=calls["environment_factory"],
    )
    assert outcome.failure is None
    manifest = json_module.loads(
        (tmp_path / "watched_run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert calls["runners"][-1].kwargs["stream_stall_timeout"] == 15 * 60
    assert manifest["stall_timeout_minutes"] == 15

    plain = build_parser().parse_args(["--game-id", "g"])
    outcome = run_game(
        plain,
        run_dir=tmp_path / "plain_run",
        environment_factory=calls["environment_factory"],
    )
    assert outcome.failure is None
    manifest = json_module.loads(
        (tmp_path / "plain_run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert calls["runners"][-1].kwargs["stream_stall_timeout"] is None
    assert manifest["stall_timeout_minutes"] == 0


class RolloverRunner:
    """--no-guide-working: a context boundary restarts from the game state."""

    def __init__(
        self,
        tmp_path: Path,
        controller: RuntimeRecoveryController,
        *,
        boundary: str,
        rollovers_before_terminal: int = 1,
    ) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.boundary = boundary
        self.rollovers_before_terminal = rollovers_before_terminal
        self.calls = []

    def run_task(self, prompt: str) -> ClaudeResult:
        self.calls.append("run")
        return result(
            self.tmp_path,
            "first-session",
            0,
            context_boundary=self.boundary,
        )

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.calls.append("fresh-runtime-recovery")
        fresh = len(self.calls) - 1
        if fresh < self.rollovers_before_terminal:
            return result(
                self.tmp_path,
                f"rollover-session-{fresh}",
                0,
                context_boundary=self.boundary,
            )
        self.controller.terminal = True
        return result(self.tmp_path, f"rollover-session-{fresh}", 0)


@pytest.mark.parametrize("boundary", ["context_rollover", "native_compact"])
def test_without_notes_a_context_boundary_restarts_from_the_game_state(
    tmp_path: Path, boundary: str
) -> None:
    controller = RuntimeRecoveryController()
    runner = RolloverRunner(tmp_path, controller, boundary=boundary)
    results = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
        notes_enabled=False,
    )

    assert runner.calls == ["run", "fresh-runtime-recovery"]
    assert controller.recoveries == 1
    assert [item.session_id for item in results] == [
        "first-session",
        "rollover-session-1",
    ]


def test_without_notes_repeated_rollovers_without_an_action_fail_closed(
    tmp_path: Path,
) -> None:
    controller = RuntimeRecoveryController()
    runner = RolloverRunner(
        tmp_path,
        controller,
        boundary="context_rollover",
        rollovers_before_terminal=10,
    )

    with pytest.raises(RuntimeError, match="rollover repeated without an action"):
        run_task_with_native_context(
            runner=runner,
            controller=controller,
            prompt="test",
            results=[],
            notes_enabled=False,
        )

    assert controller.recoveries == harness.MAX_COMPACT_CYCLES_WITHOUT_ACTION


def test_with_notes_a_context_rollover_boundary_is_a_harness_error(
    tmp_path: Path,
) -> None:
    controller = RuntimeRecoveryController()
    runner = RolloverRunner(tmp_path, controller, boundary="context_rollover")

    with pytest.raises(RuntimeError, match="context rollover boundary with notes"):
        run_task_with_native_context(
            runner=runner,
            controller=controller,
            prompt="test",
            results=[],
        )

    assert runner.calls == ["run"]


def test_run_game_without_guide_working_has_no_notes_anywhere(
    monkeypatch, tmp_path: Path
) -> None:
    import json as json_module

    calls = _wire_fake_claude_run(monkeypatch, tmp_path, terminal_state=None)

    args = build_parser().parse_args(["--game-id", "g", "--no-guide-working"])
    outcome = run_game(
        args,
        run_dir=tmp_path / "no_notes_run",
        environment_factory=calls["environment_factory"],
    )
    assert outcome.failure is None

    player_dir = tmp_path / "no_notes_run" / "player"
    assert not (player_dir / "GUIDE.md").exists()
    assert not (player_dir / "WORKING.md").exists()
    agents = (player_dir / "AGENTS.md").read_text(encoding="utf-8")
    assert "GUIDE.md" not in agents
    assert "WORKING.md" not in agents
    assert agents.endswith("expected or not.\n")

    runner = calls["runners"][-1]
    assert runner.kwargs["notes_enabled"] is False
    controller = calls["controllers"][-1]
    assert controller.guide_path is None
    assert controller.working_path is None
    assert controller.compact_checkpoint_marker is None
    assert controller.compact_checkpoint_ready is None
    assert controller.compact_restore_marker is None
    assert calls["loop_kwargs"][-1]["notes_enabled"] is False
    manifest = json_module.loads(
        (tmp_path / "no_notes_run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["guide_working_enabled"] is False

    plain = build_parser().parse_args(["--game-id", "g"])
    outcome = run_game(
        plain,
        run_dir=tmp_path / "plain_run",
        environment_factory=calls["environment_factory"],
    )
    assert outcome.failure is None
    plain_player = tmp_path / "plain_run" / "player"
    assert (plain_player / "GUIDE.md").read_text(encoding="utf-8").strip()
    assert "GUIDE.md" in (plain_player / "AGENTS.md").read_text(encoding="utf-8")
    assert calls["runners"][-1].kwargs["notes_enabled"] is True
    assert calls["controllers"][-1].guide_path == plain_player / "GUIDE.md"
    assert calls["controllers"][-1].compact_checkpoint_marker is not None
    assert calls["loop_kwargs"][-1]["notes_enabled"] is True
    manifest = json_module.loads(
        (tmp_path / "plain_run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["guide_working_enabled"] is True


class ProviderRecoveryRunner:
    def __init__(self, tmp_path: Path, controller: RuntimeRecoveryController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.fresh = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(
            self.tmp_path,
            "failed-provider-session",
            1,
            api_error_status=500,
        )

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.fresh += 1
        if self.fresh < 3:
            return result(
                self.tmp_path,
                f"fresh-provider-session-{self.fresh}",
                1,
                api_error_status=500,
            )
        self.controller.terminal = True
        return result(self.tmp_path, "fresh-provider-session-3", 0)


def test_repeated_provider_5xx_backs_off_and_always_uses_fresh_sessions(
    tmp_path: Path,
) -> None:
    controller = RuntimeRecoveryController()
    runner = ProviderRecoveryRunner(tmp_path, controller)
    results = []
    delays = []

    run_task_with_native_context(
        runner=runner,
        controller=controller,
        prompt="test",
        results=results,
        sleep=delays.append,
    )

    assert runner.fresh == 3
    assert delays == [10, 30, 60]
    assert [item.session_id for item in results] == [
        "failed-provider-session",
        "fresh-provider-session-1",
        "fresh-provider-session-2",
        "fresh-provider-session-3",
    ]


class UnconfirmedPlayRunner:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.fresh = 0

    def run_task(self, prompt: str) -> ClaudeResult:
        return result(
            self.tmp_path,
            "uncertain-session",
            1,
            decision_lease_failure="runtime_exit",
            unconfirmed_play_calls=1,
        )

    def run_fresh_runtime_recovery(self, recovery) -> ClaudeResult:
        self.fresh += 1
        raise AssertionError("An uncertain play must never be relayed")


def test_unconfirmed_play_fails_closed_without_starting_a_fresh_session(
    tmp_path: Path,
) -> None:
    controller = RuntimeRecoveryController()
    runner = UnconfirmedPlayRunner(tmp_path)

    with pytest.raises(RuntimeError, match="unconfirmed play call"):
        run_task_with_native_context(
            runner=runner,
            controller=controller,
            prompt="test",
            results=[],
        )

    assert runner.fresh == 0
    assert controller.recoveries == 0
