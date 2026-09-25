"""Run one visual-game trajectory through Claude Code and native MCP."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..shared.http import (
    ARC_CONNECT_RETRIES,
    ARC_CONNECT_TIMEOUT_SECONDS,
    ARC_READ_TIMEOUT_SECONDS,
)
from .mcp_host import CONTEXT_ROLLOVER_BOUNDARY, ClaudeGameMcpHost
from ..shared.evolve import EvolveStore
from .quota import ClaudeRateLimitEvent, rate_limit_event_from_message
from .controller import CompactRecovery, RetryRecovery, RuntimeRecovery
from .tools import build_tools
from .recovery import (
    compact_recovery_prompt,
    fresh_runtime_recovery_prompt,
    retry_recovery_prompt,
)
from .dispatcher import GameToolDispatcher


PLAYER_IMAGE = "arc3-claude-player:0.1"
CONTAINER_CLAUDE_BIN = "/opt/claude/claude"
CONTAINER_MCP_BRIDGE = "/opt/arc3/claude_mcp_bridge.py"
CONTAINER_COMPACT_HOOK = "/opt/arc3/claude_compact_hook.py"
PINNED_CLAUDE_VERSION = "2.1.220"
DEFAULT_DOCKER_HOST_ALIAS = "host.docker.internal"
DEFAULT_CLAUDE_MODEL = "opus"
DEFAULT_CLAUDE_EFFORT = "xhigh"
CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
CLAUDE_PERMISSION_MODE = "default"
MAX_MCP_OUTPUT_TOKENS = 100_000
CLAUDE_PROVIDER_IMAGE_LIMIT = 600
CLAUDE_PROVIDER_IMAGE_COMPACT_AT = CLAUDE_PROVIDER_IMAGE_LIMIT - 1
CLAUDE_AUTO_COMPACT_WINDOW = 1_000_000
CLAUDE_AUTO_COMPACT_PERCENT = 90
DEFAULT_DECISION_LEASE_TIMEOUT_SECONDS = 120 * 60
# Opt-in stall watchdog (off by default so the raw harness is unchanged).
# Normal Opus thinking produces a visible stream event at least every few
# minutes (the longest genuine think observed across runs was ~10 minutes);
# multi-hour gaps with tiny thinking signatures were dead API streams.
SUGGESTED_STREAM_STALL_TIMEOUT_SECONDS = 15 * 60
PLAY_SETTLEMENT_TIMEOUT_SECONDS = (
    ARC_READ_TIMEOUT_SECONDS
    + ARC_CONNECT_TIMEOUT_SECONDS * (ARC_CONNECT_RETRIES + 1)
    + 5
)
CREDENTIAL_ENV_NAMES = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ARC_API_KEY",
)
# How the Claude Code container authenticates and therefore who pays:
#   oauth   -> CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token` (subscription quota)
#   api-key -> ANTHROPIC_API_KEY (Anthropic platform billing)
CLAUDE_AUTH_MODES = ("api-key", "oauth", "auto")
CLAUDE_CREDENTIAL_ENV = {
    "oauth": "CLAUDE_CODE_OAUTH_TOKEN",
    "api-key": "ANTHROPIC_API_KEY",
}
RUNTIME_PREFLIGHT_PROMPT = (
    "This is a runtime preflight, not a game. Do not call tools. Reply exactly READY."
)
STALL_RESUME_PROMPT = (
    "Your previous turn was interrupted because the model stream produced no "
    "output for a long time (a transport stall, not a game event). The game state "
    "is unchanged and the previous tool results still apply. Continue."
)


@dataclass(frozen=True)
class ClaudeUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float | None
    num_turns: int | None


@dataclass(frozen=True)
class ClaudeResult:
    segment: int
    returncode: int
    process_returncode: int
    final_message: str
    session_id: str | None
    resolved_models: tuple[str, ...]
    stdout_path: Path
    stderr_path: Path
    input_path: Path
    command_path: Path
    bridge_log_path: Path
    compact_events_path: Path
    usage: ClaudeUsage | None
    api_error_status: int | None = None
    session_image_blocks: int = 0
    peak_session_image_blocks: int = 0
    reset_boundary: bool = False
    context_boundary: str | None = None
    init_envelope: dict[str, Any] | None = None
    compact_boundaries: tuple[dict[str, Any], ...] = ()
    decision_lease_failure: str | None = None
    unconfirmed_play_calls: int = 0
    rate_limit_events: tuple[ClaudeRateLimitEvent, ...] = ()


@dataclass
class DecisionLeaseWatchdog:
    timeout_seconds: float
    started_at: float
    completed_tool_calls: int = 0

    def expired(
        self,
        *,
        now: float,
        completed_tool_calls: int,
        tool_calls_in_flight: int,
        last_tool_completed_at: float | None,
    ) -> bool:
        if completed_tool_calls != self.completed_tool_calls:
            self.completed_tool_calls = completed_tool_calls
            self.started_at = last_tool_completed_at or now
        if tool_calls_in_flight:
            return False
        return now - self.started_at >= self.timeout_seconds


def stream_stalled(*, now: float, last_output_at: float, timeout_seconds: float) -> bool:
    """True when Claude Code has emitted nothing on stdout for the timeout."""
    return now - last_output_at >= timeout_seconds


class ClaudePreflightController:
    """A non-game controller used to initialize the exact formal runtime."""

    compact_restore_marker = None
    compact_checkpoint_marker = None
    retry_boundary_pending = False
    reset_starts_fresh_session = False

    @staticmethod
    def initial_image_paths() -> tuple[Path, ...]:
        return ()

    @staticmethod
    def handle(request: object) -> dict[str, Any]:
        return {
            "ok": False,
            "metadata": {"error": "Game tools are unavailable during runtime preflight."},
        }


def default_claude_bin() -> Path:
    configured = os.getenv("ARC3_CLAUDE_BIN")
    if not configured:
        raise FileNotFoundError(
            "Host Claude binaries are no longer required. "
            "Build the Docker image instead."
        )
    candidate = Path(configured).expanduser()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"Configured Claude binary was not found at {candidate}.")
    return candidate.resolve()


def default_claude_image() -> str:
    return os.getenv("ARC3_CLAUDE_IMAGE", PLAYER_IMAGE)


def default_docker_host_alias() -> str:
    return os.getenv("ARC3_DOCKER_HOST_ALIAS", DEFAULT_DOCKER_HOST_ALIAS)


def credential_free_environment(
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    child = dict(os.environ if environ is None else environ)
    for name in tuple(child):
        if name in CREDENTIAL_ENV_NAMES or name.startswith(
            "CLAUDE_CODE_OAUTH_TOKEN_"
        ):
            child.pop(name)
    return child


def claude_version(image: str | Path | None = None) -> str:
    resolved_image = (
        default_claude_image() if image is None or isinstance(image, Path) else image
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--entrypoint",
            CONTAINER_CLAUDE_BIN,
            resolved_image,
            "--version",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=credential_free_environment(),
    )
    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"Claude runtime image {resolved_image!r} failed version validation: "
            f"{output or 'unknown error'}"
        )
    return output.split()[0] if output else ""


def validate_claude_runtime(
    image: str | Path | None = None,
    *,
    oauth_token: str | None = None,
) -> str:
    resolved_image = (
        default_claude_image() if image is None or isinstance(image, Path) else image
    )
    image_check = subprocess.run(
        ["docker", "image", "inspect", resolved_image],
        text=True,
        capture_output=True,
        check=False,
    )
    if image_check.returncode != 0:
        raise FileNotFoundError(
            f"Claude runtime image {resolved_image!r} was not found. "
            "Build it with: docker build -t arc3-claude-player:0.1 "
            "-f Dockerfile.claude-player ."
        )
    version = claude_version(resolved_image)
    if version != PINNED_CLAUDE_VERSION:
        raise RuntimeError(
            f"Claude Code {PINNED_CLAUDE_VERSION} is required; found {version or 'unknown'}."
        )
    if oauth_token is None:
        resolve_claude_credential()  # raises when no credential is configured
    elif not oauth_token:
        raise RuntimeError(MISSING_CLAUDE_CREDENTIAL)
    return resolved_image


MISSING_CLAUDE_CREDENTIAL = (
    "No Claude credential is set. Put ANTHROPIC_API_KEY in .env for API billing, "
    "or run `claude setup-token` and set CLAUDE_CODE_OAUTH_TOKEN for the "
    "subscription."
)


def default_claude_auth_mode() -> str:
    return os.getenv("ARC3_CLAUDE_AUTH", "auto")


def claude_credential_env_name(auth_mode: str) -> str:
    try:
        return CLAUDE_CREDENTIAL_ENV[auth_mode]
    except KeyError:
        raise ValueError(
            f"Claude auth mode must be one of {', '.join(CLAUDE_CREDENTIAL_ENV)}; "
            f"got {auth_mode!r}"
        ) from None


def resolve_claude_credential(
    mode: str | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Pick the credential the Claude Code container should run with.

    Returns ``(auth_mode, secret)``. ``auto`` (default) bills the Anthropic API
    key when ANTHROPIC_API_KEY is set and otherwise falls back to the
    subscription token in CLAUDE_CODE_OAUTH_TOKEN.
    """
    source = os.environ if environ is None else environ
    mode = (mode or default_claude_auth_mode()).strip().lower()
    if mode not in CLAUDE_AUTH_MODES:
        raise ValueError(
            f"Claude auth mode must be one of {', '.join(CLAUDE_AUTH_MODES)}; "
            f"got {mode!r}"
        )
    api_key = source.get("ANTHROPIC_API_KEY", "").strip()
    oauth_token = source.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if mode == "auto":
        mode = "api-key" if api_key else "oauth"
    if mode == "api-key":
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; put it in .env or export it, or "
                "run with --auth oauth to use the subscription token instead"
            )
        return "api-key", api_key
    if not oauth_token:
        raise RuntimeError(
            "CLAUDE_CODE_OAUTH_TOKEN is not set; run `claude setup-token` and "
            "export the generated subscription token, or run with --auth api-key "
            "to bill ANTHROPIC_API_KEY instead"
        )
    return "oauth", oauth_token


def claude_process_environment(
    credential: str,
    environ: dict[str, str] | None = None,
    *,
    auth_mode: str = "oauth",
) -> dict[str, str]:
    """Build a child environment containing only the assigned Claude credential.

    The credential is exported under the variable Claude Code expects for its
    kind: CLAUDE_CODE_OAUTH_TOKEN for subscription tokens (``oauth``),
    ANTHROPIC_API_KEY for Anthropic API keys (``api-key``).
    """

    if not credential:
        raise ValueError("Claude credential must not be empty")
    child = credential_free_environment(environ)
    child[claude_credential_env_name(auth_mode)] = credential
    return child


def prepare_claude_config(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o700)
    return target.resolve()


def create_short_socket_dir(segment: int) -> Path:
    """Create a private host path that stays below the AF_UNIX path limit."""

    directory = Path(tempfile.mkdtemp(prefix=f"arc3-claude-{segment:03d}-"))
    directory.chmod(0o700)
    socket_path = directory / "controller.sock"
    if len(os.fsencode(socket_path)) >= 100:
        shutil.rmtree(directory, ignore_errors=True)
        raise RuntimeError("Unable to create a sufficiently short game socket path.")
    return directory


class ClaudeCodeRunner:
    def __init__(
        self,
        *,
        visible_dir: Path,
        claude_config_dir: Path,
        io_dir: Path,
        controller: Any,
        claude_bin: Path | None = None,
        ca_bundle: Path | None = None,
        model: str = DEFAULT_CLAUDE_MODEL,
        effort: str = DEFAULT_CLAUDE_EFFORT,
        expected_model_id: str | None = None,
        oauth_token: str | None = None,
        auth_mode: str | None = None,
        task_timeout: int = 3600,
        image: str = PLAYER_IMAGE,
        display_size: int = 512,
        auto_compact_window: int | None = CLAUDE_AUTO_COMPACT_WINDOW,
        auto_compact_percent: int | None = CLAUDE_AUTO_COMPACT_PERCENT,
        decision_lease_timeout: int = DEFAULT_DECISION_LEASE_TIMEOUT_SECONDS,
        stream_stall_timeout: int | None = None,
        rate_limit_observer: Callable[[ClaudeRateLimitEvent], None] | None = None,
        credential_yield_requested: Callable[[], bool] | None = None,
        notes_enabled: bool = True,
        evolve: EvolveStore | None = None,
    ) -> None:
        self.visible_dir = visible_dir.resolve()
        self.guide_path = self.visible_dir / "GUIDE.md"
        self.working_path = self.visible_dir / "WORKING.md"
        self.agents_path = self.visible_dir / "AGENTS.md"
        self.agents_instructions = self.agents_path.read_text(encoding="utf-8")
        self.claude_config_dir = prepare_claude_config(claude_config_dir)
        self.io_dir = io_dir.resolve()
        self.controller = controller
        # --no-guide-working disables the GUIDE.md / WORKING.md tools entirely.
        self.notes_enabled = notes_enabled
        # --evolve adds the skills / analysis tools / hooks tools.
        self.evolve = evolve
        self.dispatcher = GameToolDispatcher(
            controller=controller,
            guide_path=self.guide_path,
            working_path=self.working_path,
            checkpoint_via_working=True,
            interrupt_on_invalid_action_limit=True,
            notes_enabled=notes_enabled,
            evolve=evolve,
        )
        self.claude_bin = claude_bin.resolve() if claude_bin is not None else None
        self.ca_bundle = ca_bundle.resolve() if ca_bundle is not None else None
        if effort not in CLAUDE_EFFORT_LEVELS:
            raise ValueError(
                f"Unsupported Claude effort {effort!r}; expected one of "
                f"{', '.join(CLAUDE_EFFORT_LEVELS)}."
            )
        self.model = model
        self.effort = effort
        self.expected_model_id = expected_model_id
        # `oauth_token` is the Claude credential the container runs with: a
        # `claude setup-token` subscription token (auth_mode "oauth", the
        # default and what the batch credential pool hands out) or an Anthropic
        # API key (auth_mode "api-key"). With no explicit credential the
        # environment decides (see resolve_claude_credential).
        if oauth_token is None:
            resolved_mode, oauth_token = resolve_claude_credential(auth_mode)
            auth_mode = resolved_mode
        elif not oauth_token:
            raise RuntimeError(MISSING_CLAUDE_CREDENTIAL)
        self.oauth_token = oauth_token
        self.auth_mode = auth_mode or "oauth"
        claude_credential_env_name(self.auth_mode)  # validates the mode
        self.task_timeout = task_timeout
        if decision_lease_timeout < 1:
            raise ValueError("The decision lease timeout must be positive.")
        self.decision_lease_timeout = decision_lease_timeout
        if stream_stall_timeout is not None and stream_stall_timeout < 1:
            raise ValueError("The stream stall timeout must be positive or None.")
        self.stream_stall_timeout = stream_stall_timeout
        self.image = image
        self.display_size = display_size
        if (auto_compact_window is None) != (auto_compact_percent is None):
            raise ValueError("Both auto-compact settings must be provided together.")
        if auto_compact_window is not None and auto_compact_window < 1:
            raise ValueError("The auto-compact window must be positive.")
        if auto_compact_percent is not None and not 1 <= auto_compact_percent <= 100:
            raise ValueError("The auto-compact percentage must be between 1 and 100.")
        self.auto_compact_window = auto_compact_window
        self.auto_compact_percent = auto_compact_percent
        self.rate_limit_observer = rate_limit_observer
        self.credential_yield_requested = credential_yield_requested
        self._segment_index = 0
        self._session_image_blocks: dict[str, int] = {}
        self.io_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.agents_path,
            self.visible_dir / "screenshots",
            Path(__file__).with_name("mcp_bridge.py"),
            Path(__file__).with_name("compact_hook.py"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Incomplete Claude player runtime: {path}")

    def run_task(self, prompt: str) -> ClaudeResult:
        return self._run_segment(
            prompt=prompt,
            initial_images=self.controller.initial_image_paths(),
        )

    def replace_oauth_token(self, oauth_token: str) -> None:
        if not oauth_token:
            raise ValueError("Claude credential must be non-empty")
        self.oauth_token = oauth_token

    def run_retry_task(self, recovery: RetryRecovery) -> ClaudeResult:
        return self._run_segment(
            prompt=retry_recovery_prompt(recovery),
            initial_images=recovery.image_paths,
            before_input=lambda: self.controller.complete_retry_recovery(recovery),
        )

    def resume_retry_task(
        self,
        session_id: str,
        recovery: RetryRecovery,
    ) -> ClaudeResult:
        return self._run_segment(
            prompt=retry_recovery_prompt(recovery),
            initial_images=recovery.image_paths,
            resume_session_id=session_id,
        )

    def resume_nonterminal_task(
        self,
        session_id: str,
        prompt: str,
    ) -> ClaudeResult:
        return self._run_segment(
            prompt=prompt,
            initial_images=(),
            resume_session_id=session_id,
        )

    def run_fresh_runtime_recovery(
        self,
        recovery: RuntimeRecovery,
    ) -> ClaudeResult:
        return self._run_segment(
            prompt=fresh_runtime_recovery_prompt(recovery),
            initial_images=recovery.image_paths,
            before_input=lambda: self.controller.complete_runtime_recovery(
                recovery,
                delivery="fresh_runtime_thread",
            ),
        )

    def run_compact_recovery(
        self,
        recovery: CompactRecovery,
        *,
        prompt: str | None = None,
    ) -> ClaudeResult:
        recovery_prompt = compact_recovery_prompt(recovery)
        if prompt is not None:
            recovery_prompt = "\n\n".join((recovery_prompt, prompt))
        return self._run_segment(
            prompt=recovery_prompt,
            initial_images=recovery.image_paths,
            before_input=lambda: self.controller.complete_compact_recovery(
                recovery,
                delivery="fresh_thread",
            ),
        )

    def _next_segment(self) -> int:
        segment = self._segment_index
        self._segment_index += 1
        return segment

    def _run_segment(
        self,
        *,
        prompt: str,
        initial_images: Any,
        resume_session_id: str | None = None,
        before_input: Callable[[], None] | None = None,
        disable_slash_commands: bool = True,
    ) -> ClaudeResult:
        validate_claude_runtime(self.image, oauth_token=self.oauth_token)
        initial_images = tuple(initial_images)
        segment = self._next_segment()
        stem = f"segment_{segment:03d}"
        requested_session_id = resume_session_id or str(uuid.uuid4())
        initial_image_blocks = (
            self._session_image_blocks.get(requested_session_id, 0)
            + len(initial_images)
        )
        if initial_image_blocks >= CLAUDE_PROVIDER_IMAGE_LIMIT:
            raise RuntimeError(
                "Claude session reached the provider image safety limit before input."
            )
        stdout_path = self.io_dir / f"{stem}.stdout.jsonl"
        stderr_path = self.io_dir / f"{stem}.stderr.txt"
        input_path = self.io_dir / f"{stem}.input.json"
        command_path = self.io_dir / f"{stem}.command.json"
        bridge_log_path = self.io_dir / f"{stem}.bridge.jsonl"
        compact_events_path = self.io_dir / f"{stem}.compact.jsonl"
        bridge_token = secrets.token_urlsafe(32)
        mcp_config_path = self.claude_config_dir / f"{stem}.mcp.json"
        settings_path = self.claude_config_dir / f"{stem}.settings.json"
        self._write_settings(settings_path, compact_events_path.name)

        host = ClaudeGameMcpHost(
            bind_host="0.0.0.0",
            port=0,
            token=bridge_token,
            dispatcher=self.dispatcher,
            display_size=self.display_size,
            log_path=bridge_log_path,
            call_namespace=stem,
            initial_image_blocks=initial_image_blocks,
            image_block_limit=CLAUDE_PROVIDER_IMAGE_LIMIT,
            image_compact_at=CLAUDE_PROVIDER_IMAGE_COMPACT_AT,
            include_compact_checkpoint=False,
            reset_requires_retry_state=(
                self.controller.reset_starts_fresh_session
            ),
            notes_enabled=self.notes_enabled,
            evolve_enabled=self.evolve is not None,
        )
        process: subprocess.Popen[str] | None = None
        stream_queue: queue.Queue[tuple[str, str] | object] = queue.Queue()
        closed = object()
        final_message = ""
        resolved_models: set[str] = set()
        result_message: dict[str, Any] | None = None
        init_envelope: dict[str, Any] | None = None
        failure: str | None = None
        reset_boundary = False
        context_boundary: str | None = None
        compact_boundaries: list[dict[str, Any]] = []
        pending_play_tool_ids: set[str] = set()
        api_error_status: int | None = None
        decision_lease_failure: str | None = None
        unconfirmed_play_calls = 0
        rate_limit_events: list[ClaudeRateLimitEvent] = []

        try:
            host.start()
            if host.port is None:
                raise RuntimeError("Claude game MCP did not publish a TCP port.")
            self._write_mcp_config(
                mcp_config_path,
                bridge_token,
                bridge_host=default_docker_host_alias(),
                bridge_port=host.port,
            )
            command = self._docker_command(
                mcp_config_path=mcp_config_path,
                settings_path=settings_path,
                bridge_host=default_docker_host_alias(),
                bridge_port=host.port,
                session_id=requested_session_id,
                resume=resume_session_id is not None,
                disable_slash_commands=disable_slash_commands,
                include_partial_messages=self.stream_stall_timeout is not None,
            )
            command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")
            input_message = stream_user_message(prompt, initial_images)
            input_path.write_text(
                json.dumps(sanitize_stream_message(input_message), indent=2),
                encoding="utf-8",
            )
            process = subprocess.Popen(
                command,
                cwd=self.visible_dir,
                env=claude_process_environment(
                    self.oauth_token, auth_mode=self.auth_mode
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_thread = _start_stream_reader(
                process.stdout, "stdout", stream_queue, stdout_path, closed
            )
            stderr_thread = _start_stream_reader(
                process.stderr, "stderr", stream_queue, stderr_path, closed
            )
            if not host.connected.wait(timeout=30):
                if host.failure is not None:
                    raise RuntimeError(
                        f"Claude game MCP failed: {type(host.failure).__name__}: "
                        f"{host.failure}"
                    )
                raise TimeoutError("Claude game MCP did not connect before input.")
            if not host.tools_ready.wait(timeout=30):
                if host.failure is not None:
                    raise RuntimeError(
                        f"Claude game MCP failed: {type(host.failure).__name__}: "
                        f"{host.failure}"
                    )
                raise TimeoutError("Claude game tools were not delivered before input.")
            process.stdin.write(json.dumps(input_message, separators=(",", ":")) + "\n")
            process.stdin.flush()
            closed_streams = 0
            init_deadline = time.monotonic() + 30
            while init_envelope is None:
                if host.failure is not None:
                    raise RuntimeError(
                        f"Claude game MCP failed: {type(host.failure).__name__}: "
                        f"{host.failure}"
                    )
                if time.monotonic() >= init_deadline:
                    raise TimeoutError("Claude Code emitted no initialization envelope.")
                try:
                    item = stream_queue.get(timeout=0.1)
                except queue.Empty:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Claude Code exited before runtime initialization."
                        )
                    continue
                if item is closed:
                    closed_streams += 1
                    continue
                source, line = item
                if source != "stdout":
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Claude Code emitted invalid stream JSON during initialization."
                    ) from exc
                if message.get("type") != "system" or message.get("subtype") != "init":
                    continue
                init_envelope = message
                model = message.get("model")
                if isinstance(model, str):
                    resolved_models.add(model)
            sanitized_init = sanitize_init_envelope(init_envelope)
            validate_instruction_envelope(
                sanitized_init,
                display_size=self.display_size,
                allow_slash_commands=not disable_slash_commands,
                include_notes=self.notes_enabled,
                include_evolve=self.evolve is not None,
            )
            if self.expected_model_id is not None:
                actual_model = resolved_model_id_from_envelope(sanitized_init)
                if actual_model != self.expected_model_id:
                    raise RuntimeError(
                        "Claude resolved a different model than the runtime preflight."
                    )
            if before_input is not None:
                before_input()
            host.enable_calls()
            decision_lease = DecisionLeaseWatchdog(
                timeout_seconds=self.decision_lease_timeout,
                started_at=time.monotonic(),
            )
            process.stdin.close()
            deadline = time.monotonic() + self.task_timeout
            last_output_at = time.monotonic()
            while True:
                if host.failure is not None:
                    failure = f"MCP host failed: {type(host.failure).__name__}: {host.failure}"
                    _terminate(process)
                if (
                    host.boundary_delivered.is_set()
                    and not reset_boundary
                    and context_boundary is None
                ):
                    if host.boundary_reason == "compact_checkpoint_saved":
                        context_boundary = "compact_checkpoint_saved"
                    elif host.boundary_reason == "compact_checkpoint":
                        context_boundary = "provider_image_limit"
                    elif host.boundary_reason == "invalid_action_limit":
                        context_boundary = "invalid_action_limit"
                        decision_lease_failure = "invalid_action_limit"
                    elif host.boundary_reason == CONTEXT_ROLLOVER_BOUNDARY:
                        context_boundary = CONTEXT_ROLLOVER_BOUNDARY
                    else:
                        reset_boundary = True
                    time.sleep(0.1)
                    _terminate(process)
                if (
                    context_boundary is None
                    and not reset_boundary
                    and self.credential_yield_requested is not None
                    and self.credential_yield_requested()
                ):
                    _, in_flight_calls, _ = host.tool_activity()
                    if credential_handoff_is_safe(
                        process_running=process.poll() is None,
                        tool_calls_in_flight=in_flight_calls,
                        pending_play_tool_ids=pending_play_tool_ids,
                        unconfirmed_play_calls=host.unconfirmed_play_calls,
                    ):
                        context_boundary = "credential_rate_limit"
                        decision_lease_failure = "credential_rate_limit"
                        host.disable_calls()
                        _terminate(process)
                if time.monotonic() >= deadline:
                    failure = "Claude player timed out."
                    _terminate(process)
                now = time.monotonic()
                completed_calls, in_flight_calls, last_completed_at = (
                    host.tool_activity()
                )
                if (
                    process.poll() is None
                    and decision_lease.expired(
                        now=now,
                        completed_tool_calls=completed_calls,
                        tool_calls_in_flight=in_flight_calls,
                        last_tool_completed_at=last_completed_at,
                    )
                ):
                    decision_lease_failure = "decision_timeout"
                    failure = (
                        "Claude decision lease expired without another tool call."
                    )
                    host.disable_calls()
                    _terminate(process)
                if (
                    self.stream_stall_timeout is not None
                    and process.poll() is None
                    and context_boundary is None
                    and not reset_boundary
                    and in_flight_calls == 0
                    and stream_stalled(
                        now=now,
                        last_output_at=last_output_at,
                        timeout_seconds=self.stream_stall_timeout,
                    )
                ):
                    decision_lease_failure = "stream_stall"
                    failure = (
                        "Claude stream stalled: no output for "
                        f"{self.stream_stall_timeout // 60} minutes."
                    )
                    host.disable_calls()
                    _terminate(process)
                try:
                    item = stream_queue.get(timeout=0.1)
                except queue.Empty:
                    if process.poll() is not None and closed_streams >= 2:
                        break
                    continue
                if item is closed:
                    closed_streams += 1
                    if process.poll() is not None and closed_streams >= 2:
                        break
                    continue
                source, line = item
                if source != "stdout":
                    continue
                last_output_at = time.monotonic()
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    failure = "Claude Code emitted invalid stream JSON."
                    continue
                rate_limit_event = rate_limit_event_from_message(message)
                if rate_limit_event is not None:
                    rate_limit_events.append(rate_limit_event)
                    if self.rate_limit_observer is not None:
                        self.rate_limit_observer(rate_limit_event)
                status = api_error_status_from_message(message)
                if status is not None:
                    api_error_status = status
                message_type = message.get("type")
                if message_type == "system" and message.get("subtype") == "init":
                    init_envelope = message
                    model = message.get("model")
                    if isinstance(model, str):
                        resolved_models.add(model)
                elif (
                    message_type == "system"
                    and message.get("subtype") == "compact_boundary"
                ):
                    compact_boundaries.append(sanitize_compact_boundary(message))
                    host.acknowledge_compaction()
                    host.disable_calls()
                    context_boundary = "native_compact"
                    _terminate(process)
                elif message_type == "assistant":
                    assistant = message.get("message")
                    if isinstance(assistant, dict):
                        record_pending_play_tool_uses(
                            assistant,
                            pending_play_tool_ids,
                        )
                        model = assistant.get("model")
                        if isinstance(model, str):
                            resolved_models.add(model)
                        for block in assistant.get("content", []):
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") == "text":
                                text = block.get("text")
                                if isinstance(text, str):
                                    final_message = text
                elif message_type == "user":
                    user_message = message.get("message")
                    if isinstance(user_message, dict):
                        acknowledge_completed_play_tools(
                            user_message,
                            pending_play_tool_ids,
                            host.acknowledge_action_result,
                        )
                elif message_type == "result":
                    result_message = message
                    reported_failure = decision_lease_failure_from_result(message)
                    if reported_failure is not None:
                        decision_lease_failure = reported_failure
                        failure = "Claude ended the decision lease with an error."
                    result_text = message.get("result")
                    if isinstance(result_text, str):
                        final_message = result_text
                    model_usage = message.get("modelUsage")
                    if isinstance(model_usage, dict):
                        resolved_models.update(
                            key for key in model_usage if isinstance(key, str)
                        )
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if process is not None:
                _terminate(process)
        finally:
            if process is not None:
                try:
                    process_returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process_returncode = process.wait()
            else:
                process_returncode = 125
            if host.unconfirmed_play_calls and host.failure is None:
                host.wait_for_play_settlement(PLAY_SETTLEMENT_TIMEOUT_SECONDS)
            unconfirmed_play_calls = host.unconfirmed_play_calls
            if (
                context_boundary is None
                and not reset_boundary
                and self.credential_yield_requested is not None
                and self.credential_yield_requested()
            ):
                _, in_flight_calls, _ = host.tool_activity()
                if credential_handoff_is_safe(
                    process_running=False,
                    tool_calls_in_flight=in_flight_calls,
                    pending_play_tool_ids=pending_play_tool_ids,
                    unconfirmed_play_calls=unconfirmed_play_calls,
                ):
                    context_boundary = "credential_rate_limit"
                    decision_lease_failure = "credential_rate_limit"
            if unconfirmed_play_calls:
                decision_lease_failure = "unconfirmed_play"
                failure = "A play call did not reach a confirmed controller result."
            host.stop()
            mcp_config_path.unlink(missing_ok=True)

        if failure:
            with stderr_path.open("a", encoding="utf-8") as stream:
                stream.write(failure + "\n")
        if reset_boundary or context_boundary is not None:
            returncode = 0
        elif failure:
            returncode = 1
        else:
            returncode = process_returncode
        if (
            returncode == 0
            and not reset_boundary
            and context_boundary is None
            and result_message is None
        ):
            returncode = 1
            with stderr_path.open("a", encoding="utf-8") as stream:
                stream.write("Claude Code exited without a result message.\n")
        if returncode != 0 and decision_lease_failure is None:
            decision_lease_failure = "runtime_exit"

        observed_session_id = None
        if init_envelope is not None and isinstance(init_envelope.get("session_id"), str):
            observed_session_id = init_envelope["session_id"]
        elif init_envelope is not None:
            observed_session_id = requested_session_id
        session_image_blocks = host.image_blocks_delivered
        if observed_session_id is not None:
            self._session_image_blocks[observed_session_id] = session_image_blocks
        self._session_image_blocks[requested_session_id] = session_image_blocks
        return ClaudeResult(
            segment=segment,
            returncode=returncode,
            process_returncode=process_returncode,
            final_message=final_message,
            session_id=observed_session_id,
            resolved_models=tuple(sorted(resolved_models)),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            input_path=input_path,
            command_path=command_path,
            bridge_log_path=bridge_log_path,
            compact_events_path=compact_events_path,
            usage=usage_from_result(result_message),
            api_error_status=api_error_status,
            session_image_blocks=session_image_blocks,
            peak_session_image_blocks=host.peak_image_blocks,
            reset_boundary=reset_boundary,
            context_boundary=context_boundary,
            init_envelope=sanitize_init_envelope(init_envelope),
            compact_boundaries=tuple(compact_boundaries),
            decision_lease_failure=decision_lease_failure,
            unconfirmed_play_calls=unconfirmed_play_calls,
            rate_limit_events=tuple(rate_limit_events),
        )

    def _write_mcp_config(
        self,
        path: Path,
        token: str,
        *,
        bridge_host: str,
        bridge_port: int,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "game": {
                            "type": "stdio",
                            "command": "/usr/bin/env",
                            "args": [
                                *(
                                    argument
                                    for name in CREDENTIAL_ENV_NAMES
                                    for argument in ("-u", name)
                                ),
                                "python3",
                                CONTAINER_MCP_BRIDGE,
                            ],
                            "env": {
                                "ARC3_GAME_HOST": bridge_host,
                                "ARC3_GAME_PORT": str(bridge_port),
                                "ARC3_GAME_TOKEN": token,
                            },
                        }
                    }
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)

    def _write_settings(
        self,
        path: Path,
        compact_event_log_name: str,
    ) -> None:
        unset_credentials = " ".join(
            f"-u {name}" for name in CREDENTIAL_ENV_NAMES
        )
        hook_environment = [
            f"ARC3_COMPACT_EVENT_LOG=/claude-io/{compact_event_log_name}"
        ]
        marker_environment = (
            (
                "ARC3_COMPACT_CHECKPOINT_REQUEST",
                getattr(self.controller, "compact_checkpoint_marker", None),
            ),
            (
                "ARC3_COMPACT_CHECKPOINT_READY",
                getattr(self.controller, "compact_checkpoint_ready", None),
            ),
        )
        for name, marker in marker_environment:
            # Without GUIDE.md / WORKING.md there is no agent-authored checkpoint
            # to wait for, so the hook only records compaction events.
            if marker is not None and self.notes_enabled:
                hook_environment.append(
                    f"{name}={self._container_io_path(Path(marker))}"
                )
        hook = {
            "type": "command",
            "command": (
                f"/usr/bin/env {unset_credentials} "
                f"{' '.join(hook_environment)} "
                f"python3 {CONTAINER_COMPACT_HOOK}"
            ),
            "timeout": 10,
        }
        path.write_text(
            json.dumps(
                {
                    "autoCompactEnabled": True,
                    "hooks": {
                        "PreCompact": [{"hooks": [hook]}],
                        "PostCompact": [{"hooks": [hook]}],
                        "Stop": [{"hooks": [hook]}],
                    }
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)

    def _container_io_path(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.io_dir)
        except ValueError as exc:
            raise ValueError(
                "Compact markers must live inside the Claude I/O dir."
            ) from exc
        return str(Path("/claude-io") / relative)

    def system_prompt(self) -> str:
        """AGENTS.md plus, with --evolve, the live skills / tools / hooks index."""
        if self.evolve is None:
            return self.agents_instructions
        return self.agents_instructions.rstrip() + "\n\n" + self.evolve.render_index() + "\n"

    def _docker_command(
        self,
        *,
        mcp_config_path: Path,
        settings_path: Path,
        bridge_host: str,
        bridge_port: int,
        session_id: str,
        resume: bool,
        disable_slash_commands: bool = True,
        include_partial_messages: bool = False,
    ) -> list[str]:
        uid = str(os.getuid())
        gid = str(os.getgid())
        bridge = Path(__file__).with_name("mcp_bridge.py").resolve()
        compact_hook = Path(__file__).with_name("compact_hook.py").resolve()
        allowed_tools = ",".join(
            f'mcp__game__{tool["name"]}'
            for tool in build_tools(
                self.display_size,
                include_compact_checkpoint=False,
                include_notes=self.notes_enabled,
                include_evolve=self.evolve is not None,
            )
        )
        command = [
            "docker",
            "run",
            "-i",
            "--rm",
            "--pull=never",
            "--network",
            "bridge",
            "--add-host",
            f"{bridge_host}:host-gateway",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--memory",
            "16g",
            "--cpus",
            "4",
            "--user",
            f"{uid}:{gid}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m",
            "--tmpfs",
            "/home/claude:rw,nosuid,nodev,size=64m",
            "--tmpfs",
            "/workspace:rw,nosuid,nodev,size=16m",
            "-e",
            claude_credential_env_name(self.auth_mode),
            "-e",
            "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1",
            "-e",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1",
            "-e",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS=1",
            "-e",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1",
            "-e",
            "CLAUDE_CODE_DISABLE_CRON=1",
            "-e",
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1",
            "-e",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
            "-e",
            "CLAUDE_CODE_AUTO_CONNECT_IDE=false",
            *self._auto_compact_environment(),
            "-e",
            "CLAUDE_CONFIG_DIR=/claude-config",
            "-e",
            "HOME=/home/claude",
            "-e",
            "TERM=dumb",
            "-e",
            "NO_COLOR=1",
            "-e",
            f"MAX_MCP_OUTPUT_TOKENS={MAX_MCP_OUTPUT_TOKENS}",
            "-v",
            f"{bridge}:{CONTAINER_MCP_BRIDGE}:ro",
            "-v",
            f"{compact_hook}:{CONTAINER_COMPACT_HOOK}:ro",
            "-v",
            f"{self.claude_config_dir}:/claude-config:rw",
            "-v",
            f"{self.io_dir}:/claude-io:rw",
            "-w",
            "/workspace",
            self.image,
            CONTAINER_CLAUDE_BIN,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-hook-events",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--system-prompt",
            self.system_prompt(),
            "--setting-sources",
            "",
            "--settings",
            f"/claude-config/{settings_path.name}",
            "--mcp-config",
            f"/claude-config/{mcp_config_path.name}",
            "--strict-mcp-config",
            "--tools",
            "",
            "--allowedTools",
            allowed_tools,
            "--permission-mode",
            CLAUDE_PERMISSION_MODE,
            "--prompt-suggestions",
            "false",
            "--no-chrome",
        ]
        if include_partial_messages:
            # Partial chunks let the opt-in stall watchdog see live generation.
            command.append("--include-partial-messages")
        if disable_slash_commands:
            command.append("--disable-slash-commands")
        if resume:
            command.extend(["--resume", session_id])
        else:
            command.extend(["--session-id", session_id])
        return command

    def _auto_compact_environment(self) -> list[str]:
        window = self.auto_compact_window
        percent = self.auto_compact_percent
        if window is None:
            return []
        return [
            "-e",
            f"CLAUDE_CODE_AUTO_COMPACT_WINDOW={window}",
            "-e",
            f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE={percent}",
        ]


def stream_user_message(prompt: str, image_paths: Any) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        path = Path(image_path)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                },
            }
        )
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


def sanitize_stream_message(message: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(message))
    content = sanitized.get("message", {}).get("content", [])
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        source = block.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("data"), str):
            continue
        data = source["data"]
        source["data"] = {
            "encoded_chars": len(data),
            "sha256": hashlib.sha256(data.encode("ascii")).hexdigest(),
        }
    return sanitized


def record_pending_play_tool_uses(
    assistant_message: dict[str, Any],
    pending: set[str],
) -> None:
    for block in assistant_message.get("content", []):
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == "mcp__game__play"
            and isinstance(block.get("id"), str)
        ):
            pending.add(block["id"])


def acknowledge_completed_play_tools(
    user_message: dict[str, Any],
    pending: set[str],
    acknowledge: Callable[[], None],
) -> None:
    returned_ids = {
        block["tool_use_id"]
        for block in user_message.get("content", [])
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("tool_use_id"), str)
    }
    completed = returned_ids & pending
    if not completed:
        return
    pending.difference_update(completed)
    acknowledge()


def credential_handoff_is_safe(
    *,
    process_running: bool,
    tool_calls_in_flight: int,
    pending_play_tool_ids: set[str],
    unconfirmed_play_calls: int,
) -> bool:
    if tool_calls_in_flight or unconfirmed_play_calls:
        return False
    return not process_running or not pending_play_tool_ids


def sanitize_init_envelope(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    allowed = {
        "type",
        "subtype",
        "session_id",
        "tools",
        "mcp_servers",
        "model",
        "permissionMode",
        "slash_commands",
        "claude_code_version",
        "output_style",
        "skills",
        "plugins",
        "capabilities",
    }
    return {key: value[key] for key in allowed if key in value}


def validate_instruction_envelope(
    envelope: dict[str, Any] | None,
    *,
    display_size: int,
    allow_slash_commands: bool = False,
    include_notes: bool = True,
    include_evolve: bool = False,
) -> None:
    if not isinstance(envelope, dict):
        raise RuntimeError("Claude Code emitted no initialization envelope.")
    expected_tools = {
        f'mcp__game__{tool["name"]}'
        for tool in build_tools(
            display_size,
            include_compact_checkpoint=False,
            include_notes=include_notes,
            include_evolve=include_evolve,
        )
    }
    actual_tools = envelope.get("tools")
    if (
        not isinstance(actual_tools, list)
        or len(actual_tools) != len(expected_tools)
        or set(actual_tools) != expected_tools
    ):
        raise RuntimeError("Claude runtime exposed an unexpected tool surface.")
    for key in ("skills", "plugins"):
        if envelope.get(key) != []:
            raise RuntimeError(f"Claude runtime exposed unexpected {key}.")
    if not allow_slash_commands and envelope.get("slash_commands") != []:
        raise RuntimeError("Claude runtime exposed unexpected slash_commands.")
    if envelope.get("permissionMode") != CLAUDE_PERMISSION_MODE:
        raise RuntimeError("Claude runtime used an unexpected permission mode.")
    if envelope.get("claude_code_version") != PINNED_CLAUDE_VERSION:
        raise RuntimeError("Claude runtime used an unexpected Claude Code version.")
    servers = envelope.get("mcp_servers")
    if not isinstance(servers, list) or len(servers) != 1:
        raise RuntimeError("Claude runtime did not expose exactly one MCP server.")
    server = servers[0]
    if not isinstance(server, dict) or server.get("name") != "game":
        raise RuntimeError("Claude runtime connected an unexpected MCP server.")
    status = server.get("status")
    if isinstance(status, str) and status.lower() != "connected":
        raise RuntimeError("Claude game MCP server is not connected.")


def resolved_model_id_from_envelope(envelope: dict[str, Any] | None) -> str:
    model = envelope.get("model") if isinstance(envelope, dict) else None
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError("Claude Code did not report its resolved model ID.")
    normalized = model.strip()
    aliases = {"opus", "sonnet", "haiku"}
    if normalized.lower() in aliases:
        raise RuntimeError(
            "Claude Code reported a mutable model alias instead of a resolved model ID."
        )
    return normalized


def resolved_model_id(result: ClaudeResult) -> str:
    return resolved_model_id_from_envelope(result.init_envelope)


def sanitize_compact_boundary(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {"type", "subtype", "session_id", "compact_metadata"}
    return {key: value[key] for key in allowed if key in value}


def usage_from_result(result: dict[str, Any] | None) -> ClaudeUsage | None:
    if not isinstance(result, dict):
        return None
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return None
    return ClaudeUsage(
        input_tokens=_integer(usage.get("input_tokens")),
        output_tokens=_integer(usage.get("output_tokens")),
        cache_creation_input_tokens=_integer(
            usage.get("cache_creation_input_tokens")
        ),
        cache_read_input_tokens=_integer(usage.get("cache_read_input_tokens")),
        cost_usd=(
            float(result["total_cost_usd"])
            if isinstance(result.get("total_cost_usd"), (int, float))
            else None
        ),
        num_turns=(
            result["num_turns"] if type(result.get("num_turns")) is int else None
        ),
    )


def api_error_status_from_message(message: object) -> int | None:
    if not isinstance(message, dict):
        return None
    for key in ("api_error_status", "apiErrorStatus"):
        status = message.get(key)
        if type(status) is int:
            return status
    return None


def decision_lease_failure_from_result(message: object) -> str | None:
    if not isinstance(message, dict) or message.get("type") != "result":
        return None
    result = message.get("result")
    if isinstance(result, str):
        normalized = result.lower()
        if "output token" in normalized and "maximum" in normalized:
            return "output_token_limit"
    if (
        message.get("is_error") is True
        or message.get("terminal_reason") == "api_error"
        or message.get("subtype") == "error_during_execution"
    ):
        return "runtime_exit"
    return None


def _integer(value: object) -> int:
    return value if type(value) is int else 0


def _start_stream_reader(
    stream: Any,
    source: str,
    output: queue.Queue[tuple[str, str] | object],
    path: Path,
    closed: object,
) -> threading.Thread:
    def read() -> None:
        with path.open("w", encoding="utf-8") as log:
            for line in stream:
                log.write(line)
                log.flush()
                output.put((source, line))
        output.put(closed)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    return thread


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
