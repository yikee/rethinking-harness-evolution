#!/usr/bin/env python
"""Run one plain-agent ARC-AGI-3 baseline (Claude Code or Codex, no VISTA harness).

    python run_plain.py --agent claude --model claude-opus-4-8 --effort xhigh --game-id s5i5
    python run_plain.py --agent codex  --model gpt-5.5         --effort high  --game-id s5i5

Creates runs/<timestamp>_<agent>_<model>_<effort>/, opens one scorecard + game
through a localhost proxy that holds the ARC credentials (the agent never sees
them and can only act on this one game), starts the agent on PROMPT.md, and
supervises it with the same rules as the VISTA watcher: kill after
IDLE_KILL_MIN minutes without activity or PROGRESS_KILL_H hours without a new
level. Closes the scorecard and writes summary.json when the agent finishes.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
VISTA = HERE.parent / "VISTA"
RUNS = HERE / "runs"
IDLE_KILL_MIN = 45
PROGRESS_KILL_H = {"claude": 5, "codex": 3}  # hours without a new level, per agent
POLL_SEC = 60
MAX_STEPS = 2000
MIN_ACTION_INTERVAL_SEC = 10  # stops scripted click sweeps; real turns are far slower
ACTIONS = {"RESET", *(f"ACTION{i}" for i in range(1, 8))}

CLAUDE_TOOLS = "Bash(python *),Bash(python3 *),Bash(ls *),Bash(cat *),Read,Write,Edit,Glob,Grep"


def log(run_dir: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (run_dir / "watch.log").open("a") as fh:
        fh.write(line + "\n")


def dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (VISTA / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


class GameProxy:
    """One ARC scorecard + one game session, behind a localhost HTTP server."""

    def __init__(self, base_url: str, api_key: str, game_id: str, run_dir: Path) -> None:
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.game_id = game_id
        self.run_dir = run_dir
        self.session = requests.Session()  # keeps the sticky server cookie
        self.lock = threading.Lock()
        self.card_id: str | None = None
        self.guid: str | None = None
        self.steps = 0
        self.levels = 0
        self.state: str | None = None
        self.last_action_at = 0.0
        self.last_progress_at = time.time()
        self.closed: dict | None = None

    def _post(self, path: str, payload: dict) -> dict:
        r = self.session.post(f"{self.base}{path}", json=payload, headers={"X-Api-Key": self.key}, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"ARC API {r.status_code} on {path}: {r.text[:300]}")
        return r.json()

    def _record(self, data: dict) -> dict:
        self.steps += 1
        self.guid = data.get("guid") or self.guid
        levels = data.get("levels_completed", 0)
        if levels > self.levels:
            self.last_progress_at = time.time()
        self.levels = levels
        self.state = data.get("state")
        self.write_state()
        return {
            "step": self.steps,
            "steps_remaining": MAX_STEPS - self.steps,
            "state": data["state"],
            "levels_completed": data["levels_completed"],
            "win_levels": data["win_levels"],
            "available_actions": data["available_actions"],
            "frame": data["frame"],
        }

    def status(self) -> dict:
        return {
            "card_id": self.card_id, "game_id": self.game_id, "steps": self.steps,
            "steps_remaining": MAX_STEPS - self.steps, "levels_completed": self.levels,
            "state": self.state, "last_progress_at": self.last_progress_at,
            "updated_at": time.time(), "closed": self.closed is not None,
        }

    def write_state(self) -> None:
        (self.run_dir / ".arc_state.json").write_text(json.dumps(self.status(), indent=1))

    def resolve_game_id(self) -> None:
        """Resolve a short id like `s5i5` to the versioned id the server expects.

        The server answers RESET for the short id but then silently ignores
        every action (action_input.id == 0), so we must do what arc.make does
        and look the full id up in /api/games.
        """
        r = self.session.get(f"{self.base}/api/games", headers={"X-Api-Key": self.key}, timeout=60)
        r.raise_for_status()
        ids = [g["game_id"] for g in r.json()]
        if self.game_id in ids:
            return
        matches = [g for g in ids if g.startswith(self.game_id + "-")]
        if len(matches) != 1:
            raise RuntimeError(f"game id {self.game_id!r} matches {matches or 'nothing'} in /api/games")
        self.game_id = matches[0]

    def open(self) -> dict:
        with self.lock:
            self.resolve_game_id()
            self.card_id = self._post("/api/scorecard/open", {"tags": []})["card_id"]
            data = self._post("/api/cmd/RESET", {"card_id": self.card_id, "game_id": self.game_id})
            self.last_action_at = time.time()
            return self._record(data)

    def act(self, action: str, x: int | None, y: int | None) -> tuple[int, dict]:
        with self.lock:
            if self.closed is not None:
                return 409, {"error": "the scorecard is already closed"}
            if action not in ACTIONS:
                return 400, {"error": f"unknown action {action!r}"}
            if self.steps >= MAX_STEPS:
                return 409, {"error": "step budget exhausted; run `python arc_cli.py close`"}
            if self.state == "WIN":
                return 409, {"error": "the game is already won; run `python arc_cli.py close`"}
            wait = MIN_ACTION_INTERVAL_SEC - (time.time() - self.last_action_at)
            if wait > 0:
                return 429, {"error": f"actions are limited to one every {MIN_ACTION_INTERVAL_SEC} s; "
                                      f"look at the frame and retry in {wait:.0f} s"}
            payload = {"card_id": self.card_id, "game_id": self.game_id, "guid": self.guid}
            if action == "ACTION6":
                if x is None or y is None or not (0 <= x <= 63 and 0 <= y <= 63):
                    return 400, {"error": "ACTION6 needs x and y in 0-63"}
                payload.update(x=x, y=y)
            try:
                data = self._post(f"/api/cmd/{action}", payload)
            except RuntimeError as exc:
                return 502, {"error": str(exc)}
            echoed = (data.get("action_input") or {}).get("id")
            expected = 0 if action == "RESET" else int(action[-1])
            if echoed not in (expected, None) and not (action == "RESET" and echoed in (0, "RESET")):
                return 502, {"error": f"server ignored {action} (echoed action id {echoed!r}); not counted"}
            self.last_action_at = time.time()
            return 200, self._record(data)

    def close(self) -> dict:
        with self.lock:
            if self.closed is None:
                self.closed = self._post("/api/scorecard/close", {"card_id": self.card_id})
                self.closed["card_id"] = self.card_id
                (self.run_dir / "scorecard.json").write_text(json.dumps(self.closed, indent=2))
                self.write_state()
            return self.closed


def serve(proxy: GameProxy) -> str:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/status":
                self._send(200, proxy.status())
            else:
                self._send(404, {"error": "unknown endpoint"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            if self.path == "/act":
                code, out = proxy.act(str(body.get("action")), body.get("x"), body.get("y"))
                self._send(code, out)
            elif self.path == "/close":
                self._send(200, proxy.close())
            else:
                self._send(404, {"error": "unknown endpoint"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def agent_env(run_dir: Path, agent: str, secrets: dict[str, str], proxy_url: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in {"ARC_API_KEY", "ARC_BASE_URL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"}}
    env["PATH"] = f"{VISTA / '.venv/bin'}:{Path.home() / '.local/bin'}:{env['PATH']}"
    env["ARC_PROXY_URL"] = proxy_url
    if agent == "claude":
        env["ANTHROPIC_API_KEY"] = secrets["ANTHROPIC_API_KEY"]
    else:
        # Codex ignores OPENAI_API_KEY when an auth.json exists, so give it a
        # private CODEX_HOME whose auth.json is the API key (platform billing),
        # not the host's ChatGPT login.
        home = run_dir / "codex_home"
        home.mkdir(mode=0o700)
        (home / "auth.json").write_text(json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": secrets["OPENAI_API_KEY"]}))
        env["CODEX_HOME"] = str(home)
    return env


def agent_command(agent: str, model: str, effort: str, prompt: str, run_dir: Path) -> list[str]:
    if agent == "claude":
        return [
            "claude", "-p", prompt,
            "--model", model,
            "--effort", effort,
            "--allowedTools", CLAUDE_TOOLS,
            "--output-format", "stream-json",
            "--verbose",
        ]
    return [
        "codex", "exec",
        "--model", model,
        "-c", f"model_reasoning_effort={effort}",
        "--sandbox", "workspace-write",
        "-c", "sandbox_workspace_write.network_access=true",
        "--skip-git-repo-check",
        "--json",
        "-C", str(run_dir),
        prompt,
    ]


def last_activity(run_dir: Path) -> float:
    latest = 0.0
    for name in (".arc_state.json", "agent.jsonl"):
        p = run_dir / name
        if p.exists():
            latest = max(latest, p.stat().st_mtime)
    return latest


def stop(proc: subprocess.Popen, run_dir: Path, reason: str) -> None:
    log(run_dir, f"STOPPING agent ({reason}) -> SIGINT pgid {proc.pid}")
    for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + wait
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(1)


def claude_cost(run_dir: Path) -> float | None:
    cost = None
    try:
        for line in (run_dir / "agent.jsonl").read_text().splitlines():
            if '"result"' not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                cost = event.get("total_cost_usd", cost)
    except OSError:
        pass
    return cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["claude", "codex"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--game-id", default="s5i5")
    parser.add_argument("--idle-kill-min", type=int, default=IDLE_KILL_MIN)
    parser.add_argument("--progress-kill-h", type=float, default=None,
                        help="hours without a new level before stopping (default: claude 5, codex 3)")
    args = parser.parse_args()
    if args.progress_kill_h is None:
        args.progress_kill_h = PROGRESS_KILL_H[args.agent]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"{stamp}_{args.agent}_{args.model}_{args.effort}"
    run_dir.mkdir(parents=True)
    shutil.copy(HERE / "arc_cli.py", run_dir / "arc_cli.py")
    prompt = (HERE / "PROMPT.md").read_text()
    (run_dir / "PROMPT.md").write_text(prompt)
    secrets = dotenv()
    started = time.time()
    log(run_dir, f"plain baseline: agent={args.agent} model={args.model} effort={args.effort} game={args.game_id}")

    proxy = GameProxy(secrets["ARC_BASE_URL"], secrets["ARC_API_KEY"], args.game_id, run_dir)
    proxy_url = serve(proxy)
    env = agent_env(run_dir, args.agent, secrets, proxy_url)
    opened = proxy.open()
    log(run_dir, f"game open via {proxy_url}: card_id={proxy.card_id} guid={proxy.guid} "
                 f"available_actions={opened['available_actions']} win_levels={opened['win_levels']}")
    # Render the first frame for the agent the same way `act` does.
    subprocess.run(["python", "-c",
                    "import json,sys; sys.argv=['arc_cli.py']; import arc_cli; arc_cli.report(json.load(sys.stdin))"],
                   cwd=run_dir, env=env, input=json.dumps(opened), text=True, check=True, capture_output=True)

    summary = {
        "agent": args.agent, "model": args.model, "effort": args.effort, "game_id": args.game_id,
        "card_id": proxy.card_id, "run_dir": str(run_dir), "started_at": stamp,
        "idle_kill_min": args.idle_kill_min, "progress_kill_h": args.progress_kill_h,
        "min_action_interval_sec": MIN_ACTION_INTERVAL_SEC, "max_steps": MAX_STEPS,
        "prompt_file": "PROMPT.md",
        "allowed_tools": CLAUDE_TOOLS if args.agent == "claude" else "codex exec --sandbox workspace-write + network",
        "billing": "ANTHROPIC_API_KEY" if args.agent == "claude" else "OPENAI_API_KEY (private CODEX_HOME)",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    cmd = agent_command(args.agent, args.model, args.effort, prompt, run_dir)
    (run_dir / "agent.command.json").write_text(json.dumps(cmd, indent=1))
    with (run_dir / "agent.jsonl").open("ab") as out, (run_dir / "agent.stderr.log").open("ab") as err:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    log(run_dir, f"agent started pid {proc.pid}: {cmd[0]} {cmd[1]} ... --model {args.model}")

    termination = "agent_exit"
    last_report = 0.0
    while proc.poll() is None:
        time.sleep(POLL_SEC)
        now = time.time()
        idle_min = (now - max(last_activity(run_dir), started)) / 60
        no_progress_h = (now - proxy.last_progress_at) / 3600
        if now - last_report >= 600:
            log(run_dir, f"alive: steps={proxy.steps} levels={proxy.levels} state={proxy.state} "
                         f"idle={idle_min:.1f}min last_gain={no_progress_h:.1f}h ago")
            last_report = now
        if proxy.state == "WIN" or proxy.closed is not None:
            continue  # let the agent wrap up and close the card itself
        if idle_min >= args.idle_kill_min:
            termination = f"idle_kill ({idle_min:.0f} min without activity)"
            stop(proc, run_dir, termination)
            break
        if no_progress_h >= args.progress_kill_h:
            termination = f"no_progress_kill ({no_progress_h:.1f} h at level {proxy.levels})"
            stop(proc, run_dir, termination)
            break

    rc = proc.poll()
    log(run_dir, f"agent finished rc={rc} ({termination})")
    try:
        closed = proxy.close()
        log(run_dir, f"scorecard closed: score={closed.get('score')} https://arcprize.org/scorecards/{proxy.card_id}")
    except Exception as exc:  # noqa: BLE001
        log(run_dir, f"scorecard close failed: {exc}")
        closed = {}
    summary.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_hours": round((time.time() - started) / 3600, 2),
        "termination": termination, "agent_returncode": rc,
        "steps": proxy.steps, "levels_completed": proxy.levels, "final_state": proxy.state,
        "score": closed.get("score"), "scorecard_url": f"https://arcprize.org/scorecards/{proxy.card_id}",
        "claude_cost_usd": claude_cost(run_dir) if args.agent == "claude" else None,
    })
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(run_dir, f"summary: score={closed.get('score')} levels={proxy.levels} steps={proxy.steps} termination={termination}")


if __name__ == "__main__":
    main()
