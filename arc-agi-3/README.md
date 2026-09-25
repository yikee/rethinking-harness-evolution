# ARC-AGI-3: the four agent settings

Code to reproduce the ARC-AGI-3 experiments. Every run plays one game with one
of two agent runtimes (Claude Code or Codex CLI) in one of four settings:

| setting   | what runs                                               | flag / entrypoint                                 |
|-----------|---------------------------------------------------------|---------------------------------------------------|
| `plain`   | the bare CLI agent on the host, no harness              | `plain/run_plain.py`                              |
| `nonotes` | VISTA harness with agent notes disabled                 | `VISTA/scripts/run_arc3_*.py --no-guide-working`  |
| `default` | VISTA harness (agent-authored `GUIDE.md` / `WORKING.md`)| `VISTA/scripts/run_arc3_*.py`                     |
| `evolve`  | VISTA + agent-authored skills, tools and hooks          | `VISTA/scripts/run_arc3_*.py --evolve`            |

Paper configuration: `claude-opus-4-8` at effort `xhigh` and `gpt-5.6-terra` at
effort `high`, 2000 game actions per run, scored by the ARC-AGI-3 scorecard
(RHAE, 0-100).

```
arc-agi-3/
  run_setting.sh   one-line launcher for <runtime> x <setting> x <game>
  VISTA/           harness (Python package + Dockerfiles + tests); see VISTA/README.md
  plain/           plain baseline: run_plain.py (supervisor + local game proxy),
                   arc_cli.py (the agent's only game interface), PROMPT.md
```

## Setup

Requirements: Python 3.12+, Docker (Engine on Linux, Docker Desktop on macOS),
an [ARC Prize platform](https://arcprize.org/platform) API key, and an
Anthropic and/or OpenAI API key. The `plain` setting additionally needs the
CLI agents on the host `PATH`; the VISTA settings run them inside Docker and
need no host binaries.

1. Install the harness into a virtualenv (the same venv also runs the plain
   baseline):

   ```bash
   cd arc-agi-3/VISTA
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -e .
   ```

2. Create `VISTA/.env` from the template and fill in the keys. Both the harness
   and `plain/run_plain.py` read this one file:

   ```bash
   cp .env.example .env && chmod 600 .env
   ```

   ```dotenv
   ARC_API_KEY=...                       # arcprize.org -> profile -> API Keys
   ARC_BASE_URL=https://three.arcprize.org
   ANTHROPIC_API_KEY=sk-ant-api03-...    # Claude runs (VISTA and plain)
   OPENAI_API_KEY=sk-...                 # Codex runs (VISTA and plain)
   ```

3. Build the player image(s). Each image bakes in the pinned runtime
   (Claude Code 2.1.220, Codex CLI 0.145.0) and builds for amd64 and arm64:

   ```bash
   docker build -t arc3-claude-player:0.1 -f Dockerfile.claude-player .
   docker build -t arc3-codex-player:0.1  -f Dockerfile.codex-player .
   ```

4. For the `plain` setting only, install the CLIs on the host:

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash -s 2.1.220   # -> ~/.local/bin/claude
   npm install -g @openai/codex                                # our runs used codex-cli 0.154.0
   ```

   `run_plain.py` prepends `~/.local/bin` and `VISTA/.venv/bin` to `PATH`, so
   the versioned Claude install above is picked up without further configuration.

Full details on the harness (auth modes, compaction, stall watchdog, batch
runs, tests) are in `VISTA/README.md`.

## Run

`run_setting.sh` picks the paper's model/effort for the runtime and dispatches
to the right script:

```bash
./run_setting.sh <claude|codex> <plain|nonotes|default|evolve> <game-id> [extra VISTA args]

./run_setting.sh claude plain   ls20
./run_setting.sh claude nonotes ls20
./run_setting.sh claude default ls20
./run_setting.sh claude evolve  ls20
./run_setting.sh codex  evolve  ls20

MODEL=gpt-5.5 EFFORT=xhigh ./run_setting.sh codex default ls20   # override model/effort
./run_setting.sh claude default ls20 --max-steps 500             # extra args go to VISTA
```

Short game ids (`ls20`) are accepted; both launchers resolve them to the
current versioned id.

The equivalent raw commands, from `arc-agi-3/`:

```bash
# plain
VISTA/.venv/bin/python plain/run_plain.py --agent claude --model claude-opus-4-8 --effort xhigh --game-id ls20
VISTA/.venv/bin/python plain/run_plain.py --agent codex  --model gpt-5.6-terra   --effort high  --game-id ls20

# VISTA (run from arc-agi-3/VISTA)
.venv/bin/python scripts/run_arc3_claude.py --game-id ls20 --model claude-opus-4-8 --effort xhigh --no-guide-working
.venv/bin/python scripts/run_arc3_claude.py --game-id ls20 --model claude-opus-4-8 --effort xhigh
.venv/bin/python scripts/run_arc3_claude.py --game-id ls20 --model claude-opus-4-8 --effort xhigh --evolve
.venv/bin/python scripts/run_arc3_codex.py  --game-id ls20 --model gpt-5.6-terra   --effort high  --evolve
```

Runs are independent processes, so several can run concurrently; each opens its
own scorecard. The binding limits are the API rate limits of your accounts
(we ran up to 8 Claude or 5 Codex runs at once) and, for VISTA, one Docker
container per run.

## What each setting does

**plain.** `run_plain.py` opens the game, starts a local HTTP proxy that holds
the ARC credentials, and launches the agent (`claude -p ...` or `codex exec
...`) in a fresh run directory containing only `PROMPT.md`, `arc_cli.py` and
`frames/`. The agent acts with `python arc_cli.py act ACTIONn [x y]`, which
prints one JSON line of state and writes the frame to `frames/latest.png`.
The proxy enforces one action per 10 s (defeats scripted click sweeps),
verifies that every action was applied to the intended game, and closes the
scorecard at the end. The agent never sees `ARC_API_KEY`; Codex gets a private
`CODEX_HOME` so it bills the API key rather than a ChatGPT login. The
supervisor kills the agent after 45 min without activity or after
5 h (Claude) / 3 h (Codex) without a new level, and always closes the
scorecard.

**nonotes / default / evolve.** The VISTA harness runs the same CLI agent
inside Docker and exposes the game through MCP tools (`play`, `inspect`, ...),
renders frames, handles context compaction and session recovery, and enforces
the 2000-action budget. `default` gives the agent `read_guide` / `write_guide`
(cross-level notes) and `read_working` / `write_working` (scratchpad);
`--no-guide-working` removes those tools and files; `--evolve` additionally
lets the agent write skills (`skills/*.md`, indexed into its prompt), tools
(`tools/*.py`, run in a sandboxed container via `run_tool`) and hooks
(`hooks.json`, declarative before/after-play rules that annotate or block
actions). All of these are editable at any point of the run. See
`VISTA/README.md` for the exact tool contracts.

## Outputs and scoring

- VISTA: `VISTA/runs/run_<timestamp>/`. `manifest.json` holds the
  configuration and, once the scorecard is closed,
  `scorecard_summary.score` (RHAE), levels and action counts; `player/` has the
  agent-visible files (`GUIDE.md`, `WORKING.md`, `skills/`, `tools/`,
  `hooks.json`, screenshots); `private/action_log.jsonl` is the full action
  trace and `private/evolve_log.jsonl` records every skill/tool/hook edit.
- plain: `plain/runs/<timestamp>_<agent>_<model>_<effort>/`. `summary.json`
  holds `score` (RHAE), `levels_completed`, `steps`, `termination` and the
  scorecard URL; `agent.jsonl` is the agent's own stream; `frames/` has every
  frame.

Both directories are git-ignored.

Supervision used in the paper, in case you want to reproduce the stopping
rule for VISTA runs (the harness itself only stops at WIN or the action
budget): a run was interrupted with SIGINT after 5 h (Claude) / 3 h (Codex)
without a new level, which closes the scorecard and records the partial score.

Plain agents have network access. One plain Codex run in our set fetched a
public solver for the game; we scored it 0 and re-ran it. Check `agent.jsonl`
for `git clone` / solver lookups before trusting plain scores.
