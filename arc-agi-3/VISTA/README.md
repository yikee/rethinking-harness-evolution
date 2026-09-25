<h2 align="center">VISTA: A Visual Harness for Reasoning in an Interactive World</h2>

<p align="center">
  Qiushi Han* &nbsp;&middot;&nbsp; Keya Hu* &nbsp;&middot;&nbsp; Linlu Qiu* &nbsp;&middot;&nbsp; Cathy Wu &nbsp;&middot;&nbsp; Kaiming He
  <br>
  Massachusetts Institute of Technology
  <br>
  <sub>* co-leads</sub>
</p>

<p align="center">
  <a href="https://vista-research.github.io/">Blog post</a>
</p>

<p align="center">
  <img src="assets/vista-18-games.gif" alt="VISTA playing ARC-AGI-3 games">
</p>

VISTA gives general-purpose multimodal models a continuous visual interface to
interactive environments. It preserves environment frames as visual memory so
the agent can revisit original evidence while reasoning and acting over long
horizons.

With Claude Opus 5.0, VISTA completes all 25 public ARC-AGI-3 games with a 100%
win rate and a Relative Human Action Efficiency (RHAE) score of 100.

Scorecards:

| Runtime | Model | Effort | RHAE |
| --- | --- | --- | ---: |
| Codex CLI | GPT-5.6 Sol | max | [99](https://arcprize.org/scorecards/abda28e2-d605-4e81-9efc-cf63dda06df5) |
| Claude Code | Opus 5.0 | xhigh | [100](https://arcprize.org/scorecards/39be671a-d0cc-48b4-ae08-1db4abc44c83) |

## Interface

```text
observe the current visual
-> reason and use memory as needed
-> execute one game action
-> observe the resulting visual
```

Every environment frame is archived. The final frame becomes the current
observation, and earlier final or animation frames remain available through
visual memory.

Player instructions:

```text
# Visual game task

Complete the game with as few game actions as possible.

Build and use a compact, revisable model of the game and its current state. Update it as new evidence changes what is supported.

Before each `play`, briefly state what you expect to see. Afterward, briefly state all visible changes, expected or not.

Keep concise, durable, revisable game understanding in `GUIDE.md`; use `WORKING.md` as a scratchpad when useful.
```

Available tools:

- use `play` to execute a game action;
- use `inspect` to revisit selected visual frames and regions;
- use `read_pixels` to sample exact colors from selected image regions;
- use `history` to revisit prior actions and environment results;
- use `GUIDE.md` and `WORKING.md` for persistent and working memory.

## Setup

VISTA runs on Linux x86_64 with Python 3.12+, Docker Engine, and either
Codex CLI 0.145.0 or Claude Code 2.1.220. Online and competition runs require
an ARC-AGI-3 API key.

Experimental support is also available for **single-game runs on macOS** with
Docker Desktop, for both `scripts/run_arc3_claude.py` and
`scripts/run_arc3_codex.py`. Both pinned runtimes are baked into their Docker
images, so no host Claude or Codex binary is required. Batch/competition
workflows remain Linux-oriented.

From a repository checkout, install the Python package:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
cp .env.example .env
chmod 600 .env
```

Sign in to the [ARC Prize platform](https://arcprize.org/platform), create a key
under your profile's **API Keys**, and add it to `.env`:

```dotenv
ARC_API_KEY=your-key
ARC_BASE_URL=https://three.arcprize.org
```

### Codex CLI

The pinned Codex CLI 0.145.0 is installed inside `arc3-codex-player:0.1`
(see the Docker section below). Runs can bill either an OpenAI API key or a
ChatGPT plan's Codex quota:

- **API key** (platform billing, all API models incl. `gpt-6-astra`): add
  `OPENAI_API_KEY=sk-...` to `.env` or export it. When the key is present the
  harness uses it automatically and writes a Codex auth file for it at
  `~/.codex/arc3-api-key-auth.json` (override with
  `ARC3_CODEX_API_KEY_AUTH_FILE`). Your ChatGPT login is left untouched.
- **ChatGPT login** (plan quota, only the models your plan lists): create a
  host `~/.codex/auth.json` with the image itself:

  ```bash
  scripts/codex_login.sh          # ChatGPT device-code login
  scripts/codex_login.sh status
  ```

Pick explicitly with `--auth api-key|chatgpt` on `scripts/run_arc3_codex.py`
or `ARC3_CODEX_AUTH=api-key|chatgpt` (also honored by batch runs); the default
`auto` prefers the API key when `OPENAI_API_KEY` is set. Each run's
`manifest.json` records `codex_auth_mode`.

If you already have a Codex CLI installed on the host, its existing
`~/.codex/auth.json` is used as-is for ChatGPT mode. Set
`ARC3_CODEX_AUTH_FILE` to point at a different file, and `ARC3_CODEX_IMAGE`
to use a different image tag.

Compaction thresholds are derived from the model's context window. Models the
harness does not recognize fall back to a conservative 128k window; set
`ARC3_CODEX_MODEL_CONTEXT_WINDOW` (tokens) to override, for example for
`gpt-6-astra`. The override is per model: leaving it exported while running a
smaller model (e.g. 1,000,000 with `gpt-5.6-terra`, a 272k model) makes Codex
compact on its own limit before the harness hook can fire, and the run stops at
the first token report with `CodexContextWindowMismatch` instead of dying at
the first compaction.

### Claude Code

Claude runs can bill either an Anthropic API key or a Claude subscription:

- **API key (platform billing).** Put it in `.env`:

  ```dotenv
  ANTHROPIC_API_KEY=sk-ant-api03-...
  ```

- **Subscription (plan quota).** Generate a token with Claude Code and add it
  to `.env`:

  ```bash
  curl -fsSL https://claude.ai/install.sh | bash -s 2.1.220
  ~/.local/share/claude/versions/2.1.220 setup-token
  ```

  ```dotenv
  CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
  ```

Pick explicitly with `--auth api-key|oauth` on `scripts/run_arc3_claude.py` or
`ARC3_CLAUDE_AUTH=api-key|oauth`; the default (`auto`) uses the API key when
`ANTHROPIC_API_KEY` is set and the subscription token otherwise. The container
only ever receives the one variable for the chosen mode, and `manifest.json`
records `claude_auth_mode`. Batch runs (`run_arc3_batch.py`) still require
`CLAUDE_CODE_OAUTH_TOKEN`: their per-account credential pool and rate-limit
rotation are subscription concepts.

### Docker

Build the image for the runtime you will use:

```bash
# Codex CLI
docker build -t arc3-codex-player:0.1 -f Dockerfile.codex-player .

# Claude Code
docker build -t arc3-claude-player:0.1 -f Dockerfile.claude-player .
```

Both images include their pinned runtime (Codex CLI 0.145.0, Claude Code
2.1.220) and build natively for `linux/amd64` and `linux/arm64`, so they work
under Docker Desktop on Apple Silicon. You still need a Claude credential
(`ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`) in `.env` and an OpenAI
credential for Codex, but no host binaries.

## Run

Run one game with Codex:

```bash
.venv/bin/python scripts/run_arc3_codex.py \
  --game-id s5i5 \
  --model gpt-5.6-sol \
  --effort max
```

Run one game with Claude:

```bash
.venv/bin/python scripts/run_arc3_claude.py \
  --game-id s5i5 \
  --model opus \
  --effort xhigh
```

On macOS with Docker Desktop, these two single-game commands are the supported
experimental entrypoints. Batch and competition commands remain
Linux-oriented.

### Stall watchdog (Claude, opt-in)

Dead Anthropic API streams (no output for hours while the game session expires
server-side) were the main cause of lost Claude runs. `--stall-timeout MINUTES`
(default 0 = off, so the raw harness is unchanged) starts Claude Code with
`--include-partial-messages` and kills a segment that emits no stream output for
that long, then resumes the *same* session with a short "continue" prompt so no
context is lost; after three stalls at one game state it falls back to the
usual fresh-thread recovery. 15 minutes is a safe value: genuine thinking
streams something within a few minutes and the longest real think observed was
about 10 minutes.

### No agent notes (`--no-guide-working`, both runtimes)

`--no-guide-working` runs the player without any agent-authored memory, as an
ablation of `GUIDE.md` / `WORKING.md`:

- no `GUIDE.md` or `WORKING.md` files are created, the note paragraph is removed
  from the player instructions, and the `read_guide`, `write_guide`,
  `read_working`, `write_working` (and, for Codex, `save_compact_checkpoint`)
  tools are neither listed nor accepted; only `play`, `inspect`, `read_pixels`,
  and `history` remain;
- context boundaries no longer ask for a note checkpoint. When Claude reaches
  the visual context limit or compacts natively, or when Codex's pre-compact
  hook fires, the harness starts a fresh player thread from the environment
  record alone (current observation, last action result, current-level
  history, and the current frame), the same handoff already used after a
  runtime failure;
- Codex's RESET still starts a fresh session, but `play` takes no `retry_state`
  note, so the fresh session likewise receives only the environment record.

The manifest records `guide_working_enabled: false`. The batch launcher accepts
the same flag and forwards it to every game.

### Self-improving player (`--evolve`, both runtimes)

`--evolve` stacks on the default mode (or on `--no-guide-working`): besides its
notes, the player maintains three kinds of reusable assets. They follow exactly
the `GUIDE.md` / `WORKING.md` model: the player edits them through game tools at
any point of the run (typically right after a level, when the `play` result
carries `level_boundary: true`), every change takes effect immediately, and the
harness never interrupts the turn or opens a separate reflection turn for them.
Everything lives on the host in the player directory (`player/` for Claude,
`codex_visible/` for Codex) and is reached only through host-side tools:

| Asset | Storage | Tools | Contract |
| --- | --- | --- | --- |
| Skills | `skills/<name>.md` (starts empty; at most 32, 16 KB each) | `read_skill`, `write_skill` (`content: null` deletes) | Markdown procedure; the first line is its one-line description. |
| Analysis tools | `tools/<name>.py` + `tools/index.json` (at most 16, 32 KB each) | `write_tool` (`code: null` deletes), `run_tool` | The code defines `analyze(frames, args)`; `frames` is the list of archived 64x64 integer grids (0–15) of one turn, last grid = final visual; returns JSON. |
| Hooks | `hooks.json` (at most 20 rules) | `write_hooks` (replaces the whole file) | Declarative `before_play` / `after_play` rules, see below. |

Names match `^[a-z0-9][a-z0-9_-]{0,39}$`. The current index (skill names and
descriptions, tool names and descriptions, active hook rules) is appended to the
system prompt of every player segment, so a fresh thread always knows what
exists; skill bodies are read on demand (`read_skill`), following the
progressive-disclosure pattern of Claude Code skills.

`run_tool(name, turn=None, args={})` executes a saved tool on the archived grids
of one turn (the current turn by default) inside a throwaway container
(`docker run --rm --network none --read-only --cap-drop ALL --pids-limit 64
--memory 512m --cpus 1`, reusing the `arc3-codex-player:0.1` or
`arc3-claude-player:0.1` image, pure Python standard library, 10 s limit,
output truncated to 8 KB). `write_tool` loads the code once in the same
sandbox and returns any import error to the model. Running a tool never counts
as a game action.

A hook rule is `{"name", "event": "before_play" | "after_play", "when": {...},
"note": "text", "block": false}`; all conditions in `when` must hold:
`action_in` (action names), `click_in_region` `{x0,y0,x1,y1}` in 64x64 grid
coordinates, `frame_unchanged_steps_at_least` (consecutive actions whose final
visual did not change), `same_action_repeated_at_least` (consecutive identical
actions, counting the one being played) and `level_steps_at_least`. A matching
rule appends `[name] note` to the `hook_notes` of the `play` result; a
`before_play` rule with `"block": true` returns `hook_blocked` without touching
the game (and without counting as an invalid action). Repeating the very same
action immediately after a block overrides it (`hook_overridden`), so a bad
rule can never dead-lock the player.

When changes become visible: hooks are re-read from `hooks.json` on every
`play`, so a rule written mid-level applies to the very next action; `run_tool`
reads `tools/<name>.py` at call time and starts a throwaway container, so there
is no stale process to restart; skill bodies are read on demand. Only the index
in the system prompt is rendered per segment, so a skill created mid-level is
listed there from the next thread on (the model that wrote it already has it in
context). Level completion itself is unchanged from the default mode: the
`play` result carries `level_boundary: true` and the turn simply continues.

Bookkeeping: every skill/tool/hook write, `run_tool` execution and hook match,
block or override is appended to `private/evolve_log.jsonl`; the manifest
records `evolve_enabled: true` and `evolve_sandbox_image`. Combined with
`--no-guide-working`, only skills, tools and hooks remain. The batch launcher
accepts `--evolve` and forwards it to every game.

```bash
.venv/bin/python scripts/run_arc3_claude.py --game-id s5i5 --model opus --effort xhigh --evolve
.venv/bin/python scripts/run_arc3_codex.py --game-id s5i5 --model gpt-5.6-sol --effort max --evolve
```

Run all games:

```bash
./scripts/run_batch.sh --runtime codex --mode online \
  --model gpt-5.6-sol --effort max -j 2
./scripts/run_batch.sh --runtime claude --mode online \
  --model opus --effort xhigh -j 2
```

Available modes are `online` and `competition`. `offline` is also available when
local game files are supplied through `ENVIRONMENTS_DIR`.

## Citation

```bibtex
@misc{han2026vista,
  title  = {{VISTA}: A Visual Harness for Reasoning in an Interactive World},
  author = {Han, Qiushi and Hu, Keya and Qiu, Linlu and Wu, Cathy and He, Kaiming},
  year   = {2026},
  month  = aug,
  note   = {Blog post},
  url    = {https://vista-research.github.io/}
}
```
