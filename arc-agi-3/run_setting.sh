#!/usr/bin/env bash
# Launch one ARC-AGI-3 run in one of the four settings used in the paper.
#
#   ./run_setting.sh <runtime> <setting> <game-id> [extra args...]
#
#   runtime : claude | codex
#   setting : plain     - plain Claude Code / Codex CLI on the host, no harness
#             nonotes   - VISTA --no-guide-working (no GUIDE.md / WORKING.md)
#             default   - VISTA (GUIDE.md / WORKING.md notes)
#             evolve    - VISTA --evolve (notes + agent-authored skills/tools/hooks)
#   game-id : e.g. ls20, ft09, vc33 (short ids are fine)
#
# Model / effort default to the paper's configuration and can be overridden:
#   MODEL=claude-opus-4-8 EFFORT=xhigh ./run_setting.sh claude evolve ls20
#   MODEL=gpt-5.6-terra   EFFORT=high  ./run_setting.sh codex  plain  ls20
#
# Extra arguments are appended to the underlying command (VISTA settings only),
# e.g. `--max-steps 500` or `--stall-timeout 15`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/VISTA/.venv/bin/python"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[[ $# -ge 3 ]] || usage
runtime="$1"; setting="$2"; game="$3"; shift 3

case "$runtime" in
  claude) MODEL="${MODEL:-claude-opus-4-8}"; EFFORT="${EFFORT:-xhigh}" ;;
  codex)  MODEL="${MODEL:-gpt-5.6-terra}";   EFFORT="${EFFORT:-high}" ;;
  *) echo "unknown runtime '$runtime' (claude|codex)" >&2; usage ;;
esac

[[ -x "$PY" ]] || { echo "missing $PY -- run the Setup steps in arc-agi-3/README.md first" >&2; exit 1; }
[[ -f "$HERE/VISTA/.env" ]] || { echo "missing $HERE/VISTA/.env -- copy VISTA/.env.example and fill in the keys" >&2; exit 1; }

case "$setting" in
  plain)
    [[ $# -eq 0 ]] || { echo "extra args are not supported for the plain setting" >&2; exit 1; }
    exec "$PY" "$HERE/plain/run_plain.py" --agent "$runtime" --model "$MODEL" --effort "$EFFORT" --game-id "$game"
    ;;
  nonotes) flags=(--no-guide-working) ;;
  default) flags=() ;;
  evolve)  flags=(--evolve) ;;
  *) echo "unknown setting '$setting' (plain|nonotes|default|evolve)" >&2; usage ;;
esac

cd "$HERE/VISTA"
exec "$PY" "scripts/run_arc3_${runtime}.py" --game-id "$game" --model "$MODEL" --effort "$EFFORT" ${flags[@]+"${flags[@]}"} "$@"
