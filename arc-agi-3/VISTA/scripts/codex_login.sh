#!/usr/bin/env bash
# Log in to Codex using the pinned runtime baked into the player image.
#
# This writes ~/.codex/auth.json on the host (the file the harness mounts
# read-only into every player container), so no host Codex install is needed.
#
# Usage:
#   scripts/codex_login.sh                 # ChatGPT device-code login
#   scripts/codex_login.sh --with-api-key  # read OPENAI_API_KEY from stdin
#   scripts/codex_login.sh status
set -euo pipefail

IMAGE="${ARC3_CODEX_IMAGE:-arc3-codex-player:0.1}"
CODEX_DIR="${ARC3_CODEX_HOME:-$HOME/.codex}"

mkdir -p "$CODEX_DIR"
chmod 700 "$CODEX_DIR"

if [ "$#" -eq 0 ]; then
  set -- --device-auth
fi

tty_flags=(-i)
if [ -t 0 ]; then
  tty_flags=(-it)
fi

exec docker run "${tty_flags[@]}" --rm --pull=never \
  --user "$(id -u):$(id -g)" \
  -e CODEX_HOME=/codex-home \
  -e HOME=/home/codex \
  --tmpfs /home/codex:rw,nosuid,nodev,size=16m \
  -v "$CODEX_DIR:/codex-home:rw" \
  "$IMAGE" /opt/codex/bin/codex login "$@"
