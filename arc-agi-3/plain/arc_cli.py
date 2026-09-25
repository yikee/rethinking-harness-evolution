#!/usr/bin/env python
"""Minimal ARC-AGI-3 command line for a plain coding agent (no VISTA harness).

    python arc_cli.py act ACTIONn [x y]   take one action (ACTION6 needs x y, 0-63)
    python arc_cli.py status              print the current session state
    python arc_cli.py close               close the scorecard and print the score

The game is opened by the run supervisor. Every action writes
frames/step_NNNN.png and frames/latest.png and prints one JSON line with
state, levels_completed, win_levels and available_actions.
"""

import json
import os
import sys
from pathlib import Path

import requests
from PIL import Image
from arc_agi.rendering import COLOR_MAP  # the official 16-colour palette

PROXY = os.environ["ARC_PROXY_URL"].rstrip("/")
FRAMES = Path("frames")
ACTIONS = {"RESET", *(f"ACTION{i}" for i in range(1, 8))}


def call(method: str, path: str, payload: dict | None = None) -> dict:
    r = requests.request(method, f"{PROXY}{path}", json=payload, timeout=120)
    try:
        body = r.json()
    except ValueError:
        body = {"error": r.text}
    if r.status_code >= 400:
        sys.exit(f"error {r.status_code}: {body.get('error', body)}")
    return body


def rgb(value: int) -> tuple[int, int, int]:
    colour = COLOR_MAP[value]
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))


def render(frame: list, path: Path, scale: int = 8) -> None:
    grid = frame[-1]  # the last animation layer is the settled frame
    img = Image.new("RGB", (len(grid[0]), len(grid)))
    px = img.load()
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            px[x, y] = rgb(v)
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(path)


def report(data: dict) -> None:
    FRAMES.mkdir(exist_ok=True)
    png = FRAMES / f"step_{data['step']:04d}.png"
    render(data["frame"], png)
    render(data["frame"], FRAMES / "latest.png")
    print(
        json.dumps(
            {
                "step": data["step"],
                "steps_remaining": data["steps_remaining"],
                "state": data["state"],
                "levels_completed": data["levels_completed"],
                "win_levels": data["win_levels"],
                "available_actions": data["available_actions"],
                "frame_png": str(png),
            }
        )
    )


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    cmd, *args = argv
    if cmd == "act":
        if not args or args[0] not in ACTIONS:
            sys.exit(f"usage: act {{{'|'.join(sorted(ACTIONS))}}} [x y]")
        payload = {"action": args[0]}
        if args[0] == "ACTION6":
            if len(args) != 3:
                sys.exit("ACTION6 needs x y (0-63)")
            payload.update(x=int(args[1]), y=int(args[2]))
        report(call("POST", "/act", payload))
    elif cmd == "status":
        print(json.dumps(call("GET", "/status"), indent=1))
    elif cmd == "close":
        result = call("POST", "/close")
        print(json.dumps({"score": result.get("score"), "card_id": result.get("card_id")}))
        print(f"https://arcprize.org/scorecards/{result.get('card_id')}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
