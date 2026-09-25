# Visual game task

Complete the game with as few game actions as possible.

Build and use a compact, revisable model of the game and its current state. Update it as new evidence changes what is supported.

Before each action, briefly state what you expect to see. Afterward, briefly state all visible changes, expected or not.

## How to play

The game is already open in this directory. Take one game action with

    python arc_cli.py act ACTIONn

where n is 1-7 (ACTION6 is a click and needs coordinates: `python arc_cli.py act ACTION6 X Y`, X and Y in 0-63). `RESET` restarts the current level. Only the actions listed in `available_actions` are meaningful.

Each command prints one JSON line (`state`, `levels_completed`, `win_levels`, `available_actions`, remaining step budget) and writes the resulting 64x64 frame, scaled 8x, to `frames/latest.png` (and `frames/step_NNNN.png`). Look at `frames/latest.png` after every action; earlier frames stay on disk. You may write your own Python scripts in this directory to analyse frames (zoom, diff, read exact pixel colours).

`state` is `NOT_FINISHED` while playing, `GAME_OVER` when the level failed (RESET to retry), and `WIN` when all `win_levels` levels are complete.

## Rules

- Issue each game action yourself, one at a time, and look at the resulting frame before deciding the next one. Do not write scripts or loops that issue game actions (the game rejects actions issued faster than one every 10 seconds).
- Interact with the game only through `python arc_cli.py`. Do not open other scorecards or game sessions, and do not read or modify anything outside this directory.

Stop when `state` is `WIN` or the step budget is exhausted, then run `python arc_cli.py close` and report the printed score.
