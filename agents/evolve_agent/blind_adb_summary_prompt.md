# Blind Task-Local Trajectory Summary

You are analyzing only one task's prior agent trajectory/trajectories.

Task: `{task_name}`
Iteration: `{iteration}`

Important constraints:
- Use only the provided sanitized trace content and the previous summary below.
- Do not use or infer verifier output, reward files, pass/fail labels, hidden test results, or any benchmark-only signal.
- Do not propose code patches or changes to the harness.
- Your output will be shown to a blind per-task evolve agent as trajectory-only context.
- The goal is to help the evolve agent notice repeated confusion, wasted exploration, incomplete actions, and weak investigative habits.

Previous cumulative trajectory summary:

```text
{previous_summary}
```

Return Markdown only. Keep it concise but operationally useful.

Include:
- What the task appears to require from the visible trajectories alone.
- What the agent tried in the current trajectory/trajectories.
- Recurring confusions, slow detours, brittle assumptions, or incomplete actions.
- Concrete reminders or investigation hints for the next evolution step.
- Any uncertainty or evidence gaps.
