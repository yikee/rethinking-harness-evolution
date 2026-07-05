# Task-Local Trajectory Summary

You are analyzing only one task's prior agent trajectory/trajectories.

Task: `{task_name}`
Iteration: `{iteration}`

Important constraints:
- Use only the provided trace content and the previous summary below.
- If external verifier/test results are appended below, treat them as higher-priority evidence than the agent's own self-assessment and use them to explain what actually failed or succeeded.
- Do not propose code patches or modifications to the agent harness.
- Your output will be shown to the same code agent on its next attempt as task-local context.
- The goal is to help the next attempt avoid repeated confusion, wasted exploration, and incomplete actions.

Previous cumulative summary:

```text
{previous_summary}
```

Return Markdown only. Keep it concise but operationally useful.

Include:
- What the task appears to require.
- What the agent tried in the current trajectory/trajectories.
- If verifier/test results are present, what they reveal about the real outcome that the agent should react to next time.
- Recurring confusions, slow detours, brittle assumptions, or incomplete actions.
- Concrete reminders/hints for the next attempt.
- Any uncertainty or evidence gaps.
