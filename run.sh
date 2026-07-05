# Harness Scaling
uv run python evolve.py --config configs/our-experiments/exp-per-task-claude-opus46-local-iter5-c10.yaml --experiment exp-per-task-claude-opus46-local-iter5-c10

# Sequential Refinement
uv run python evolve_seq.py --config configs/our-experiments/exp-seq-claude-opus46-local-iter5-c10.yaml --experiment exp-seq-claude-opus46-local-iter5-c10-test

# Parallel Sampling
uv run python run_code_agent_baseline.py --config configs/our-experiments/parallel-exp-claude-opus46-k5.yaml --experiment parallel-exp-claude-opus46-k5

uv run python run_blind_rollout_selector.py --experiment-dir experiments/parallel-exp-claude-opus46-k5 --provider claude --name claude-reasoning-high-select-full

uv run python audit_blind_rollout_selection.py experiments/parallel-exp-claude-opus46-k5/blind_rollout_selector/claude-reasoning-high-select-full
