# Bomberman RL: a linear agent and a deep policy-gradient agent

Final project for the Machine Learning Essentials course. Team: Alexander Huber and Leonardo Reiter.

This repository contains our code base: the course framework, the two agents we developed with their trained parameters, the training code for both, and the evaluation harness that measured every agent the same way. The tournament entry is `agent_code/dynamic_duo`.

## What is in here

| Path | What it is |
| --- | --- |
| `main.py`, `environment.py`, `agents.py`, ... | the course framework (unchanged except for two type annotations in `agents.py`) |
| `agent_code/dynamic_duo/` | **the submitted agent**: a residual CNN policy trained with PPO, CPU inference, no dependency outside the directory besides `torch` and `numpy` |
| `agent_code/lin_agent/` | the linear action-value agent (27 hand-designed features, LSTD in closed form) with its weights and its trainer `train_lin.py` |
| `agent_code/deep_f/` | the deep training toolbox: vectorised `numba` simulator, PPO trainer `train_ppo.py`, probes |
| `configs/deep/` | the arm configurations of the deep training rounds |
| `agent_code/_eval/` | the shared evaluation harness: gauntlet, arena and ladder probes with per-round statistics |
| `agent_code/rule_based_agent`, `coin_collector_agent`, ... | the agents shipped with the framework |

## Additional libraries

`numpy`, `scipy`, `numba`, `torch` (training only; the submitted agent runs on the CPU). The exact versions are in `pyproject.toml` and `requirements.txt`; `uv sync` creates the environment.

## Running things

```bash
# watch the submitted agent against three rule-based agents
uv run python main.py play --agents dynamic_duo rule_based_agent rule_based_agent rule_based_agent

# the gauntlet the report uses: 200 rounds, classic, seed 42, per-round statistics
uv run python -m agent_code._eval gauntlet --agent dynamic_duo --rounds 200 --out-dir results/gauntlet_dd
uv run python -m agent_code._eval gauntlet --agent lin_agent   --rounds 200 --out-dir results/gauntlet_lin

# retrain the linear agent (about two minutes per 12 000 episodes on one core);
# the shipped weights use the 16-feature set `--features graded`
uv run python -m agent_code.lin_agent.train_lin --rule lstd --features conjesc \
    --crate-reward potential --crate-kappa 0.0 --episodes 12000 --out weights.npy

# retrain the deep agent (GPU; 19.5 M environment steps take two to three hours on one card)
BOMBERMAN_DEVICE=cuda:0 uv run python -m agent_code.deep_f.train_ppo \
    --variant-json configs/deep/f_ref.json --total-steps 19513344 --seed 12 --out-dir runs/f_ref
```

The evaluation harness writes one CSV row per round and seat.
