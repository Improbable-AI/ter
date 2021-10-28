# Installation
1. `git clone https://github.com/williamd4112/topological_experience_replay.git`
2. `cd topological_experience_replay`
1. `conda env create -f environment.yml`
2. `pip install -e .`

# Outline
```
- backtracking: (algorithms' implementation)
  | - agent_config: (network architecture definitions)
  | - agents: (agents' implementations)
  | - common: (utility functions)
  | - envs: (environment wrappers)
  | - experiments: (helper functions for experiments)
  | - replay_buffers: (TER implementations)
- plotting: (plotting scripts)
- scripts: (shell scripts for reproducing experiments)
- rs_configs: (TER configuration files, usually feed them as arguments in the scripts, see `scripts`)
- dqn_configs: (hyperparameters configuration files)
- examples: (toy examples for the graph structure)
```

# Reproduce experiments
## Overall performance in MiniGrid/Sokoban (Section 5.3)
- `cd topological_experience_replay`
- Run the following scripts with `seed` and `gpu_id` in the arguments. The results will be stored at `results/overall_perf`.
- MiniGrid
```
scripts/overall_perf/minigrid/simplecrossing_easy/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/simplecrossing_easy/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/simplecrossing_easy/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/simplecrossing_easy/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/simplecrossing_easy/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/simplecrossing_hard/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/simplecrossing_hard/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/simplecrossing_hard/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/simplecrossing_hard/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/simplecrossing_hard/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/lavacrossing_easy/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_easy/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_easy/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_easy/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_easy/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/lavacrossing_hard/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/doorkey/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/doorkey/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/doorkey/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/doorkey/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/doorkey/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/unlock/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/redbluedoor/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/redbluedoor/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/redbluedoor/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/redbluedoor/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/redbluedoor/train_er_mixin_ter.sh {seed} {gpu_id}
```
- Sokoban
```
scripts/overall_perf/sokoban/5x5_1/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/5x5_1/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/5x5_1/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/5x5_1/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/5x5_1/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/5x5_2/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/5x5_2/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/5x5_2/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/5x5_2/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/5x5_2/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/6x6_1/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_1/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_1/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_1/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_1/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/6x6_2/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/6x6_3/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/7x7_1/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/7x7_1/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/7x7_1/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/7x7_1/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/7x7_1/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/7x7_2/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/7x7_2/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/7x7_2/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/7x7_2/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/7x7_2/train_er_mixin_ter.sh {seed} {gpu_id}
```

## Replay ratio experiments (Section 5.4)
- `cd topological_experience_replay`
- Run the following scripts with `seed` and `gpu_id` in the arguments. `1.0` and `2.0` in the directory names denote replay ratios. The results will be stored at `results/replay_ratio`.
- MiniGrid
```
scripts/overall_perf/minigrid/lavacrossing_hard_1.0/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard_1.0/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard_1.0/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard_1.0/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard_1.0/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/unlock_1.0/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock_1.0/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock_1.0/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock_1.0/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock_1.0/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/lavacrossing_hard_2.0/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard_2.0/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard_2.0/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard_2.0/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/lavacrossing_hard_2.0/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/minigrid/unlock_2.0/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock_2.0/train_per.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock_2.0/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock_2.0/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/minigrid/unlock_2.0/train_er_mixin_ter.sh {seed} {gpu_id}
```
- Sokoban
```
scripts/overall_perf/sokoban/6x6_2_1.0/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2_1.0/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2_1.0/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2_1.0/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2_1.0/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/6x6_3_1.0/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3_1.0/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3_1.0/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3_1.0/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3_1.0/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/6x6_2_2.0/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2_2.0/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2_2.0/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2_2.0/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_2_2.0/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/overall_perf/sokoban/6x6_3_2.0/train_uer.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3_2.0/train_per.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3_2.0/train_ebu.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3_2.0/train_discor.sh {seed} {gpu_id}
scripts/overall_perf/sokoban/6x6_3_2.0/train_er_mixin_ter.sh {seed} {gpu_id}
```

# Run in docker
- `docker build -t ter .`
- `./dockerun ${COMMAND}`

# Develop with graph memory
See `examples/graph/hash_graph.py`