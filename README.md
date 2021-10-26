# Installation
1. `git clone https://github.com/williamd4112/topological_experience_replay.git`
2. `cd topological_experience_replay`
1. `conda env create -f environment.yml`
2. `pip install -e .`

# Reproduce experiments
## Overall performance in MiniGrid/Sokoban
- `cd topological_experience_replay`
- Run the following scripts with `seed` and `gpu_id` in the arguments.
- MiniGrid
```
scripts/minigrid/simplecrossing_easy/train_uer.sh {seed} {gpu_id}
scripts/minigrid/simplecrossing_easy/train_per.sh {seed} {gpu_id}
scripts/minigrid/simplecrossing_easy/train_ebu.sh {seed} {gpu_id}
scripts/minigrid/simplecrossing_easy/train_discor.sh {seed} {gpu_id}
scripts/minigrid/simplecrossing_easy/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/minigrid/simplecrossing_hard/train_uer.sh {seed} {gpu_id}
scripts/minigrid/simplecrossing_hard/train_per.sh {seed} {gpu_id}
scripts/minigrid/simplecrossing_hard/train_ebu.sh {seed} {gpu_id}
scripts/minigrid/simplecrossing_hard/train_discor.sh {seed} {gpu_id}
scripts/minigrid/simplecrossing_hard/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/minigrid/lavacrossing_easy/train_uer.sh {seed} {gpu_id}
scripts/minigrid/lavacrossing_easy/train_per.sh {seed} {gpu_id}
scripts/minigrid/lavacrossing_easy/train_ebu.sh {seed} {gpu_id}
scripts/minigrid/lavacrossing_easy/train_discor.sh {seed} {gpu_id}
scripts/minigrid/lavacrossing_easy/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/minigrid/lavacrossing_hard/train_uer.sh {seed} {gpu_id}
scripts/minigrid/lavacrossing_hard/train_per.sh {seed} {gpu_id}
scripts/minigrid/lavacrossing_hard/train_ebu.sh {seed} {gpu_id}
scripts/minigrid/lavacrossing_hard/train_discor.sh {seed} {gpu_id}
scripts/minigrid/lavacrossing_hard/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/minigrid/doorkey/train_uer.sh {seed} {gpu_id}
scripts/minigrid/doorkey/train_per.sh {seed} {gpu_id}
scripts/minigrid/doorkey/train_ebu.sh {seed} {gpu_id}
scripts/minigrid/doorkey/train_discor.sh {seed} {gpu_id}
scripts/minigrid/doorkey/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/minigrid/unlock/train_uer.sh {seed} {gpu_id}
scripts/minigrid/unlock/train_per.sh {seed} {gpu_id}
scripts/minigrid/unlock/train_ebu.sh {seed} {gpu_id}
scripts/minigrid/unlock/train_discor.sh {seed} {gpu_id}
scripts/minigrid/unlock/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/minigrid/redbluedoor/train_uer.sh {seed} {gpu_id}
scripts/minigrid/redbluedoor/train_per.sh {seed} {gpu_id}
scripts/minigrid/redbluedoor/train_ebu.sh {seed} {gpu_id}
scripts/minigrid/redbluedoor/train_discor.sh {seed} {gpu_id}
scripts/minigrid/redbluedoor/train_er_mixin_ter.sh {seed} {gpu_id}
```
- Sokoban
```
scripts/sokoban/5x5_1/train_uer.sh {seed} {gpu_id}
scripts/sokoban/5x5_1/train_per.sh {seed} {gpu_id}
scripts/sokoban/5x5_1/train_ebu.sh {seed} {gpu_id}
scripts/sokoban/5x5_1/train_discor.sh {seed} {gpu_id}
scripts/sokoban/5x5_1/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/sokoban/5x5_2/train_uer.sh {seed} {gpu_id}
scripts/sokoban/5x5_2/train_per.sh {seed} {gpu_id}
scripts/sokoban/5x5_2/train_ebu.sh {seed} {gpu_id}
scripts/sokoban/5x5_2/train_discor.sh {seed} {gpu_id}
scripts/sokoban/5x5_2/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/sokoban/6x6_1/train_uer.sh {seed} {gpu_id}
scripts/sokoban/6x6_1/train_per.sh {seed} {gpu_id}
scripts/sokoban/6x6_1/train_ebu.sh {seed} {gpu_id}
scripts/sokoban/6x6_1/train_discor.sh {seed} {gpu_id}
scripts/sokoban/6x6_1/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/sokoban/6x6_2/train_uer.sh {seed} {gpu_id}
scripts/sokoban/6x6_2/train_per.sh {seed} {gpu_id}
scripts/sokoban/6x6_2/train_ebu.sh {seed} {gpu_id}
scripts/sokoban/6x6_2/train_discor.sh {seed} {gpu_id}
scripts/sokoban/6x6_2/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/sokoban/6x6_3/train_uer.sh {seed} {gpu_id}
scripts/sokoban/6x6_3/train_per.sh {seed} {gpu_id}
scripts/sokoban/6x6_3/train_ebu.sh {seed} {gpu_id}
scripts/sokoban/6x6_3/train_discor.sh {seed} {gpu_id}
scripts/sokoban/6x6_3/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/sokoban/7x7_1/train_uer.sh {seed} {gpu_id}
scripts/sokoban/7x7_1/train_per.sh {seed} {gpu_id}
scripts/sokoban/7x7_1/train_ebu.sh {seed} {gpu_id}
scripts/sokoban/7x7_1/train_discor.sh {seed} {gpu_id}
scripts/sokoban/7x7_1/train_er_mixin_ter.sh {seed} {gpu_id}

scripts/sokoban/7x7_2/train_uer.sh {seed} {gpu_id}
scripts/sokoban/7x7_2/train_per.sh {seed} {gpu_id}
scripts/sokoban/7x7_2/train_ebu.sh {seed} {gpu_id}
scripts/sokoban/7x7_2/train_discor.sh {seed} {gpu_id}
scripts/sokoban/7x7_2/train_er_mixin_ter.sh {seed} {gpu_id}

```


# Run in docker
- `docker build -t ter .`
- `./dockerun ${COMMAND}`

# Develop with graph memory
See `examples/graph/hash_graph.py`