# Installation
1. `conda env create -f environment.yml`
2. `pip install -e .`

# Minigrid
Available tasks: `Push_5x5_1`, `Push_5x5_2`, `Push_6x6_1`, `Push_6x6_2`, `Push_6x6_3`, `Push_7x7_1`, and `Push_7x7_2`
- `bash scripts/minigrid/train_ddqn_uer.sh {Task: e.g., Push_5x5_1} {GPU_ID: e.g., 0} {Random seed: e.g., 0}`
- `bash scripts/minigrid/train_ddqn_per.sh {Task: e.g., Push_5x5_1} {Random seed: e.g., 0}`
- `bash scripts/minigrid/train_ddqn_ebu.sh {Task: e.g., Push_5x5_1} {Random seed: e.g., 0}`
- `bash scripts/minigrid/train_ddqn_ter.sh {Task: e.g., Push_5x5_1} {Random seed: e.g., 0}`
- `bash scripts/minigrid/train_ddqn_er_mixin_ter.sh {Task: e.g., Push_5x5_1} {Random seed: e.g., 0}`

# Sokoban
Available tasks: `SimpleCrossingS9N1` (SimpleCrossing-Easy), `SimpleCrossingS9N2` (SimpleCrossing-Hard), `LavaCrossingS9N1` (LavaCrossing-Easy), `LavaCrossingS9N2` (LavaCrossing-Hard), `DoorKey-6x6` (DoorKey), `Unlock` (Unlock), and `RedBlueDoors-6x6` (RedBlueDoors)
- `bash scripts/sokoban/train_ddqn_uer.sh {Task: e.g., LavaCrossingS9N1} {GPU_ID: e.g., 0} {Random seed: e.g., 0}`
- `bash scripts/sokoban/train_ddqn_per.sh {Task: e.g., LavaCrossingS9N1} {Random seed: e.g., 0}`
- `bash scripts/sokoban/train_ddqn_ebu.sh {Task: e.g., LavaCrossingS9N1} {Random seed: e.g., 0}`
- `bash scripts/sokoban/train_ddqn_ter.sh {Task: e.g., LavaCrossingS9N1} {Random seed: e.g., 0}`
- `bash scripts/sokoban/train_ddqn_er_mixin_ter.sh {Task: e.g., LavaCrossingS9N1} {Random seed: e.g., 0}`
