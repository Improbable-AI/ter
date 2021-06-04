# Sokoban
- `train_ddqn_ebu.sh Push_5x5_1 -1 0 large_atari 3e-4`
- `train_ddqn_uer.sh Push_5x5_1 -1 0 large_atari 3e-4`
- `train_ddqn_per.sh Push_5x5_1 -1 0 large_atari 3e-4`
- `train_ddqn_rs.sh Push_5x5_1 -1 0 gameover-rs large_atari 3e-4 --rs-gin-files=gameover_rs.gin`
- - `--rs-gin-files=n16_rs.gin` (n_initial_nodes = 16)
- - `--rs-gin-files=n4_rs.gin` (n_initial_nodes = 4)
- - `--rs-gin-files=maxp_1_rs.gin` (max predecessors = 1)
- - `--rs-gin-files=er_mix_0.5_rs.gin` (mix PER with 0.5 sampling probability)
- - `--rs-gin-files=er_mixin_0.5_rs.gin` (mix half of PER samples)