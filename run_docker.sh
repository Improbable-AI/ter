docker run -it --mount type=bind,Source=$(pwd)/,Target=/code,bind-propagation=shared ter bash scripts/minigrid/train_ddqn_er_mixin_ter.sh Empty-5x5
