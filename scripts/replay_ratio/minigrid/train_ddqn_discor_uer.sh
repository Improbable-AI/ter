ENV="MiniGrid-$1-v0"
SEED=${2-"0"}
GPU=${3-"-1"}
NTIMESUPDATE=${4-1}
EXTRA_ARGS=${@:5}

echo "Env: $ENV"
echo "Outdir label: ddqn-discor-uer"
echo "Seed: $SEED"
echo "GPU: $GPU"

python scripts/train_dqn.py \
    --gpu $GPU \
    --algo DISCOR-DDQN \
    --replay UER \
    --outdir results/replay_ratio/minigrid/ddqn-discor-uer/large_atari/$ENV \
    --env $ENV \
    --monitor \
    --config dqn_configs/minigrid.yaml \
    --n-times-update ${NTIMESUPDATE} \
    $EXTRA_ARGS
