ENV="MiniGrid-$1-v0"
SEED=${2-"0"}
GPU=${3-"-1"}
EXTRA_ARGS=${@:4}

echo "Env: $ENV"
echo "Outdir label: per"
echo "Seed: $SEED"
echo "GPU: $GPU"

python scripts/train_dqn.py \
    --gpu $GPU \
    --algo DDQN \
    --replay PER \
    --outdir results/minigrid/ddqn-per/large_atari/$ENV \
    --env $ENV \
    --monitor \
    --config dqn_configs/minigrid.yaml \
    $EXTRA_ARGS
