ENV="MiniGrid-$1-v0"
SEED=${2-"0"}
GPU=${3-"-1"}
NTIMESUPDATE=${4-1}
EXTRA_ARGS=${@:5}

echo "Env: $ENV"
echo "Outdir label: ebu"
echo "Seed: $SEED"
echo "GPU: $GPU"

python scripts/train_dqn.py \
    --gpu $GPU \
    --algo DDQN \
    --replay EBU \
    --outdir debug_results/minigrid/ddqn-ebu/large_atari/$ENV \
    --env $ENV \
    --monitor \
    --config dqn_configs/minigrid_debug.yaml \
    --n-times-update ${NTIMESUPDATE} \
    $EXTRA_ARGS
