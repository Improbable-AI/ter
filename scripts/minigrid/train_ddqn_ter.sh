ENV="MiniGrid-$1-v0"
SEED=${2-"0"}
GPU=${3-"-1"}
OUTDIR_LABEL="gameover-rs"
EXTRA_ARGS=${@:4}

echo "Env: $ENV"
echo "Outdir label: ${OUTDIR_LABEL}"
echo "Seed: $SEED"
echo "GPU: $GPU"

python scripts/train_dqn.py \
    --gpu $GPU \
    --algo DDQN \
    --replay RS \
    --outdir results/minigrid/ddqn-${OUTDIR_LABEL}/large_atari/$ENV \
    --env $ENV \
    --monitor \
    --config dqn_configs/minigrid.yaml \
    --rs-config-dir rs_configs \
    --rs-gin-files rs.gin \
    --rs-gin-files gameover_rs.gin \
    $EXTRA_ARGS