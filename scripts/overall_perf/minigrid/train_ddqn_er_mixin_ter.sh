ENV="MiniGrid-$1-v0"
SEED=${2-"0"}
GPU=${3-"-1"}
OUTDIR_LABEL="er-mixin-0.5-ter"
EXTRA_ARGS=${@:4}

echo "Env: $ENV"
echo "Outdir label: ${OUTDIR_LABEL}"
echo "Seed: $SEED"
echo "GPU: $GPU"

python scripts/train_dqn.py \
    --gpu $GPU \
    --algo DDQN \
    --replay TER \
    --outdir results/minigrid/ddqn-${OUTDIR_LABEL}/large_atari/$ENV \
    --env $ENV \
    --monitor \
    --config dqn_configs/minigrid.yaml \
    --rs-config-dir rs_configs \
    --rs-gin-files rs.gin \
    --rs-gin-files gameover_rs.gin \
    --rs-gin-files er_mixin_0.5_rs.gin \
    --rs-gin-files maxp_1_rs.gin \
    $EXTRA_ARGS
