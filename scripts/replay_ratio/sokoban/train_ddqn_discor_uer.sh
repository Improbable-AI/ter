ENV="Sokoban-$1"
SEED=${2-"0"}
GPU=${3-"-1"}
OUTDIR_LABEL="discor-uer"
NET_ARCH="large_atari"
LR="3e-4"
TARGET_UPDATE_INTERVAL="10000"
NTIMESUPDATE=${4-1}
STEPS=${5-3000000}
EXTRA_ARGS=${@:9}
REPLAY_START_SIZE=50000

echo "Env: $ENV"
echo "Outdir label: $OUTDIR_LABEL"
echo "Seed: $SEED"
echo "GPU: $GPU"
echo "Learning rate: $LR"
echo "Net arch: $NET_ARCH"
echo "Steps: $STEPS"
echo "Replay start size: $REPLAY_START_SIZE"
echo "Target update interval: ${TARGET_UPDATE_INTERVAL}"

python scripts/train_dqn.py \
    --gpu $GPU \
    --algo DISCOR-DDQN \
    --replay UER \
    --explorer linear-decay \
    --outdir results/sokoban/ddqn-$OUTDIR_LABEL/$NET_ARCH/$ENV \
    --env $ENV \
    --steps $STEPS \
    --lr $LR \
    --noise-eval 0.01 \
    --monitor \
    --eval-n-runs 100 \
    --eval-interval 100000 \
    --update-interval 4 \
    --n-times-update ${NTIMESUPDATE} \
    --target-update-interval $TARGET_UPDATE_INTERVAL \
    --final-exploration-steps 1000000 \
    --start-epsilon 1.0 \
    --end-epsilon 0.01 \
    --replay-start-size $REPLAY_START_SIZE \
    --batch-size 32 --seed $SEED $EXTRA_ARGS
