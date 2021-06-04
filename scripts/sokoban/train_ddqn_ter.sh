ENV="Sokoban-$1"
SEED=${2-"0"}
GPU=${3-"-1"}
OUTDIR_LABEL=${4-"rs"}
NET_ARCH=${5-"large_atari"}
LR=${6-"3e-4"}
TARGET_UPDATE_INTERVAL=${7-"1000"}
STEPS=${8-10000000}
RS_CONFIGS=${@:9} # Addon gin (e.g., wide, rprs, ...)
echo "Env: $ENV"
echo "Outdir label: $OUTDIR_LABEL"
echo "Addon RS Configs: $RS_CONFIGS"
echo "Seed: $SEED"
echo "GPU: $GPU"
echo "Learning rate: $LR"
echo "Net arch: $NET_ARCH"
echo "Steps: $STEPS"

if [[ $NET_ARCH -eq "large_atari" ]]
then    
    if [[ $TARGET_UPDATE_INTERVAL -ge "10000" ]]
    then
        REPLAY_START_SIZE=50000
    else
        REPLAY_START_SIZE=20000 
    fi
else
    REPLAY_START_SIZE=5000
fi
echo "Replay start size: $REPLAY_START_SIZE"
echo "Target update interval: ${TARGET_UPDATE_INTERVAL}"

python scripts/train_dqn.py \
    --gpu $GPU \
    --algo DDQN \
    --replay TER \
    --net-arch $NET_ARCH \
    --explorer linear-decay \
    --outdir results/sokoban/ddqn-$OUTDIR_LABEL-lr_${LR}-tu_${TARGET_UPDATE_INTERVAL}/$NET_ARCH/$ENV \
    --env $ENV \
    --steps $STEPS \
    --lr $LR \
    --noise-eval 0.01 \
    --monitor \
    --eval-n-runs 100 \
    --eval-interval 100000 \
    --update-interval 4 \
    --target-update-interval $TARGET_UPDATE_INTERVAL \
    --final-exploration-steps 1000000 \
    --start-epsilon 1.0 \
    --end-epsilon 0.01 \
    --replay-start-size $REPLAY_START_SIZE \
    --replay-capacity 1000000 \
    --batch-size 32 --seed $SEED \
    --rs-config-dir rs_configs/minigrid \
    --rs-gin-files rs.gin \
    --rs-gin-files terminal_remove_at_sample.gin \
    $RS_CONFIGS
