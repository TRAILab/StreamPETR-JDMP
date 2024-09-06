#!/usr/bin/env bash

# Parameters
GPUS=0
NUM_GPUS=1
CONFIG_NAME=jdmpvov_attforecast_prop_graddetach_qembsep_6lay_attmem_bs8_2gpu
DOCKER_IMG=spapais/streampetr:latest

# Host paths
HOME_DIR=/home/trail/workspace
PROJ_DIR=$HOME_DIR/StreamPETR-JDMP
DATA_DIR=/data/sets/nuscenes
OUTPUT_DIR=$PROJ_DIR/output

# Container paths
CONFIG_DIR=/proj/projects/configs/StreamPETR
CONFIG_FILE=$CONFIG_DIR/$CONFIG_NAME.py
MODEL_CKPT=/proj/output/$CONFIG_NAME/iter_19338.pth
EVAL_OPT=jsonfile_prefix=/proj/output/$CONFIG_NAME/iter_19338

VOLUMES="-v $PROJ_DIR/:/proj/
-v $DATA_DIR/samples:/proj/data/nuscenes/samples
-v $DATA_DIR/v1.0-mini:/proj/data/nuscenes/v1.0-mini
-v $DATA_DIR/v1.0-trainval:/proj/data/nuscenes/v1.0-trainval
-v $DATA_DIR/v1.0-test:/proj/data/nuscenes/v1.0-test
-v $DATA_DIR/lidarseg:/proj/data/nuscenes/lidarseg
-v $DATA_DIR/maps:/proj/data/nuscenes/maps
-v $DATA_DIR/sweeps:/proj/data/nuscenes/sweeps
-v $OUTPUT_DIR:/proj/output/"

# For single checkpoint
BASE_CMD="tools/dist_test.sh $CONFIG_FILE $MODEL_CKPT $NUM_GPUS --eval bbox --eval-options '$EVAL_OPT'"

# For multiple checkpoints
# BASE_CMD="/bin/bash -c '"
# MODEL_CKPTS=(output/$CONFIG_NAME/iter_*.pth)
# echo "MODEL_CKPTS: ${MODEL_CKPTS[@]}"
# for MODEL_CKPT in "${MODEL_CKPTS[@]}"
# do
#     EVAL_OPT=jsonfile_prefix=${MODEL_CKPTS%.*}
#     BASE_CMD="${BASE_CMD} tools/dist_test.sh $CONFIG_FILE $MODEL_CKPT $NUM_GPUS --eval bbox --eval-options '$EVAL_OPT';"
# done
# BASE_CMD="${BASE_CMD}'"

CONTAINER_CMD="docker run -d --ipc host --gpus $GPUS -w /proj/
--env="WANDB_API_KEY=$WANDB_API_KEY"
$VOLUMES
$DOCKER_IMG
$BASE_CMD"

echo "$CONTAINER_CMD"
eval $CONTAINER_CMD