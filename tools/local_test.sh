#!/usr/bin/env bash

# Parameters
GPUS=0
NUM_GPUS=1
CONFIG_NAME=jdmpvov_cvforecast_prop_bs8_1gpu_lre6
# CONFIG_NAME=jdmpvov_mini_attforecast_noprop_bs4_1gpu_freezedet_lre4_20e
DOCKER_IMG=spapais/streampetr:latest

# Host paths
HOME_DIR=/home/trail/workspace
PROJ_DIR=$HOME_DIR/StreamPETR-JDMP
DATA_DIR=/data/sets/nuscenes
OUTPUT_DIR=$PROJ_DIR/output

# Container paths
CONFIG_DIR=/proj/projects/configs/StreamPETR
CONFIG_FILE=$CONFIG_DIR/$CONFIG_NAME.py
MODEL_CKPT=/proj/output/$CONFIG_NAME/latest.pth
# MODEL_CKPT=/proj/ckpts/jdmpvov_pretrain_baseline.pth
WRK_DIR=/proj/output/$CONFIG_NAME
EVAL_OPT=jsonfile_prefix=/proj/output/$CONFIG_NAME

VOLUMES="-v $PROJ_DIR/:/proj/
-v $DATA_DIR/samples:/proj/data/nuscenes/samples
-v $DATA_DIR/v1.0-mini:/proj/data/nuscenes/v1.0-mini
-v $DATA_DIR/v1.0-trainval:/proj/data/nuscenes/v1.0-trainval
-v $DATA_DIR/v1.0-test:/proj/data/nuscenes/v1.0-test
-v $DATA_DIR/lidarseg:/proj/data/nuscenes/lidarseg
-v $DATA_DIR/maps:/proj/data/nuscenes/maps
-v $DATA_DIR/sweeps:/proj/data/nuscenes/sweeps
-v $OUTPUT_DIR:/proj/output/"

BASE_CMD="tools/dist_test.sh $CONFIG_FILE $MODEL_CKPT $NUM_GPUS --eval bbox --eval-options '$EVAL_OPT'"

CONTAINER_CMD="docker run -it --ipc host --gpus $GPUS -w /proj/
--env="WANDB_API_KEY=$WANDB_API_KEY"
$VOLUMES
$DOCKER_IMG
$BASE_CMD"

echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
