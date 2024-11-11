#!/usr/bin/env bash

# Parameters
GPUS=0
NUM_GPUS=1
CONFIG_NAME=jdmp_mini_attforecast_noprop_bs8_1gpu_freezefinetune_60e
DOCKER_IMG=spapais/streampetr:latest

# Host paths
HOME_DIR=/home/trail/workspace
PROJ_DIR=$HOME_DIR/StreamPETR-JDMP
DATA_DIR=/data/sets/nuscenes
OUTPUT_DIR=$PROJ_DIR/output

# Container paths
CONFIG_DIR=/proj/projects/configs/StreamPETR
CONFIG_FILE=$CONFIG_DIR/$CONFIG_NAME.py
MODEL_CKPT=/proj/output/$CONFIG_NAME/$CONFIG_NAME.pth
WRK_DIR=/proj/output/$CONFIG_NAME
CFG_OPT=model.pts_bbox_head.viz_forecast_loss=True

VOLUMES="-v $PROJ_DIR/:/proj/
-v $DATA_DIR/samples:/proj/data/nuscenes/samples
-v $DATA_DIR/v1.0-mini:/proj/data/nuscenes/v1.0-mini
-v $DATA_DIR/v1.0-trainval:/proj/data/nuscenes/v1.0-trainval
-v $DATA_DIR/v1.0-test:/proj/data/nuscenes/v1.0-test
-v $DATA_DIR/lidarseg:/proj/data/nuscenes/lidarseg
-v $DATA_DIR/maps:/proj/data/nuscenes/maps
-v $DATA_DIR/sweeps:/proj/data/nuscenes/sweeps
-v $OUTPUT_DIR:/proj/output/"

BASE_CMD="tools/dist_train.sh $CONFIG_FILE $NUM_GPUS --work-dir $WRK_DIR --cfg-options '$CFG_OPT'"

CONTAINER_CMD="docker run -it --ipc host --gpus $GPUS -w /proj/
--env="WANDB_API_KEY=$WANDB_API_KEY"
$VOLUMES
$DOCKER_IMG
$BASE_CMD"
# --env="TORCH_DISTRIBUTED_DEBUG=DETAIL"

echo "$CONTAINER_CMD"
eval $CONTAINER_CMD