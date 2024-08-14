#!/usr/bin/env bash

# Parameters
DOCKER_IMG=spapais/streampetr:latest

# Host paths
HOME_DIR=/home/trail/workspace
PROJ_DIR=$HOME_DIR/StreamPETR-JDMP
DATA_DIR=/data/sets/nuscenes
OUTPUT_DIR=$PROJ_DIR/output

VOLUMES="-v $PROJ_DIR/:/proj/
-v $DATA_DIR/samples:/proj/data/nuscenes/samples
-v $DATA_DIR/v1.0-mini:/proj/data/nuscenes/v1.0-mini
-v $DATA_DIR/v1.0-trainval:/proj/data/nuscenes/v1.0-trainval
-v $DATA_DIR/v1.0-test:/proj/data/nuscenes/v1.0-test
-v $DATA_DIR/lidarseg:/proj/data/nuscenes/lidarseg
-v $DATA_DIR/maps:/proj/data/nuscenes/maps
-v $DATA_DIR/sweeps:/proj/data/nuscenes/sweeps
-v $OUTPUT_DIR:/proj/output/"

BASE_CMD="python tools/create_data_nusc.py --root-path /proj/data/nuscenes --extra-tag nuscenes2d --version v1.0"
# BASE_CMD="python tools/create_data_nusc.py --root-path /proj/data/nuscenes --extra-tag nuscenes2d_mini --version v1.0-mini"

CONTAINER_CMD="docker run -it --ipc host -w /proj/
$VOLUMES
$DOCKER_IMG
$BASE_CMD"

echo "$CONTAINER_CMD"
eval $CONTAINER_CMD