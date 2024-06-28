# Parameters
GPUS=all
NUM_GPUS=0
CONFIG_NAME=stream_petr_r50_flash_704_bs2_seq_428q_nui_60e
DOCKER_IMG=spapais/streampetr:latest

# Host paths
HOME_DIR=/home/trail/workspace
PROJ_DIR=$HOME_DIR/StreamPETR
DATA_DIR=/data/sets/nuscenes
OUTPUT_DIR=$HOME_DIR/output/streampetr

# Container paths
CONFIG_DIR=/proj/projects/configs/StreamPETR
CONFIG_FILE=$CONFIG_DIR/$CONFIG_NAME.py
MODEL_CKPT=/proj/output/$CONFIG_NAME/$CONFIG_NAME.pth
WRK_DIR=/proj/output/$CONFIG_NAME

VOLUMES="-v $PROJ_DIR/:/proj/
-v $DATA_DIR/samples:/proj/data/nuscenes/samples
-v $DATA_DIR/v1.0-mini:/proj/data/nuscenes/v1.0-mini
-v $DATA_DIR/v1.0-trainval:/proj/data/nuscenes/v1.0-trainval
-v $DATA_DIR/v1.0-test:/proj/data/nuscenes/v1.0-test
-v $DATA_DIR/lidarseg:/proj/data/nuscenes/lidarseg
-v $DATA_DIR/maps:/proj/data/nuscenes/maps
-v $DATA_DIR/sweeps:/proj/data/nuscenes/sweeps
-v $OUTPUT_DIR:/proj/output/"

BASE_CMD="tools/dist_train.sh $CONFIG_FILE $NUM_GPUS --work-dir $WRK_DIR"

CONTAINER_CMD="docker run -it --ipc host --gpus $GPUS -w /proj/
$VOLUMES
$DOCKER_IMG
$BASE_CMD"

echo "$CONTAINER_CMD"
eval $CONTAINER_CMD