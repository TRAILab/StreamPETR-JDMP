#!/usr/bin/env bash

# RUN_DIR=wandb/latest-run
RUN_DIR=(wandb/offline-*)
SING_IMG=/home/spapais/projects/rrg-swasland/spapais/singularity_images/streampetr.sif
HOST_DIR=/home/spapais/StreamPETR-JDMP/wandb
CONTAINER_DIR=/wandb
OUT_DIR=/home/spapais/output/streampetr_jdmp

CONTAINER_CMD="apptainer --silent exec --nv -c -e --pwd /
--env "WANDB_API_KEY=$WANDB_API_KEY"
--bind=$HOST_DIR/:$CONTAINER_DIR/
$SING_IMG
wandb sync $RUN_DIR"

module load StdEnv/2020
module load apptainer
# echo "$CONTAINER_CMD"

while :
do
    RUN_DIRS=(wandb/offline-*/)
    date
    for RUN_DIR in "${RUN_DIRS[@]}"
    do
        echo "running wandb sync $RUN_DIR"
        CONTAINER_CMD="apptainer --silent exec --nv -c -e --pwd /
        --env "WANDB_API_KEY=$WANDB_API_KEY"
        --bind=$OUT_DIR:/proj/output
        --bind=$HOST_DIR/:$CONTAINER_DIR/
        $SING_IMG
        wandb sync $RUN_DIR"
        eval $CONTAINER_CMD
    done
    echo ""
    sleep 600
done