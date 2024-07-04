#!/usr/bin/env bash

# RUN_DIR=wandb/latest-run
RUN_DIR=(wandb/offline-*)
SING_IMG=/home/spapais/projects/rrg-swasland/singularity/streampetr.sif
HOST_DIR=/home/spapais/StreamPETR-JDMP/wandb
CONTAINER_DIR=/wandb

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
    RUN_DIR=(wandb/offline-*)
    CONTAINER_CMD="apptainer --silent exec --nv -c -e --pwd /
    --env "WANDB_API_KEY=$WANDB_API_KEY"
    --bind=$HOST_DIR/:$CONTAINER_DIR/
    $SING_IMG
    wandb sync $RUN_DIR"
    date
    echo "running wandb sync $RUN_DIR" 
    eval $CONTAINER_CMD
    echo ""
    sleep 3600
done