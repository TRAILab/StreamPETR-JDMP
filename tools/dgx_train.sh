#!/bin/bash
#SBATCH --job-name=train_jdmpvov_baseline_bs8_2gpu   # Job name
#SBATCH --ntasks=1                    # Run on n CPUs
#SBATCH --mem=100gb                     # Job memory request
#SBATCH --time=7-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=./output/jdmp/%x-%j.log   # Standard output and error log
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=ALL

# Parameters
NUM_GPUS=2
CFG_NAME=jdmpvov_baseline_bs8_2gpu
SING_IMG=/raid/singularity/streampetr.sif
CFG_FOLDER=projects/configs/StreamPETR

# Host paths
DATA_DIR=/raid/datasets/nuscenes
HOME_DIR=/raid/home/spapais
PROJ_DIR=$HOME_DIR/StreamPETR-JDMP
OUT_DIR=$HOME_DIR/output/jdmp

# Container paths
VOLUMES="--bind=$PROJ_DIR:/proj
         --bind=$DATA_DIR:/proj/data/nuscenes
         --bind=$OUT_DIR:/proj/output
        "
CFG_FILE=$CFG_FOLDER/$CFG_NAME.py
WRK_DIR=output/train_$CFG_NAME/

# Command
BASE_CMD="./tools/dist_train.sh $CFG_FILE $NUM_GPUS --work-dir $WRK_DIR"
CONTAINER_CMD="singularity exec --nv -c -e --pwd /proj/ \
--env="WANDB_API_KEY=$WANDB_API_KEY"
$VOLUMES \
$SING_IMG \
$BASE_CMD
"

# Run
# echo "Debug mode: sleep engaged" && sleep 5d
echo "Running train"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
echo "Done train"