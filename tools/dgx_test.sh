#!/bin/bash
#SBATCH --job-name=test_jdmpvov_baseline_bs8_2gpu    # Job name
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

# Host paths
DATA_DIR=/raid/datasets/nuscenes
HOME_DIR=/raid/home/spapais
PROJ_DIR=$HOME_DIR/StreamPETR
OUT_DIR=$HOME_DIR/output

# Container paths
VOLUMES="--bind=$PROJ_DIR:/proj
         --bind=$DATA_DIR:/proj/data/nuscenes
         --bind=$OUT_DIR:/proj/output
        "
CFG_FILE=projects/configs/StreamPETR/$CFG_NAME.py
CKPT_FILE=ckpts/$CFG_NAME.pth
WRK_DIR=output/test_$CFG_NAME/
EVAL_OPT=jsonfile_prefix=$WRK_DIR

# Command
# BASE_CMD="python tools/test.py $CFG_FILE $CKPT_FILE --eval bbox"
BASE_CMD="./tools/dist_test.sh $CFG_FILE $CKPT_FILE $NUM_GPUS --eval bbox --eval-options '$EVAL_OPT'"
# BASE_CMD="bash"
CONTAINER_CMD="singularity exec --nv -c -e --pwd /proj/ \
$VOLUMES \
$SING_IMG \
$BASE_CMD
"

# Run
echo "Running eval"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
echo "Done eval"