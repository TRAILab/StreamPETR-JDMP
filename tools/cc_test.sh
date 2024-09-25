#!/bin/bash
#SBATCH --job-name=test_jdmpvov_baseline_bs8_2gpu    # Job name
#SBATCH --account=rrg-swasland
#SBATCH --ntasks=1                    # Run on n CPUs
#SBATCH --mem=120gb                     # Job memory request
#SBATCH --time=2:59:00               # Time limit hrs:min:sec
#SBATCH --output=/home/spapais/output/streampetr_jdmp/%x-%j.log   # Standard output and error log
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:2          # gpu:t4:4 (graham) or gpu:a100:2 (narval)
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=ALL

# Parameters
SERVER=narval
NUM_GPUS=2
CFG_NAME=jdmpvov_baseline_bs8_2gpu

# Host paths
HOME_DIR=/home/spapais
TMP_DATA_DIR=$SLURM_TMPDIR/data
# TMP_DATA_DIR=/home/spapais/scratch/temp_data # Slurm unzip alternative
PROJ_DIR=$HOME_DIR/StreamPETR-JDMP
OUT_DIR=$HOME_DIR/output/streampetr_jdmp
SING_IMG=/home/spapais/projects/rrg-swasland/singularity/streampetr.sif
if [ "$SERVER" = "graham" ]; then
    DATA_DIR=/home/spapais/projects/rrg-swasland/Datasets/nuscenes
    DATA_PKL_DIR=/home/spapais/projects/rrg-swasland/Datasets/nuscenes
fi
if [ "$SERVER" = "narval" ]; then
    DATA_DIR=/home/spapais/projects/rrg-swasland/datasets/nuscenes/
    DATA_PKL_DIR=/home/spapais/datasets/nuscenes/
fi

# Container paths
VOLUMES="--bind=$PROJ_DIR:/proj
         --bind=$TMP_DATA_DIR:/proj/data/nuscenes
         --bind=$OUT_DIR:/proj/output
        "
CFG_FILE=projects/configs/StreamPETR/$CFG_NAME.py
WRK_DIR=output/test_$CFG_NAME/
CKPT_FILE=ckpts/$CFG_NAME.pth
EVAL_OPT=jsonfile_prefix=$WRK_DIR

# Command
WANDB_MODE='offline'
BASE_CMD="./tools/dist_test.sh $CFG_FILE $CKPT_FILE $NUM_GPUS --eval bbox --eval-options '$EVAL_OPT'"
CONTAINER_CMD="apptainer exec --nv -c -e --pwd /proj/ \
$VOLUMES \
$SING_IMG \
$BASE_CMD
"

# Start script
echo "SLURM_JOB_ID=$SLURM_JOB_ID
CFG_NAME=$CFG_NAME
NUM_GPUS=$NUM_GPUS
"

# Extract dataset
echo "Extracting data"
SECONDS=0
for file in $DATA_DIR/*.zip; do
    duration=$SECONDS
    echo "[$((duration/3600))h$((duration%3600/60))m]: Unzipping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
for file in $DATA_PKL_DIR/*.pkl; do
    duration=$SECONDS
    echo "[$((duration/3600))h$(((duration%3600)/60))m]: Copying $file to $TMP_DATA_DIR"
    cp $file $TMP_DATA_DIR
done
echo "Done extracting data"

# Run command
# echo "Debug mode: sleep engaged" && sleep 5d
module load StdEnv/2020
module load apptainer
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Running command"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Done"