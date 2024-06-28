#!/bin/bash
#SBATCH --job-name=train_stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_4gpu    # Job name
#SBATCH --account=rrg-swasland
#SBATCH --ntasks=1                    # Run on n CPUs
#SBATCH --mem=180gb                     # Job memory request
#SBATCH --time=3-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=./output/streampetr_jdmp/%x-%j.log   # Standard output and error log
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:t4:4
#SBATCH --mail-user="sandro.papais@robotics.utias.utoronto.ca"
#SBATCH --mail-type=ALL

# Parameters
NUM_GPUS=4
CFG_NAME=stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_4gpu
SING_IMG=/home/spapais/projects/rrg-swasland/singularity/streampetr.sif
echo "SLURM_JOB_ID=$SLURM_JOB_ID
CFG_NAME=$CFG_NAME
NUM_GPUS=$NUM_GPUS
"

# Host paths
DATA_DIR=/home/spapais/projects/rrg-swasland/Datasets/nuscenes
HOME_DIR=/home/spapais/projects/rrg-swasland/spapais
TMP_DATA_DIR=$SLURM_TMPDIR/data
# TMP_DATA_DIR=/home/spapais/scratch/temp_data # Slurm unzip alternative
PROJ_DIR=$HOME_DIR/StreamPETR
OUT_DIR=$HOME_DIR/output

# Extract Dataset
echo "Extracting data"
SECONDS=0
for file in $DATA_DIR/*.zip; do
    duration=$SECONDS
    echo "[$((duration/3600))h$((duration%3600/60))m]: Unzipping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
for file in $DATA_DIR/*.pkl; do
    duration=$SECONDS
    echo "[$((duration/3600))h$(((duration%3600)/60))m]: Copying $file to $TMP_DATA_DIR"
    cp $file $TMP_DATA_DIR
done
echo "Done extracting data"

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
BASE_CMD="./tools/dist_test.sh $CFG_FILE $CKPT_FILE $NUM_GPUS --eval bbox --eval-options '$EVAL_OPT'"
CONTAINER_CMD="apptainer exec --nv -c -e --pwd /proj/ \
$VOLUMES \
$SING_IMG \
$BASE_CMD
"

# Run
# echo "Debug mode: sleep engaged" && sleep 5d
module load StdEnv/2020
module load apptainer
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Running eval"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Done eval"