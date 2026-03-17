#!/bin/bash
#SBATCH --job-name=cluster_perm
#SBATCH --array=0-12               # batch size 10 * 13 array jobs = 130 channels
#SBATCH --cpus-per-task=8          # for n_jobs=-1 within each task
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/milgram/pi/turk-browne/projects/NSD_iEEG_share/slurms/%x_%A_%a.out
#SBATCH --error=/gpfs/milgram/pi/turk-browne/projects/NSD_iEEG_share/slurms/%x_%A_%a.err

cd /gpfs/milgram/pi/turk-browne/projects/NSD_iEEG_share
mkdir -p slurms
source .venv/bin/activate
cd scripts

jupyter nbconvert --to script data_inspection-power.ipynb && \
    python data_inspection-power.py $SLURM_ARRAY_TASK_ID 10