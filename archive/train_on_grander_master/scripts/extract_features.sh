#!/bin/bash -l
#SBATCH -o std_out_features_meghna
#SBATCH -e std_err_features_meghna
#SBATCH -p SIPEIE23
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate llm_stance

python /home/p/parush/style_markers/train_on_grander_master/extract_features_meghna/extract_features.py


