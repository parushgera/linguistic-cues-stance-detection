#!/bin/bash -l
#SBATCH -o std_out_features_wtwt_test
#SBATCH -e std_err_features_wtwt_test
#SBATCH -p CiBeR
#SBATCH -w GPU43
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate stance

python /home/p/parush/style_markers/extract_features.py


