#!/bin/bash -l
#SBATCH -o std_out_train_with_wtwt
#SBATCH -e std_err_train_with_wtwt
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate llm_stance

python /home/p/parush/style_markers/train_on_grander_master/train.py


