#!/bin/bash -l
#SBATCH -o std_out_rnn_pstance_dataset_kim
#SBATCH -e std_err_rnn_pstance_dataset_kim
#SBATCH -p SIPEIE23
##SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate stance

python /home/p/parush/style_markers/custom_dataset_fine_tune.py


