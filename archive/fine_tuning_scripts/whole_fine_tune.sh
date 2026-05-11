#!/bin/bash -l
#SBATCH -o std_out_rnn_whole_KimCNN
#SBATCH -e std_err_rnn_whole_KimCNN
#SBATCH -p nopreempt
#SBATCH -w GPU2
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate stance

python /home/p/parush/style_markers/whole_fine_tune.py


