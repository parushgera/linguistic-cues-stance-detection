#!/bin/bash -l
#SBATCH -o std_out_rnn_covid_target
#SBATCH -e std_err_rnn_covid_target
#SBATCH -p Contributors
#SBATCH -w GPU45
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate stance

python /home/p/parush/style_markers/custom_target_fine_tune.py


