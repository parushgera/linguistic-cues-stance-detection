#!/bin/bash -l
#SBATCH -o std_out_evaluation
#SBATCH -e std_err_evaluation
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate style

python /home/p/parush/style_markers/evaluation.py


