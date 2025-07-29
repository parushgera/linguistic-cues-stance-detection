#!/bin/bash -l
#SBATCH -o std_out_main
#SBATCH -e std_err_main
#SBATCH -p CiBeR
#SBATCH -w GPU43
#SBATCH --time=7-0
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate stance

python /home/p/parush/style_markers/main.py


