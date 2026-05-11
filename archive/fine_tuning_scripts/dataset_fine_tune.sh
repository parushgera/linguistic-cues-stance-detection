#!/bin/bash -l
#SBATCH -o std_out_fine_dataset
#SBATCH -e std_err_fine_dataset
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate stance

python /home/p/parush/style_markers/dataset_fine_tune.py


