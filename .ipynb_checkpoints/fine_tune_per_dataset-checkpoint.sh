#!/bin/bash -l
#SBATCH -o std_out_fine_per_pstance_kim
#SBATCH -e std_err_fine_per_pstance_kim
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --time=7-0
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate stance

python /home/p/parush/style_markers/fine_tune_per_dataset.py


