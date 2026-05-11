#!/bin/bash -l
#SBATCH -o std_out_train
#SBATCH -e std_err_train
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# Train all six neural stance detection models on the unified corpus.
# Run from the repo root so relative paths resolve correctly.
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

source activate llm_stance
cd "${REPO_ROOT}"
python scripts/train.py
