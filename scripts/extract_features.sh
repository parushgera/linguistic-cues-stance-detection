#!/bin/bash -l
#SBATCH -o std_out_features
#SBATCH -e std_err_features
#SBATCH -p SIPEIE23
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# Extract the 43 stylistic features from the test splits.
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

source activate llm_stance
cd "${REPO_ROOT}"
python scripts/extract_features.py
