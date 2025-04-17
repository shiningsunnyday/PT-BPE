#!/bin/bash
#
#SBATCH -p kempner # partition (queue)
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 10 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH --gres=gpu:1 # gpu
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm/slurm.%N.%j.err # STDERR

# load your bash config and (re)activate the conda env
source ~/.bashrc
conda deactivate || true
conda activate esm_env

# change if needed
cd "/n/holylfs06/LABS/mzitnik_lab/Users/${USER}/foldingdiff"

python bin/predict.py --cuda cuda:0 --pkl-file '/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ckpts/1744875790.3072364/bpe_iter=6000.pkl' --auto