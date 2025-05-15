#!/bin/bash
#
#SBATCH -p kempner # partition (queue)
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 50 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH --gres=gpu:1 # gpu
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_learn.%j.out # STDOUT
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_learn.%j.err # STDERR

# load your bash config and (re)activate the conda env
source ~/.bashrc
conda deactivate || true
conda activate esm_env

# change if needed
cd "/n/holylfs06/LABS/mzitnik_lab/Users/${USER}/foldingdiff"

case "$1" in
  1)
    python bin/learn.py --data-dir $2 --cuda cuda --epochs 1000 --num-workers 50 --auto --toy $3 --pad $4 --gamma $5 --model "feats" --max-seg-len 10000000000 --debug
    ;;
  *)
    python bin/learn.py --data-dir $2 --cuda cuda --epochs 1000 --num-workers 50 --auto --toy $3 --pad $4 --gamma $5 --model "feats" --edge --max-seg-len 20
esac
