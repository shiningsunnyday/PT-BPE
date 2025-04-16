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

cd
. .bashrc
conda deactivate
conda activate esm_env
cd foldingdiff
python bin/predict.py --cuda cuda:0 --pkl-file '/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ckpts/1744613497.675644/bpe_iter=3290.pkl'