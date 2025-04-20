#!/bin/bash
#
#SBATCH -p kempner # partition (queue)
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 10 # number of cores
#SBATCH --mem 500g # memory pool for all cores
#SBATCH --gres=gpu:1 # gpu
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm/slurm.%N.%j.err # STDERR

# load your bash config and (re)activate the conda env
source ~/.bashrc
conda deactivate || true
conda activate esm_env

# change if needed
project_dir="/n/holylfs06/LABS/mzitnik_lab/Users/${USER}/foldingdiff"

cd $project_dir

# train
# python bin/predict.py --cuda cuda:0 --pkl-file ${project_dir}/ckpts/1744875790.3072364/bpe_iter=6000.pkl --pkl-data-file ${project_dir}/homo_datasets_with_test.pkl --auto

# test
python bin/predict.py --cuda cuda:0 --pkl-file ${project_dir}/ckpts/1744875790.3072364/bpe_iter=6000.pkl --pkl-data-file ${project_dir}/homo_datasets_with_test.pkl --test --save-dir ${project_dir}/ckpts/1744932629.834232