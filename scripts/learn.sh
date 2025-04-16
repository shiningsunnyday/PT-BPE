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
conda activate foldingdiff
cd foldingdiff
python bin/learn.py --cuda cuda:0 --epochs 1000 --auto --toy $1 --pad $2 --gamma $3