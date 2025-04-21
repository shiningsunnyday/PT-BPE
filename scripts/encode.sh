#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 10 # number of cores
#SBATCH --mem 50g # memory pool for all cores
#SBATCH -t 0-12:00 # time
#SBATCH -o slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm/slurm.%N.%j.err # STDERR

# load your bash config and (re)activate the conda env
# source ~/.bashrc
# conda deactivate || true
# conda activate /n/netscratch/mzitnik_lab/Lab/afang/envs/ptbpe

# change if needed
cd "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE"

toy=$1
pad="512"
bins="1-10"
bin_strat="histogram"
sec="False"
res_init="residue"
# ckpt_dir=$7

# base command
PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE \
  /n/netscratch/mzitnik_lab/Lab/afang/envs/ptbpe/bin/python -m bin.encode \
  --auto \
  --bin-strategy histogram \
  --bins 1-20 \
  --data-dir homo \
  --log-dir /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/logs \
  --pad 512 \
  --save-dir /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/plots \
  --toy 1000000 \
  --vis