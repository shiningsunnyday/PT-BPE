#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 10 # number of cores
#SBATCH --mem 50g # memory pool for all cores
#SBATCH -t 0-12:00 # time
#SBATCH -o /n/netscratch/mzitnik_lab/Lab/afang/job_status/PTBPE_encode.%j.out # STDOUT
#SBATCH -e /n/netscratch/mzitnik_lab/Lab/afang/job_status/PTBPE_encode.%j.err # STDERR

# load your bash config and (re)activate the conda env
# source ~/.bashrc
# conda deactivate || true
# conda activate /n/netscratch/mzitnik_lab/Lab/afang/envs/ptbpe

# change if needed
cd "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE"

# module load ncf/1.0.0-fasrc01; module load miniconda3/4.12.0-ncf; module load python/3.10.12-fasrc01
# conda env create --prefix /n/netscratch/mzitnik_lab/Lab/afang/envs/ptbpe --file environment.yml

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
  --data-dir repeat \
  --log-dir /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/logs \
  --pad 512 \
  --save-dir /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/plots \
  --toy 1000000 \
  --cache