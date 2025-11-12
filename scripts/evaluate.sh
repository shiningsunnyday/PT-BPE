#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 3 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o scripts/slurm/PTBPE_evaluate.%j.out # STDOUT
#SBATCH -e scripts/slurm/PTBPE_evaluate.%j.err # STDERR

max_len=50
for ckpt in "$@"; do
    python scripts/evaluate.py --pkl_file $ckpt --max_len $max_len
done
