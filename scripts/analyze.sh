#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 10 # number of cores
#SBATCH --mem 500g # memory pool for all cores
#SBATCH -t 0-12:00 # time
#SBATCH -o scripts/slurm/PTBPE_analyze.%j.out # STDOUT
#SBATCH -e scripts/slurm/PTBPE_analyze.%j.err # STDERR

python scripts/analyze.py --d $1
