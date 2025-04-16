#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 10 # number of cores
#SBATCH --mem 50g # memory pool for all cores
#SBATCH -t 12:00:00 # time
#SBATCH -o slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm/slurm.%N.%j.err # STDERR

cd
. .bashrc
conda deactivate
conda activate foldingdiff
cd /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ # /path/to/directory
python bin/encode.py --toy $1 --pad $2 --vis --data-dir homo --auto --sec $3