#!/bin/bash
#
#SBATCH -p kempner # partition (queue)
##SBATCH -p test
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 16 # number of cores
#SBATCH --mem 200g # memory pool for all cores
#SBATCH --gres=gpu:1 # gpu
#SBATCH -t 3-00:00 # time (D-HH:MM)
##SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_learn.%j.out # STDOUT
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_learn.%j.err # STDERR

# load your bash config and (re)activate the conda env
source ~/.bashrc
conda deactivate || true
conda activate esm_env

# change if needed
cd "/n/holylfs06/LABS/mzitnik_lab/Users/${USER}/foldingdiff"

if [ $1 -eq 1 ]; then
  debug="--debug"
  echo "debug"
else
  debug=""
fi

if [ -n "$7" ]; then
  extra="--save_dir $7"
else
  extra="--auto"
fi


case "$2" in
  1)
    python bin/learn.py --data-dir $3 --config config.json --cuda cuda --epochs 1000 --toy $4 --pad $5 --model "feats" --max-seg-len 10000000000 --l1 $6 $extra $debug;;
  *)
    python bin/learn.py --data-dir $3 --config config.json --cuda cuda --epochs 1000 --toy $4 --pad $5 --model "feats" --edge --max-seg-len 20 --l1 $6 $extra $debug
esac
