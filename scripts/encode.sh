#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 10 # number of cores
#SBATCH --mem 50g # memory pool for all cores
#SBATCH -t 0-12:00 # time
#SBATCH -o slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm/slurm.%N.%j.err # STDERR

# load your bash config and (re)activate the conda env
source ~/.bashrc
conda deactivate || true
conda activate foldingdiff

# change if needed
cd "/n/holylfs06/LABS/mzitnik_lab/Users/${USER}/foldingdiff"

toy=$1
pad=$2
bins=$3
bin_strat=$4
sec=$5
res_init=$6
ckpt_dir=$7

# base command
cmd=( python bin/encode.py
      --toy     "$toy"
      --pad     "$pad"
      --bins    "$bins"
      --bin-strategy     "$bin_strat"
      --vis
      --sec     "$sec"
      --res-init "$res_init"
      --data-dir homo
      --auto
)

# only append --ckpt-dir if $5 is non‑empty
if [ -n "$ckpt_dir" ]; then
  cmd+=( --ckpt-dir "$ckpt_dir" )
fi

"${cmd[@]}"