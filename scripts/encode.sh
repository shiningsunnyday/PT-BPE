#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 10 # number of cores
#SBATCH --mem 500g # memory pool for all cores
#SBATCH -t 0-12:00 # time
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_encode.%j.out # STDOUT
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_encode.%j.err # STDERR

# load your bash config and (re)activate the conda env
# source ~/.bashrc
# conda deactivate || true
# conda activate /n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/foldingdiff

# change if needed
cd "/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff"

# module load ncf/1.0.0-fasrc01; module load miniconda3/4.12.0-ncf; module load python/3.10.12-fasrc01
# conda env create --prefix /n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/foldingdiff --file environment.yml

toy=$1
data_dir=$2
pad="512"
bins=$3
bin_strat="uniform"
sec="False"
res_init="true"
save_every=1

extra=""

if [ -n "$4" ]; then
  extra="$extra --p-min-size $4"
fi

if [ -n "$5" ]; then
  extra="$extra --num-p $5"
fi

PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff \
  /n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/foldingdiff/bin/python -m bin.encode \
  --auto \
  --bin-strategy $bin_strat \
  --bins $bins \
  --res-init $res_init \
  --sec $sec \
  --data-dir $data_dir \
  --save-every $save_every \
  --log-dir /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/logs \
  --pad $pad \
  --toy $toy \
  --cache \
  $extra