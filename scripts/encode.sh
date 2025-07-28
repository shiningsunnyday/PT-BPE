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
vis="True"
res_init="true"
save_every=$4
plot_every=$5
p_size=$6
num_p=$7

if [ -n "$8" ]; then
  extra="--save-dir $8"
else
  extra="--auto"
fi

PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff \
  /n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/foldingdiff/bin/python -m bin.encode \
  --bin-strategy $bin_strat \
  --bins $bins \
  --res-init $res_init \
  --sec $sec \
  --vis $vis \
  --data-dir $data_dir \
  --save-every $save_every \
  --plot-every $plot_every \
  --log-dir /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/logs \
  --pad $pad \
  --toy $toy \
  --p-min-size $p_size \
  --num-p $num_p \
  $extra
