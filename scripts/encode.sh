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

runner="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/foldingdiff/bin/python"

if [ $1 -eq 1 ]; then
  echo "debug"
  runner="${runner} -m pdb -c continue"
fi

toy=$2
data_dir=$3
pad="512"
bins=$4
bin_strat=$5
sec="False"
vis="True"
res_init="True"
save_every=$6
plot_every=$7
p_size=$8
num_p=$9
max_num_strucs=${10}
glue_opt=${11}
glue_opt_method=${12}

if [ -n "${13}" ]; then
  extra="--save-dir ${13}"
else
  extra="--auto"
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff \
  $runner bin/encode.py \
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
  --max-num-strucs $max_num_strucs \
  --glue-opt $glue_opt \
  --glue-opt-method $glue_opt_method \
  $extra
