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
  export SLURM_CPUS_PER_TASK=0
  runner="${runner} -m pdb -c continue"
fi

toy=$2
num_ref=$3
num_vis=$4
data_dir=$5
pad="512"
bins=$6
bin_strat=$7
sec="False"
vis="True"
res_init="True"
save_every=$8
plot_every=$9
rmsd_only=${10}
p_size=${11}
num_p=${12}
max_num_strucs=${13}
glue_opt=${14}
glue_opt_method=${15}
free_bonds=${16}
rmsd_super_res=${17}

if [ -n "${18}" ]; then
  extra="--save-dir ${18}"
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
  --free-bonds $free_bonds \
  --sec $sec \
  --vis $vis \
  --data-dir $data_dir \
  --save-every $save_every \
  --plot-every $plot_every \
  --log-dir /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/logs \
  --pad $pad \
  --toy $toy \
  --p-min-size $p_size \
  --rmsd-only $rmsd_only \
  --rmsd-super-res $rmsd_super_res \
  --num-p $num_p \
  --num-vis $num_vis \
  --num-ref $num_ref \
  --max-num-strucs $max_num_strucs \
  --glue-opt $glue_opt \
  --glue-opt-method $glue_opt_method \
  $extra
