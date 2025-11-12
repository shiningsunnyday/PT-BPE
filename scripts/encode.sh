#!/bin/bash
#
#SBATCH -c 20 # number of cores
#SBATCH --mem 500g # memory pool for all cores
#SBATCH -t 0-12:00 # time
#SBATCH -o scripts/slurm/PTBPE_encode.%j.out # STDOUT
#SBATCH -e scripts/slurm/PTBPE_encode.%j.err # STDERR

CONDA_ENV=${CONDA_ENV:-GeoBPE}                 # name OR absolute path
CONDA_BASE=$(conda info --base)
PYTHON_BIN=${CONDA_PREFIX:-${CONDA_BASE}/envs/${CONDA_ENV}}/bin/python
export PYTHONPATH="$PWD:$PYTHONPATH"
runner="$PYTHON_BIN"

if [ $1 -eq 1 ]; then
  echo "debug"
  # export SLURM_CPUS_PER_TASK=0
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
glue_opt_prior=${15}
glue_opt_method=${16}
free_bonds=${17}
rmsd_super_res=${18}
glue_opt_every=${19}
max_iter=${20}

if [ -n "${21}" ]; then
  extra="--save-dir ${21}"
else
  extra="--auto"
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
cmd=(  
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
    --log-dir logs \
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
    --glue-opt-prior $glue_opt_prior \
    --glue-opt-method $glue_opt_method \
    --glue-opt-every $glue_opt_every \
    --max-iter $max_iter \
    $extra
)
echo "${cmd[@]}"
"${cmd[@]}"