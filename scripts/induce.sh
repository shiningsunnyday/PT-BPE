#!/bin/bash
#
#SBATCH -p test # partition (queue)
#SBATCH -c 20 # number of cores
#SBATCH --mem 500g # memory pool for all cores
#SBATCH -t 0-12:00 # time
#SBATCH -o scripts/slurm/PTBPE_induce.%j.out # STDOUT
#SBATCH -e scripts/slurm/PTBPE_induce.%j.err # STDERR

CONDA_ENV=${CONDA_ENV:-GeoBPE}                 # name OR absolute path
CONDA_BASE=$(conda info --base)
PYTHON_BIN=${CONDA_PREFIX:-${CONDA_BASE}/envs/${CONDA_ENV}}/bin/python
export PYTHONPATH="$PWD:$PYTHONPATH"
runner="$PYTHON_BIN"

if [ $1 -eq 1 ]; then
  echo "debug"
  export SLURM_CPUS_PER_TASK=0
  runner="${runner} -m pdb -c continue"
fi

toy=$2
data_dir=$3
pad="512"
src_pkl=$4
processed=$5 # due to time constraints

if [ -n "${6}" ]; then
  extra="--save-dir ${6}"
else
  extra=""
fi
if (( $3 == prevalid )); then
  extra="${extra} --append"
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
cmd=(
  $runner bin/induce.py \
  --data-dir $data_dir \
  --pad $pad \
  --toy $toy \
  --src-pkl $src_pkl \
  --processed $processed \
  $extra
)
echo "${cmd[@]}"
"${cmd[@]}"
