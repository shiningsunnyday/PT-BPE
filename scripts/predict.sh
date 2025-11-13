#!/bin/bash
#
#SBATCH -c 16 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH --gres=gpu:1 # gpu
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o scripts/slurm/PTBPE_predict.%j.out # STDOUT
#SBATCH -e scripts/slurm/PTBPE_predict.%j.err # STDERR


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

# train

if [[ $# -ne 3 && $# -ne 4 ]]; then
  echo "Usage: $0 {0|1} {1|2|..|10} {pkl_file} [train-frac]"
  exit 1
fi

level="residue"
reg="false"
case "$2" in
  1)
    task="BindInt"
    ;;
  2)
    task="BindBio"
    ;;  
  3)
    task="CatInt"
    ;;
  4)
    task="CatBio"
    ;;
  5)
    task="conserved-site-prediction"     
    ;;
  6)
    task="repeat-motif-prediction"
    ;;  
  7)
    task="epitope-prediction"
    ;;  
  8)
    task="structural-flexibility-prediction"
    reg="true"
    ;;  
  9)
    task="BindShake"    
    ;;  
  10)
    task="remote-homology-detection"
    level="protein"
    ;;  
  *)
    echo "Invalid option: $2"
    echo "Usage: $0 {1|2|...|10} {pkl_file} [train-frac]"
    exit 1
    ;;
esac

pkl_file=$3
train_frac=${4:-}
cmd=(
  $runner -m bin.predict \
  --cuda cuda:0 \
  --pkl-file ${pkl_file} \
  --task ${task} \
  --pkl-data-file data/struct_token_bench/processed_pickles/${task}.pkl \
  --level ${level} \
  --regression $reg \
  --auto
  # --test \
  # --save-dir ${ckpt_dir} \
)
if [[ -n "$train_frac" ]]; then
  cmd+=( --train-frac "$train_frac" )
fi
echo "${cmd[@]}"
"${cmd[@]}"
