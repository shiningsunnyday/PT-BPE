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
  config="--config ${10}"
  echo "config ${10}"
  runner="python -m pdb -c continue"
  echo "debug"
  device="cpu"
else
  export NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
  if [ $NGPU -eq 1 ]; then
    runner="python"
    echo "runner is python"
  else
    runner="torchrun --nproc_per_node=$NGPU"
    echo "runner is torchrun"
  fi
  debug=""
  device="cuda"
  config="--config config.json"
fi

vis="73 111 32 15 85"

if [ -n "$9" ]; then
  SAVE_DIR=$9
  # ID="${SAVE_DIR##*/}"
  # SRC="/n/holylfs06/LABS/mzitnik_lab/Users/${USER}/foldingdiff/ckpts/$ID/"
  # DST="/n/netscratch/mzitnik_lab/Lab/${USER}/$ID/"
  # mkdir -p "$DST"
  # rsync -av --ignore-existing "${SRC}"*.pkl "$DST"
  extra="--save-dir $9"
else
  extra="--auto"
fi

case "$2" in
  1)    
    if [ $3 -ne 10000000000 ]; then
      echo "max-seg-len must be 10000000000"
      exit 1
    fi
    mode="--mode unary --max-seg-len $3"
    ;;
  2)
    mode="--mode binary --max-seg-len $3"
    ;;
  *)
    mode="--mode recursive --max-seg-len $3"
    ;;
esac

cmd="$runner bin/learn.py --data-dir $4 $config --cuda $device --epochs 1000 --vis-idxes $vis --toy $5 --pad $6 --model "feats" $mode --l1 $7 --gamma $8 $extra $debug"
echo $cmd
$cmd