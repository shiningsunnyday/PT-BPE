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
  config="--config config_debug.json"
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

if [ -n "$8" ]; then
  SAVE_DIR=$8
  # ID="${SAVE_DIR##*/}"
  # SRC="/n/holylfs06/LABS/mzitnik_lab/Users/${USER}/foldingdiff/ckpts/$ID/"
  # DST="/n/netscratch/mzitnik_lab/Lab/${USER}/$ID/"
  # mkdir -p "$DST"
  # rsync -av --ignore-existing "${SRC}"*.pkl "$DST"
  extra="--save-dir $8"
else
  extra="--auto"
fi

case "$2" in
  1)
    mode="--mode unary"
    ;;
  2)
    mode="--mode binary --max-seg-len 20"
    ;;
  *)
    mode="--mode recursive --max-seg-len 20"
    ;;
esac

cmd="$runner bin/learn.py --data-dir $3 $config --cuda $device --epochs 1000 --toy $4 --pad $5 --model "feats" $mode --l1 $6 --gamma $7 $extra $debug"
echo $cmd
$cmd