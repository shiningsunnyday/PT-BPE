#!/bin/bash
#
#SBATCH -c 16 # number of cores
#SBATCH --mem 200g # memory pool for all cores
#SBATCH --gres=gpu:4 # gpu
#SBATCH -t 3-0:00 # time (D-HH:MM)
##SBATCH -t 0-12:00
#SBATCH -o scripts/slurm/PTBPE_train.%j.out # STDOUT
#SBATCH -e scripts/slurm/PTBPE_train.%j.err # STDERR

# If CUDA_MOD or CUDNN_MOD are exported in the environment, load them;
# otherwise do nothing.
if [[ -n "$CUDA_MOD" ]]; then
  module load "cuda/${CUDA_MOD}"
fi
if [[ -n "$CUDNN_MOD" ]]; then
  module load "cudnn/${CUDNN_MOD}"
fi

CONDA_ENV=${CONDA_ENV:-GeoBPE}                 # name OR absolute path
CONDA_BASE=$(conda info --base)
PYTHON_BIN=${CONDA_PREFIX:-${CONDA_BASE}/envs/${CONDA_ENV}}/bin/python
export PYTHONPATH="$PWD:$PYTHONPATH"
runner="$PYTHON_BIN"

if [ $1 -eq 1 ]; then
  debug="--debug"
  echo "debug"
else
  debug=""
fi

case "$2" in
  1)
    task="conserved-site-prediction"
    # ckpt_path="./ckpts/1752523675.4143364/bpe_iter=215.pkl"
    ;;
  2)
    task="CatBio"
    # ckpt_path="./ckpts/1752523819.479008/bpe_iter=155.pkl"
    ;;
  3)
    task="BindBio"
    # ckpt_path="./ckpts/1752566951.7877476/bpe_iter=33.pkl" # still working on it
    ;;
  4)
    task="CatInt"
    # ckpt_path="./ckpts/1752567036.6695304/bpe_iter=100.pkl"
    ;;
  5)
    task="BindInt"
    # ckpt_path="./ckpts/1752610090.6529822/bpe_iter=30.pkl" # untrained
    ;;
  6)
    task="repeat-motif-prediction"
    # ckpt_path="./ckpts/1752521366.505088/bpe_iter=100.pkl"    
    ;;
  7)
    task="remote-homology-detection"
    # ckpt_path="./ckpts/1752525499.4329767/bpe_iter=135.pkl"
    # level="protein"
    ;;
  8)
    task="pretrain"
    ;;
  *)
    echo "Invalid option: $2"
    echo "Usage: $0 {1|2|...|8} 0 {ckpt_path} {model_path} {num_samples} or {1|2|..|8} 1 {train_path} {valid_path} {output_path} {num_samples}"
    exit 1
    ;;
esac

if [ $task != "pretrain" ]; then
  labels_path="./data/struct_token_bench/processed_csvs/${task}.csv"
else
  labels_path=""
fi

if (( $3 == 1 )); then # input mode 1  
  ckpt_info="--train_path $4 --valid_path $5 --output_path $6 --num_samples $7" 
else # input mode 2  
  ckpt_info="--checkpoint_path ${4}"
  if [ -n "${5:-}" ]; then
    extra="--inference --model_path $5 --num_samples $6"
    if [ "${6:-}" = "780" ]; then
      extra="$extra --length_ladder"
    fi
    podman load -i ost.tar
  else
    extra=""
  fi
fi

cmd=(
  $runner -m bin.train \
  $ckpt_info \
  --wandb_team $USER \
  --labels_path "$labels_path" \
  --batch_size 8 \
  # --d_model 512 --num_layers 20 --num_heads 16 --d_ff 2048 \
  --task $task \
  --epochs 1000 \
  $extra $debug
)
echo "${cmd[@]}"
"${cmd[@]}"
