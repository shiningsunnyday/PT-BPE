#!/bin/bash
#
##SBATCH -p gpu_test # partition (queue)
#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 16 # number of cores
#SBATCH --mem 600g # memory pool for all cores
#SBATCH --gres=gpu:4 # gpu
#SBATCH -t 3-0:00 # time (D-HH:MM)
##SBATCH -t 0-12:00
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_train.%j.out # STDOUT
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_train.%j.err # STDERR


# if [ $# -ne 1 ]; then
#   echo "Usage: $0 {1|2|3|4|5|6|7}"
#   exit 1
# fi

module load cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01

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
  if [ -n "$5" ]; then # specify model for sampling
    extra="--inference --model_path $5 --num_samples $6"
    # load docker image for lddt
    podman load -i ost.tar
  else
    extra=""
  fi
fi

export PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff
cmd=(
  /n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/esm_env/bin/python -m bin.train \
  $ckpt_info \
  --labels_path "$labels_path" \
  --batch_size 8 \
  --task $task \
  $extra $debug
)
echo "${cmd[@]}"
"${cmd[@]}"
