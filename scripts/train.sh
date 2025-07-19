#!/bin/bash
#
##SBATCH -p gpu_test # partition (queue)
#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 16 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH --gres=gpu:1 # gpu
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
    ckpt_path="./ckpts/1752523675.4143364/bpe_iter=215.pkl"
    task="conserved-site-prediction"
    ;;
  2)
    ckpt_path="./ckpts/1752523819.479008/bpe_iter=155.pkl"
    task="CatBio"
    ;;
  3)
    ckpt_path="./ckpts/1752566951.7877476/bpe_iter=33.pkl" # still working on it
    task="BindBio"
    ;;
  4)
    ckpt_path="./ckpts/1752567036.6695304/bpe_iter=100.pkl"
    task="CatInt"
    ;;
  5)
    ckpt_path="./ckpts/1752610090.6529822/bpe_iter=30.pkl"
    task="BindInt"
    ;;
  6)
    ckpt_path="./ckpts/1752521366.505088/bpe_iter=100.pkl"
    task="repeat-motif-prediction"
    ;;
  7)
    ckpt_path="./ckpts/1752525499.4329767/bpe_iter=135.pkl"
    task="remote-homology-detection"
    level="protein"
    ;;
  *)
    echo "Invalid option: $2"
    echo "Usage: $0 {1|2|3|4|5}"
    exit 1
    ;;
esac

if [ -n "$3" ]; then
  extra="--inference --model_path $3"
else
  extra=""
fi

labels_path="./data/struct_token_bench/processed_csvs/${task}.csv"
python bin/train.py \
    --checkpoint_path $ckpt_path \
    --labels_path $labels_path \
    --batch_size 8 \
    --eval_interval 100 \
    --task $task \
    $extra $debug
