#!/bin/bash
#
#SBATCH -p kempner # partition (queue)
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 16 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH --gres=gpu:1 # gpu
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_predict.%j.out # STDOUT
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/PTBPE_predict.%j.err # STDERR


# change if needed
project_dir="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff"

cd $project_dir

module load cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01

# train

if [ $# -ne 1 ]; then
  echo "Usage: $0 {1|2|3|4|5}"
  exit 1
fi

level="residue"
case "$1" in
  1)
    task="conserved-site-prediction"
    pkl_file="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ckpts/1746813758.8588822/bpe_iter=3920.pkl"
    ckpt_dir="./ckpts/1747270965.922862" # done
    ;;
  2)
    task="CatBio"
    pkl_file="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ckpts/1746813758.8587544/bpe_iter=5590.pkl"
    ckpt_dir="./ckpts/1747272230.710532" # done
    ;;
  3)
    task="BindBio"
    pkl_file="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ckpts/1746804072.8771565/bpe_iter=2500.pkl"
    ckpt_dir="./ckpts/1747272233.400883"
    ;;
  4)
    task="CatInt"
    pkl_file="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ckpts/1746804072.8771634/bpe_iter=7200.pkl"
    ckpt_dir="./ckpts/1747272259.176812/" # done
    ;;
  5)
    task="repeat-motif-prediction"
    pkl_file="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ckpts/1746804072.8772147/bpe_iter=9990.pkl"    
    ckpt_dir="./ckpts/1747271619.576858" # done    
    ;;
  6)
    task="remote-homology-detection"
    pkl_file="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/ckpts/1744875790.3072364/bpe_iter=6000.pkl"
    ckpt_dir="./ckpts/1744932629.834232"
    level="protein"
    ;;
  *)
    echo "Invalid option: $1"
    echo "Usage: $0 {1|2|3|4|5}"
    exit 1
    ;;
esac


PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff \
  /n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/predict_env/bin/python -m bin.predict \
  --cuda cuda:0 \
  --pkl-file ${pkl_file} \
  --task ${task} \
  --pkl-data-file /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/data/struct_token_bench/processed_pickles/${task}.pkl \
  --level ${level} \
  --auto
  # --test \
  # --save-dir ${ckpt_dir} \

# test
# PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff \
#   /n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/predict_env/bin/python -m bin.predict \
#   --cuda cuda:0 \
#   --pkl-file ${pkl_file} \
#   --task conserved-site-prediction \
#   --pkl-data-file /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/data/struct_token_bench/processed_pickles/conserved-site-prediction.pkl \
#   --level residue \
  # --save-dir ${pkl_file}/1746646960.2097185/


  # --pkl-data-file ${project_dir}/data/enzyme_commission/EC2_seq30.pkl \
  # ${project_dir}/ckpts/1745337217.4614503/bpe_iter=510.pkl \

# test
# python bin/predict.py --cuda cuda:0 --pkl-file ${project_dir}/ckpts/1745337217.4614503/bpe_iter=510.pkl --test --save-dir ${project_dir}/ckpts/1745337217.4614503

# project_dir="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/"
# PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff \
#   /n/netscratch/mzitnik_lab/Lab/afang/envs/esmenv/bin/python -m bin.predict \
#   --cuda cuda:0 \
#   --pkl-file ${project_dir}/ckpts/1744875790.3072364/bpe_iter=6000.pkl \
#   --pkl-data-file ${project_dir}/homo_datasets_with_test.pkl \
#   --auto