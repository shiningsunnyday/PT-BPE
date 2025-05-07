#!/bin/bash
#
#SBATCH -p kempner # partition (queue)
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 10 # number of cores
#SBATCH --mem 500g # memory pool for all cores
#SBATCH --gres=gpu:1 # gpu
#SBATCH -t 0-1:00 # time (D-HH:MM)
#SBATCH -o slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm/slurm.%N.%j.err # STDERR


# change if needed
project_dir="/n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE"

cd $project_dir

module load cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01

# train
# PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE \
#   /n/netscratch/mzitnik_lab/Lab/afang/envs/stbenv/bin/python -m pdb -m bin.predict \
#   --cuda cuda:0 \
#   --pkl-file /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/ckpts/1746641431.580507/bpe_iter=0.pkl \
#   --task repeat-motif-prediction \
#   --pkl-data-file /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/data/struct_token_bench/processed_pickles/repeat-motif-prediction.pkl \
#   --level residue \
#   --auto

# test
PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE \
  /n/netscratch/mzitnik_lab/Lab/afang/envs/stbenv/bin/python -m pdb -m bin.predict \
  --cuda cuda:0 \
  --pkl-file /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/ckpts/1746641431.580507/bpe_iter=0.pkl \
  --task repeat-motif-prediction \
  --pkl-data-file /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/data/struct_token_bench/processed_pickles/repeat-motif-prediction.pkl \
  --level residue \
  --save-dir /n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE/ckpts/1746646960.2097185/ \
  --test


  # --pkl-data-file ${project_dir}/data/enzyme_commission/EC2_seq30.pkl \
  # ${project_dir}/ckpts/1745337217.4614503/bpe_iter=510.pkl \

# test
# python bin/predict.py --cuda cuda:0 --pkl-file ${project_dir}/ckpts/1745337217.4614503/bpe_iter=510.pkl --test --save-dir ${project_dir}/ckpts/1745337217.4614503

# project_dir="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/"
# PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Lab/afang/PT-BPE \
#   /n/netscratch/mzitnik_lab/Lab/afang/envs/esmenv/bin/python -m bin.predict \
#   --cuda cuda:0 \
#   --pkl-file ${project_dir}/ckpts/1744875790.3072364/bpe_iter=6000.pkl \
#   --pkl-data-file ${project_dir}/homo_datasets_with_test.pkl \
#   --auto