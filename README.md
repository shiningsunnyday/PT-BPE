# Protein Geometric Byte Pair Encoding

## Installation

To install, clone this using `git clone`. Set up conda env.

```bash
conda env create -f environment.yml
conda activate GeoBPE
pip install esm==3.2.0 --no-deps # install ESM
pip install -e ./  # make sure ./ is the dir including setup.py
```

### Downloading data

### Run GeoBPE

```bash
# go to repo root
./scripts/encode.sh 0 300 100 1 pretrain '1-500' histogram 5 5 false 0 2-100:3-500:5-20:6-100:8-5:9-20:11-1:12-5:14-1 5000 false 1.0 all true true 10 10000 # slurm ready; prepend sbatch settings if using
```
