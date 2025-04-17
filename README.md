# Protein Byte Pair Encoding

## Installation

To install, clone this using `git clone`. Set up conda env.

```bash
conda env create -f environment.yml
conda activate foldingdiff
pip install -e ./  # make sure ./ is the dir including setup.py
```

Set up another conda env to use ESM.

```bash
conda env create -f esm_env.yml
conda activate foldingdiff
pip install -e ./  # make sure ./ is the dir including setup.py
```

### Downloading data

#### Original (CATH)
We require some data files not packaged on Git due to their large size. These are not required for sampling (as long as you are not using the `--testcomparison` option, see below); this is required for training your own model. We provide a script in the `data` dir to download requisite CATH data.

```bash
# Download the CATH dataset
cd data  # Ensure that you are in the data subdirectory within the codebase
chmod +x download_cath.sh
./download_cath.sh
```

#### Remote Homology Detection (Tape)

Get the LMDB data from https://github.com/songlab-cal/tape.

#### ... (...)

### Run Algorithm

```bash
cd scripts
chmod +x encode.sh
./encode.sh
```