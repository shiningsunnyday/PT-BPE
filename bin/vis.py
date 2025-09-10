import argparse
import concurrent.futures
import subprocess
import pickle
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff Induction Script")
    # folder
    parser.add_argument("--pkl-path")
    args = parser.parse_args()
    return args

def _build_pymol_cmd(t, i):
    """Return the single-line PyMOL command for tokenizer t / index i."""
    start, _, length = max(t.bond_to_token.values(), key=lambda x: x[2])
    start, length = start//3, length//3
    chain  = Path(t.fname).stem.split('_')[1]
    outpng = f"{Path(t.fname).stem}_{start}_{length}.png"
    return (
        "pymol -cq "
        "-d 'run highlight_span.py' "
        f"-d 'highlight_span {t.fname}, {start}, {start+length}, {outpng}, {chain}'"
    )

def _run_batch(batch):
    """
    Worker helper: activate conda env `pstbench` once, then execute
    every PyMOL command in *batch* sequentially.
    `batch` is a list of (tokenizer, index) tuples.
    """
    # create the shell lines
    lines = [
        "source $(conda info --base)/etc/profile.d/conda.sh",
        "conda activate pstbench",
    ] + [_build_pymol_cmd(t, i) for t, i in batch]

    # run them in a single shell so activation is done only once
    script = " && ".join(lines)
    subprocess.run(["bash", "-c", script], check=True)

# --------------------------------------------------------------------------
def main():
    args = parse_args()
    bpe = pickle.load(open(args.pkl_path, "rb"))

    # pick the ten most interesting tokenizers (same logic as before)
    selected = sorted(
        [(i,
          max(l for (_, _, l) in t.bond_to_token.values()),
          -len(t.bond_to_token) / t.n)
         for i, t in enumerate(bpe.tokenizers)]
    )[-100:]

    tasks = [(bpe.tokenizers[i], i) for i, *_ in selected]

    # split the 10 tasks across a handful of workers
    n_workers   = min(4, len(tasks))                 # tweak as you like
    chunk_size  = (len(tasks) + n_workers - 1) // n_workers
    batches     = [tasks[k:k + chunk_size] for k in range(0, len(tasks), chunk_size)]

    # run the batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        pool.map(_run_batch, batches)
    
        
if __name__ == "__main__":
    main()