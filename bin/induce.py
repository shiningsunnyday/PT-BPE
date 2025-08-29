import argparse
import pickle
import time
import shutil
import os
import sys
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
from foldingdiff.tokenizer import Tokenizer
from concurrent.futures import ProcessPoolExecutor
from foldingdiff.utils import load_args_from_txt, validate_args_match


def _effective_cpus() -> int:
    """Return the number of CPUs *actually* available to this task."""
    if "SLURM_CPUS_PER_TASK" in os.environ:        
        n = int(os.environ["SLURM_CPUS_PER_TASK"])
        print(f"SLURM_CPUS_PER_TASK={n}")
        return n
    try:                                   # Linux cpuset / cgroups
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1

def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff Induction Script")
    # folder
    parser.add_argument("--src-pkl")
    parser.add_argument("--base-dir", type=str, default="./")
    parser.add_argument("--save-dir")
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory where log files will be saved.")    
    parser.add_argument("--data-dir")
    # data params
    parser.add_argument("--toy", default=0, type=int)
    parser.add_argument("--pad", default=512, type=int)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    return args


def _init_tokenize_worker(bpe, save_dir):
    global BPE, SAVE_DIR
    BPE = bpe
    SAVE_DIR = save_dir


def tokenize_structure(args):
    """Build and tokenize a single structure inside the worker."""
    idx, struc = args
    try:
        pickle.load(open(SAVE_DIR / f"{idx}.pkl", "rb"))
        return
    except:
        print('start tokenize_structure')    
    tok = Tokenizer(struc)
    res = BPE.tokenize(tok)
    pickle.dump((res, tok), open(SAVE_DIR / f"{idx}.pkl", "wb+"))    


def plot_stats(all_stats, output_path, total_ticks=20):
    stats = {
        key: np.mean([d[key] for d in all_stats], axis=0).tolist()
        for key in all_stats[0]
    }
    rmsds, lddts, Ls = stats["rmsd"], stats["lddt"], stats["L"]
    fig, (ax1, ax_rmsd) = plt.subplots(1, 2, figsize=(16, 5)) # make figure + first (left) axis

    # ---------------- left panel : L vs K + BPR on right ---------------
    ax1.plot(Ls,
            marker='o',
            label="L vs Iter",
            linewidth=2)
    skip = (len(Ls)+total_ticks-1)//(total_ticks)    
    ax1.set_ylabel("L (# Motif-Tokens Per PDB)")    
    ax1.set_xlabel(f"Step")
    ax1.set_xticks(range(0, len(Ls), skip))
    ax1.tick_params(axis="y", labelcolor='tab:orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="best")
    ax1.set_title(f"L w/ {len(Ls)} BPE rounds")

    # -------- right panel: BB-RMSD (left-y) & LDDT (right-y) ----------
    ax_rmsd.plot(rmsds,
                 marker='s', linestyle='--', color='tab:red',
                 label="Backbone RMSD")
    ax_rmsd.set_xlabel(f"Step")
    ax_rmsd.set_xticks(range(0, len(Ls), skip))
    ax_rmsd.set_ylabel("Backbone RMSD (Å)", color='tab:red')
    ax_rmsd.tick_params(axis='y', labelcolor='tab:red')

    ax_lddt = ax_rmsd.twinx()                           # second y-axis
    ax_lddt.plot(lddts,
                 marker='o', linestyle='--', color='tab:blue',
                 label="LDDT (mean)")
    ax_lddt.set_ylabel("LDDT", color='tab:blue')
    ax_lddt.tick_params(axis='y', labelcolor='tab:blue')
    ax_rmsd.set_title("Backbone RMSD & LDDT vs K")

    # -------------------- annotate best points ------------------------
    best_rmsd_idx = np.argmin(rmsds)
    ax_rmsd.scatter(best_rmsd_idx, rmsds[best_rmsd_idx],
                    color='tab:red', zorder=5)
    ax_rmsd.annotate(f"Lowest RMSD: {rmsds[best_rmsd_idx]:.2f}",
                     xy=(best_rmsd_idx, rmsds[best_rmsd_idx]),
                     xytext=(10, 15), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color='tab:red'),
                     color='tab:red')

    best_lddt_idx = np.argmax(lddts)
    ax_lddt.scatter(best_lddt_idx, lddts[best_lddt_idx],
                    color='tab:blue', zorder=5)
    ax_lddt.annotate(f"Highest LDDT: {lddts[best_lddt_idx]:.2f}",
                     xy=(best_lddt_idx, lddts[best_lddt_idx]),
                     xytext=(10, -15), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color='tab:blue'),
                     color='tab:blue')

    # ---------------- combine legends for right panel -----------------
    h1, l1 = ax_rmsd.get_legend_handles_labels()
    h2, l2 = ax_lddt.get_legend_handles_labels()
    ax_rmsd.legend(h1 + h2, l1 + l2, loc='best')
    
    fig.tight_layout()
    plt.show()
    plt.savefig(output_path)


def main():
    args = parse_args()
    logging.info(args)
    max_workers = _effective_cpus()
    print(f"{max_workers} workers")
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        cur_time = time.time()
        save_dir = Path(args.base_dir) / f'./ckpts/{cur_time}'
        os.makedirs(save_dir, exist_ok=True)        
    
    print(f"save_dir: {save_dir}")
    src_file = Path(args.src_pkl)
    bpe = pickle.load(open(src_file, "rb"))
    # store metadata
    arg_path = src_file.parent / "args.txt"
    args_path = save_dir / "args.txt"
    out_path = save_dir / src_file.name
    shutil.copyfile(arg_path, save_dir / "orig_args.txt")
    if os.path.exists(args_path):
        print(f"loading args from {args_path}")
        loaded_args = load_args_from_txt(args_path)    
        validate_args_match(
            current   = args,
            loaded    = loaded_args,
            skip = ["auto", "save_dir"]
        )
    else:
        with open(args_path, "w") as f:
            for arg_name, arg_value in sorted(args.__dict__.items()):
                f.write(f"{arg_name}: {arg_value}\n")    
    # induce
    dataset = FullCathCanonicalCoordsDataset(args.data_dir, 
                                        zero_center=False, 
                                        toy=args.toy,
                                        use_cache=False, 
                                        pad=args.pad)
    pargs = []
    idx = 0
    for struc in dataset.structures:
        if (struc['angles']['psi']==struc['angles']['psi']).sum() < len(struc['angles']['psi'])-1:
            print(f"skipping {i}, {struc['fname']} because of missing dihedrals")
        else:
            pargs.append((idx, struc))
            idx += 1

    N = len(pargs)
    tokenizers = []
    all_stats = []
    if max_workers:
        # ----- parallel tokenisation -----------------------------------------------------            
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_tokenize_worker,
            initargs=(bpe, save_dir)                                # only BPE is broadcast
        ) as pool:
            for _ in tqdm(pool.map(
                    tokenize_structure, pargs, chunksize=1),
                    total=N, desc="tokenizing"):
                pass
    else:
        global BPE, SAVE_DIR
        BPE, SAVE_DIR = bpe, save_dir
        for parg in pargs: 
            tokenize_structure(parg)

    for i in tqdm(range(N), desc="loading done tokenizers"):
        assert os.path.exists(save_dir / f"{i}.pkl")
        stats, t = pickle.load(open(save_dir / f"{i}.pkl", "rb"))
        all_stats.append(stats)
        tokenizers.append(t)    
        
    plot_stats(all_stats, save_dir / "stats.png")
    bpe.tokenizers = tokenizers
    pickle.dump(bpe, open(out_path, "wb+"))
    print(os.path.abspath(out_path))


if __name__ == "__main__":
    main()
