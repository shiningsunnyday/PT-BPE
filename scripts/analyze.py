#!/usr/bin/env python3
import os
import math
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from pathlib import Path
os.chdir(Path(__file__).parents[1])
from foldingdiff.bpe import BPE
from foldingdiff.utils import load_args_from_txt
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
from foldingdiff.algo import compute_rmsd

NO_ITERS   = 500
STEP_ITER  = 10
RATIO      = 10  # for the L=K diagonal

def modified(t):
    """Return list of indices where bond_to_token value is a tuple."""
    return [k for k,v in t.bond_to_token.items() if isinstance(v[1], tuple)]


def compare(t1, t2):
    """RMSD between two Tokenizers' coords."""
    return compute_rmsd(t1.compute_coords(), t2.compute_coords())


def vis_images(*paths):
    """
    Display an arbitrary number of images in a square-ish grid layout.
    """
    n = len(paths)
    if n == 0:
        return
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.flatten() if hasattr(axes, "__len__") else [axes]
    for ax, p in zip(axes, paths):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_title(os.path.basename(p))
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_history(ckpt_dir, ref, data_dir):
    Ks, Ls, errs = [], [], []
    for t in range(0, NO_ITERS, STEP_ITER):
        pkl_path = os.path.join(ckpt_dir, f"bpe_iter={t}.pkl")
        if not os.path.exists(pkl_path):
            break
        bpe = pickle.load(open(pkl_path, "rb"))
        usage = [len(tok.bond_to_token) for tok in bpe.tokenizers]
        N = len(bpe.tokenizers)
        K = len(bpe._tokens)
        L = float(np.mean(usage))

        # compute error on a small subset
        errors = []
        for i in tqdm(range(min(N, 10)), desc=f"iter {t}"):
            errors.append(compare(bpe.tokenizers[i], ref.tokenizers[i]))
        Ks.append(K); Ls.append(L); errs.append(float(np.mean(errors)))

    Ks = np.array(Ks); Ls = np.array(Ls); errs = np.array(errs)
    N   = len(Ks)

    fig, ax1 = plt.subplots(figsize=(8,5))
    x_diag = np.linspace(Ks.min(), Ks.max(), 100)
    ax1.plot(x_diag, x_diag/RATIO, "--", label=f"L=K (K/L={RATIO:.1f})")
    ax1.plot(Ks, Ls, "o-", label="L vs K")
    ax1.set_xlabel("K (Vocab Size)")
    ax1.set_ylabel("L (#Motif-Tokens Per PDB)")
    ax1.set_xticks(Ks)

    ax2 = ax1.twinx()
    ax2.plot(Ks, errs, "x:", label="Error", color="tab:red")
    ax2.set_ylabel("Error", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # unified legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="best")

    ax1.set_title(f"L vs K for N={N} (BPE rounds: {len(Ks)})")
    fig.tight_layout()

    out_png = os.path.join(ckpt_dir, "run.png")
    plt.savefig(out_png)
    print(f"Saved plot to {out_png}")
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot BPE history for a given run directory")
    p.add_argument("--d", required=True,
                   help="subfolder under ./ckpts/ to load (e.g. '1752210667.563783')")
    p.add_argument("--ckpt-root", default="./ckpts",
                   help="root directory where ckpt subfolders live")
    args = p.parse_args()

    ckpt_dir = os.path.join(args.ckpt_root, args.d)
    args_path = os.path.join(ckpt_dir, "args.txt")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"No args.txt in {ckpt_dir}")

    print(f"Loading saved args from {args_path}")
    saved_args = load_args_from_txt(args_path)

    # build & clean dataset
    ds = FullCathCanonicalCoordsDataset(args.data_dir, 
                                            use_cache=False, 
                                            debug=False, 
                                            zero_center=False, 
                                            toy=args.toy, 
                                            pad=args.pad, 
                                            secondary=args.sec) 
    clean = []
    for i, s in enumerate(ds.structures):
        if (s["angles"]["psi"] == s["angles"]["psi"]).sum() < len(s["angles"]["psi"]) - 1:
            print(f"Skipping structure {i} due to missing dihedrals")
        else:
            clean.append(s)
    ds.structures = clean

    # initialize reference BPE
    ref = BPE(
        ds.structures,
        bins=saved_args.bins,
        bin_strategy=saved_args.bin_strategy,
        save_dir=ckpt_dir,
        rmsd_partition_min_size=saved_args.p_min_size,
        num_partitions=saved_args.num_p,
        compute_sec_structs=saved_args.sec,
        plot_iou_with_sec_structs=saved_args.sec_eval,
        res_init=saved_args.res_init
    )
    ref.initialize()

    # produce and save the history plot
    plot_history(ckpt_dir, ref, saved_args.data_dir)


if __name__ == "__main__":
    main()
