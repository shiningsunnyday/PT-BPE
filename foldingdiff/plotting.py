"""
Utility functions for plotting
"""
import os, sys
import re
from pathlib import Path
from typing import Optional, Sequence, Union
import pickle
from copy import deepcopy
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
import argparse
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerBase
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
from foldingdiff.algo import compute_rmsd
from esm.utils.structure.protein_chain import ProteinChain
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import torch
from torch.utils.data import Dataset

PLOT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots"))
if not PLOT_DIR.is_dir():
    os.makedirs(PLOT_DIR)

# Dummy class representing a backbone fragment (for legend handles)
class BackboneFragment:
    def __init__(self, token_label, token_lookup, color):
        self.token_label = token_label
        self.color = color
        self.token_lookup = token_lookup

# Custom legend handler drawing a backbone fragment.
class HandlerBackboneFragment(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Assume color_list is defined, e.g.:
        # color_list = ['red', 'blue', 'green', 'orange']  # or any list of colors
        color_list = orig_handle.token_lookup
        n = len(color_list)
        # Determine a circle radius.
        # We use the smaller of (height/4) or half the spacing available if we equally divide the width.
        r = min(height / 2.0, (width / (n + 1)) / 2.0)

        # Compute x positions for the circle centers.
        # We add a margin of r from each edge to prevent clipping.
        x_positions = np.linspace(xdescent + r, xdescent + width - r, n)
        y_center = ydescent + height / 2.0

        # Create circles for each color.
        circles = [Circle((x, y_center), radius=r, fc=color)
                for x, color in zip(x_positions, color_list)]

        # Create lines connecting consecutive circles.
        lines = [Line2D([x_positions[i], x_positions[i + 1]], [y_center, y_center],
                        color=orig_handle.color, lw=r) for i in range(n - 1)]

        # Combine all artists.
        artists = circles + lines

        # Apply the transformation.
        for artist in artists:
            artist.set_transform(trans)
        return artists


def get_codebook_utility(input_ids, vocab_size, eps=1e-8):
    index_count = torch.bincount(input_ids, minlength=vocab_size)
    # normalize frequency to probs
    probs = index_count / torch.sum(index_count)
    # perplexity
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + eps), dim=-1))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    # the percentage of used indices
    num_total = len(index_count)
    use_ratio = torch.count_nonzero(index_count) / num_total
    utility = {
        "perplexity": perplexity,
        "perplexity_normalized": perplexity / vocab_size,
        "entropy": entropy,
        "entropy_normalized": entropy / vocab_size,
        "use_ratio": use_ratio,
    }
    return {k: v.item() for k, v in utility.items()}


def plot_joint_kde(
    x_values, y_values, show_axes: bool = True, fname: Optional[str] = None, **kwargs
):
    """
    Plot a density scatter plot (KDE) of the values. kwargs are passed to
    ax.set()

    Useful for plotting Ramachandran plot - x = phi, y = psi
    https://proteopedia.org/wiki/index.php/Ramachandran_Plots
    """
    fig, ax = plt.subplots(dpi=300)
    sns.kdeplot(x=x_values, y=y_values, levels=100, fill=True, norm=LogNorm(), ax=ax)
    if show_axes:
        ax.axvline(0, color="grey", alpha=0.5)
        ax.axhline(0, color="grey", alpha=0.5)
    ax.set(**kwargs)
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_val_dists_at_t(
    t: int,
    dset: Dataset,
    share_axes: bool = True,
    zero_center_angles: bool = False,
    fname: Optional[str] = None,
):
    select_by_attn = lambda x: x["corrupted"][torch.where(x["attn_mask"])]

    retval = []
    for i in range(len(dset)):
        vals = dset.__getitem__(i, use_t_val=t)
        assert vals["t"].item() == t, f"Unexpected values of t: {vals['t']} != {t}"
        retval.append(select_by_attn(vals))
    vals_flat = torch.vstack(retval).numpy()
    assert vals_flat.ndim == 2

    ft_names = dset.feature_names["angles"]
    n_fts = len(ft_names)
    assert vals_flat.shape[1] == n_fts

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_fts,
        sharex=share_axes,
        sharey=share_axes,
        dpi=300,
        figsize=(2.6 * n_fts, 2.5),
    )
    for i, (ax, ft_name) in enumerate(zip(axes, ft_names)):
        # Plot the values
        vals = vals_flat[:, i]
        sns.histplot(vals, ax=ax)
        if "dist" not in ft_name:
            if zero_center_angles:
                ax.axvline(np.pi, color="tab:orange")
                ax.axvline(-np.pi, color="tab:orange")
            else:
                ax.axvline(0, color="tab:orange")
                ax.axvline(2 * np.pi, color="tab:orange")
        ax.set(title=f"Timestep {t} - {ft_name}")
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_losses(
    log_fname: str,
    out_fname: Optional[str] = None,
    simple: bool = False,
    pattern: Optional[str] = None,
):
    """
    Plot the validation loss values from a log file. Spuports multiple
    validation losses if present in log file. Plots per epoch, and if multiple
    values are record for an epoch, plot the median.
    """

    def keyfunc(x: str) -> tuple:
        """
        Validation first, then train
        """
        ordering = ["test", "val", "train"]
        if "_" in x:
            x_split, x_val = x.split("_", maxsplit=1)
            x_retval = tuple([ordering.index(x_split), x_val])
        else:
            x_retval = (len(ordering), x)
        assert len(x_retval) == 2
        return x_retval

    if simple:
        assert pattern is None
        pattern = re.compile(r"_loss$")
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    fig, ax = plt.subplots(dpi=300)

    df = pd.read_csv(log_fname)
    cols = df.columns.to_list()
    cols = sorted(cols, key=keyfunc)
    for colname in df.columns:
        if "loss" not in colname:
            continue
        if pattern is not None:
            if not pattern.search(colname):
                continue
        vals = df.loc[:, ["epoch", colname]]
        vals.dropna(axis="index", how="any", inplace=True)
        sns.lineplot(x="epoch", y=colname, data=vals, ax=ax, label=colname, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss over epochs")

    if out_fname is not None:
        fig.savefig(out_fname, bbox_inches="tight")
    return fig


def plot_consecutive_heatmap(
    vals: Union[Sequence[float], Sequence[Sequence[float]]],
    fname: Optional[str] = None,
    logstretch_vmax: float = 2e3,
    **kwargs,
):
    """
    Plot a heatmap of consecutive values.
    """
    consecutive_pairs = []

    get_pairs = lambda x: np.array(list(zip(x[:-1], x[1:])))
    # Require these more complex checks because vals may not be of the same
    # size and therefore may not be stackable
    if isinstance(vals[0], (float, int)):
        # 1-dimensional
        consecutive_pairs = get_pairs(vals)
    else:
        # 2-dimensional
        consecutive_pairs = np.vstack([get_pairs(vec) for vec in vals])
    assert consecutive_pairs.ndim == 2
    assert consecutive_pairs.shape[1] == 2

    norm = ImageNormalize(vmin=0.0, vmax=logstretch_vmax, stretch=LogStretch())

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(
        consecutive_pairs[:, 0], consecutive_pairs[:, 1], norm=norm
    )
    fig.colorbar(density, label="Points per pixel")

    ax.set(**kwargs)

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig

def equal_count_bin_edges(values, n_bins):
    """
    Compute bin edges so that `values` are equally partitioned into `n_bins` bins.
    The first edge is min(values), the last is max(values), and widths can vary.

    Parameters
    ----------
    values : array‑like
        1D array of data points.
    n_bins : int
        Number of bins to create.

    Returns
    -------
    edges : ndarray, shape (n_bins+1,)
        The bin edges from min to max.
    """
    values = np.sort(np.asarray(values))
    # Quantiles from 0 to 1 in n_bins steps → edges at those quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, quantiles) 
    return edges


def save_histogram_equal_counts(angles, circular=True, path=None, bins=10, title=None):
    angles = np.asarray(angles)
    if circular:
        # map from [-π,π] → [0,2π)
        angles = (angles + 2*np.pi) % (2*np.pi)

    # get variable-width edges so each bin has ~equal count
    edges = equal_count_bin_edges(angles, bins)
    counts, _   = np.histogram(angles, bins=edges)
    widths      = np.diff(edges)
    centers     = edges[:-1] + widths/2
    default_title = "Circular Histogram (Equal‑Count Bins)" if circular else "Histogram (Equal-Count Bins)"

    if path is not None:            
        if circular:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        else:
            fig, ax = plt.subplots()
        ax.bar(centers, counts, width=widths, bottom=0.0, edgecolor='black', align='center')        
        ax.set_title(title or default_title)
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    return edges[:-1], edges[1:], widths, counts


def save_histogram(angles, circular=True, path=None, bins=10, title=None, cover=False):
    """
    Creates a circular (rose) histogram from a list of angle values (in radians, range -pi to pi)
    and saves the figure at the specified file path.
    
    Parameters:
        angles (list or array-like): Angle values in radians, expected in the range [-pi, pi].
        path (str): The file path where the figure will be saved.
        bins (int or str, optional): See numpy.histogram 'bins' parameter
    """
    # Convert angles from [-pi, pi] to [0, 2*pi]
    angles = np.array(angles)
    if circular:
        angles = (angles + 2*np.pi) % (2*np.pi)
        counts, bin_edges = np.histogram(angles, bins=bins, range=(0, 2*np.pi) if cover else None)
    else:
        counts, bin_edges = np.histogram(angles, bins=bins)
    widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + widths / 2
    default_title = "Circular Histogram (Rose Diagram)" if circular else "Histogram (Rose Diagram)"
    if path is not None:
        # Create a polar subplot
        if circular:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})    
        else:
            fig, ax = plt.subplots()
        # Create a bar plot for each bin in the polar coordinate system
        ax.bar(bin_centers, counts, width=widths, bottom=0.0, edgecolor='black', align='center')    
        ax.set_title(title if title else default_title)
        plt.savefig(path, bbox_inches='tight')
        print("Histogram saved at", os.path.abspath(path))
        plt.close()
    return bin_edges[:-1], bin_edges[1:], widths, counts


def legend_key_to_tuple(label):
    if label in ['N', 'CA', 'C']:
        return (0,)
    else:
        tuple_pat = re.compile(r'^Token\s+(\(\d+,\s*\d+\))$')
        int_pat = re.compile(r'^Token\s+(\d+)$')
        match = tuple_pat.match(label)
        if match is None:
            match = int_pat.match(label)
        assert match is not None
        assert len(match.groups()) == 1
        v = eval(match.group(1))
        if isinstance(v, tuple):
            return v
        else:            
            return (v,)

# Helper: Compute dihedral angle from four points.
def dihedral(p0, p1, p2, p3):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    n0 = np.cross(b0, b1)
    n1 = np.cross(b1, b2)
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    x = np.dot(n0, n1)
    y = np.dot(np.cross(n0, n1), b1)
    return np.arctan2(y, x)

def column_fill(handles, labels, ncol):
    """
    Reorder (handles, labels) so that legend(..., ncol=ncol) will fill
    full first column before moving on to the next.
    """
    N = len(handles)
    nrows = int(np.ceil(N / ncol))
    pad   = nrows * ncol - N

    # pad out so we have a full rectangle
    h_pad = handles + [None]*pad
    l_pad = labels  + ['']*pad

    # build it row‑wise
    rows_h = [ h_pad[i*ncol:(i+1)*ncol] for i in range(nrows) ]
    rows_l = [ l_pad[i*ncol:(i+1)*ncol] for i in range(nrows) ]

    # now pull out column‑by‑column
    new_h, new_l = [], []
    for c in range(ncol):
        for r in range(nrows):
            if rows_l[r][c]:
                new_h.append(rows_h[r][c])
                new_l.append(rows_l[r][c])
    return new_h, new_l


# plot(bpe, num_tokens, ref_coords, output_path, no_iters, prev_iter, step_iter, ratio, num_random_baseline, total_ticks)
def plot(bpe, num_tokens, ref_coords, output_path, no_iters=500, prev_iter=0, step_iter=10, ratio=1, num_random_baseline=10, total_ticks=20):
    N = len(bpe.tokenizers)
    d = Path(output_path).parent
    if prev_iter >= 0:
        prev_path = d / f"run_iter={prev_iter}.npy"
        print(f"loading prev_path: {prev_path}")
        prev = np.load(prev_path)
        Ks, Ls, bprs, *errs = map(list, tuple(prev))
        errs = list(zip(*errs))
        start_iter = prev_iter+step_iter
    else:
        Ks, Ls, bprs, errs = [], [], [], []
        tokenizers = bpe.tokenizers[:len(ref_coords)]
        start_iter = 0
    orig_chains = [ProteinChain.from_pdb(bpe.tokenizers[i].fname) for i in range(len(ref_coords))]
    for t in range(start_iter, no_iters+1, step_iter):
        path = f'{d}/ref_tokenizers={t}.pkl'
        stats_path = f'{d}/stats={t}.json'
        if not os.path.exists(path):
            break
        tokenizers = pickle.load(open(path, 'rb'))
        assert len(tokenizers) == len(ref_coords)
        stats = json.load(open(stats_path))
        Ks.append(stats["K"])
        Ls.append(stats["L"])
        bprs.append(stats["bpr"])
        cur_coords = []
        for i in range(len(ref_coords)):
            coord = tokenizers[i].compute_coords()
            assert ref_coords[i].shape == coord.shape
            cur_coords.append(coord)
        errors = []
        for i in tqdm(range(len(ref_coords))):
            # error = compute_rmsd(cur_coords[i], ref_coords[i]) 
            orig_chain = orig_chains[i]
            chain_recon = ProteinChain.from_backbone_atom_coordinates(cur_coords[i].reshape(-1, 3, 3))
            bb_rmsd = chain_recon.rmsd(orig_chain, only_compute_backbone_rmsd=True)
            lddt = np.array(chain_recon.lddt_ca(orig_chain))
            # errors.append((error, bb_rmsd, lddt.mean()))
            errors.append((bb_rmsd, lddt.mean()))
        err = np.mean(errors, axis=0)
        errs.append(err)
    Ks = np.array(Ks)
    Ls = np.array(Ls)
    bprs = np.array(bprs)
    errs = np.array(errs)        
    cur_path = d / f"run_iter={no_iters}.npy" # cache
    np.save(Path(output_path).with_suffix(".npy"), np.concatenate((np.stack((Ks, Ls, bprs), axis=0), errs.T), axis=0))
    final_iter = t+1
    ref_errs = [] # add a permutation test baseline 
    for k in tqdm(range(num_random_baseline), "random angles baseline"): # sample num_random_baseline random permutations    
        for key in ["tau","CA:C:1N","C:1N:1CA"]+["psi","omega","phi"]:
            bin_c = np.array(bpe._bin_counts[1][key])
            threshs = bpe._thresholds[1][key]
            keep_nan_resample_val = lambda val: np.random.uniform(*threshs[np.random.choice(len(bin_c), p=bin_c/bin_c.sum())]) if val==val else val
            for t in tokenizers:
                randvals = t.angles_and_dists[key].map(keep_nan_resample_val)
                t.angles_and_dists[key] = randvals
        alt_coords = [tokenizers[i].compute_coords() for i in range(len(ref_coords))]
        errors = []
        for i in range(len(ref_coords)):
            # error = compute_rmsd(alt_coords[i], ref_coords[i]) 
            orig_chain = orig_chains[i]
            chain_recon = ProteinChain.from_backbone_atom_coordinates(alt_coords[i].reshape(-1, 3, 3))
            bb_rmsd = chain_recon.rmsd(orig_chain, only_compute_backbone_rmsd=True)
            lddt = np.array(chain_recon.lddt_ca(orig_chain))
            # errors.append((error, bb_rmsd, lddt.mean()))
            errors.append((bb_rmsd, lddt.mean()))
        ref_errs.append(np.mean(errors, axis=0))
    ref_err = np.mean(ref_errs, axis=0)
    if ratio is None:
        ratio = N/1000    
    fig, (ax1, ax_rmsd) = plt.subplots(1, 2, figsize=(16, 5)) # make figure + first (left) axis
    # ---------------- left panel : L vs K + BPR on right ---------------
    x_diag = np.linspace(Ks.min(), Ks.max(), 100)
    ax1.plot(x_diag, x_diag/ratio,
            linestyle='--',
            label=f"L=K/ratio={ratio:.1f}")
    ax1.plot(Ks, Ls,
            marker='o',
            label="L vs K",
            linewidth=2)
    skip = (len(Ks)+total_ticks-1)//(total_ticks)    
    diff = np.abs(Ls - Ks/ratio)
    idx  = np.argmin(diff)      
    K_int, L_int = Ks[idx], Ls[idx]
    ax1.scatter([K_int], [L_int],
                color='orange',
                s=100,
                zorder=5,
                label="Approx. Intersection")
    ax1.annotate(
        f"K≈{K_int}", 
        xy=(K_int, L_int),
        xytext=(15, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="orange"),
        color="orange"
    )
    ax1.set_ylabel("L (# Motif-Tokens Per PDB)")    
    ax1.set_xlabel(f"K (Vocab Size) Every {skip*step_iter} Rounds")
    ax1.set_xticks(Ks if skip == 1 else Ks[0::skip])
    ax1.tick_params(axis="y", labelcolor='tab:orange')

    ax2 = ax1.twinx()
    ax2.plot(Ks, bprs,
             marker='x',
             linestyle=':',
             color='tab:purple',
             label="BPR")
    ax2.set_ylabel("Bits-per-residue (BPR)", color='tab:purple')
    ax2.tick_params(axis="y", labelcolor="tab:purple")    

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()    
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(f"L & Compression vs K for N={N} w/ {final_iter} BPE rounds")

    # -------- right panel: BB-RMSD (left-y) & LDDT (right-y) ----------
    ax_rmsd.plot(Ks, errs[:, 0],
                 marker='s', linestyle='--', color='tab:red',
                 label="Backbone RMSD")
    ax_rmsd.axhline(y=ref_err[0], color='tab:red', linestyle=':',
                    label=f"Ref. Backbone RMSD ({num_random_baseline} perm.)")
    ax_rmsd.set_xlabel("K (Vocab Size)" if skip == 1
                       else f"K (Vocab Size) Every {skip} Rounds")
    ax_rmsd.set_ylabel("Backbone RMSD (Å)", color='tab:red')
    ax_rmsd.tick_params(axis='y', labelcolor='tab:red')

    ax_lddt = ax_rmsd.twinx()                           # second y-axis
    ax_lddt.plot(Ks, errs[:, 1],
                 marker='o', linestyle='--', color='tab:blue',
                 label="LDDT (mean)")
    ax_lddt.axhline(y=ref_err[1], color='tab:blue', linestyle=':',
                    label=f"Ref. LDDT ({num_random_baseline} perm.)")
    ax_lddt.set_ylabel("LDDT", color='tab:blue')
    ax_lddt.tick_params(axis='y', labelcolor='tab:blue')
    ax_rmsd.set_title("Backbone RMSD & LDDT vs K")

    # -------------------- annotate best points ------------------------
    best_rmsd_idx = np.argmin(errs[:, 0])
    ax_rmsd.scatter(Ks[best_rmsd_idx], errs[best_rmsd_idx, 0],
                    color='tab:red', zorder=5)
    ax_rmsd.annotate(f"Lowest RMSD: {errs[best_rmsd_idx,0]:.2f}",
                     xy=(Ks[best_rmsd_idx], errs[best_rmsd_idx,0]),
                     xytext=(10, 15), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color='tab:red'),
                     color='tab:red')

    best_lddt_idx = np.argmax(errs[:, 1])
    ax_lddt.scatter(Ks[best_lddt_idx], errs[best_lddt_idx, 1],
                    color='tab:blue', zorder=5)
    ax_lddt.annotate(f"Highest LDDT: {errs[best_lddt_idx,1]:.2f}",
                     xy=(Ks[best_lddt_idx], errs[best_lddt_idx,1]),
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


# bpe = pickle.load(open(f'./ckpts/{d}/bpe_iter=0.pkl', 'rb'))

def plot_backbone(coords, output_path, atom_types, tokens=None, title="", zoom_factor=0.7, vis_dihedral=True, vis_bond_angle=True, xlim=None, ylim=None, zlim=None, n_per_row=50, bbox_inches=None):
    """
    Plots a protein backbone given an array of coordinates and saves the image as a PNG.
    
    Parameters:
      coords : np.ndarray
          A (3*n, 3) array of 3D coordinates in the order:
          N, CA, C, N, CA, C, ... for each residue.
      output_path : str
          The path where the PNG image will be saved.
      atom_types : list or np.ndarray of str
          Atom symbols for each coordinate in coords.
      tokens : list of tuples (int, str, int), optional
          Each tuple (start, bt, length) indicates that the bonds from index start to start+length-1 
          (i.e. atoms start through start+length) belong to the same contiguous token.
          Each token will be drawn in color (using a colormap) and each individual bond segment is
          annotated with its token label (bt). If None, the entire backbone is drawn in red.
      title : str, optional
          Title for the plot.
      zoom_factor : float, optional
          Factor for zooming in on the coordinate bounds.
      vis_dihedral : bool, optional
          Whether to visualize dihedral angles (turn off for readability)
      vis_bond_angle : bool, optional
          Whether to visualize bond angles
      xlim : (xmin, xmax) figure ranges, e.g. for consistency with previous iterations
      ylim, zlim : similar as xlim
    """
    coords = np.asarray(coords)
    n_atoms = coords.shape[0]
    # Compute a marker/font scale: if many atoms, scale down; if few, scale up.
    marker_scale = np.clip(np.sqrt(300.0 / n_atoms), 0.1, 10.)
    atom_marker_size = 20 * marker_scale
    boundary_marker_size = 50 * marker_scale
    angle_font_size = 4 * marker_scale

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    atom_types_arr = np.array(atom_types)
    N_coords = coords[atom_types_arr == 'N']
    CA_coords = coords[atom_types_arr == 'CA']
    C_coords = coords[atom_types_arr == 'C']
    
    ax.scatter(N_coords[:, 0], N_coords[:, 1], N_coords[:, 2],
               color='blue', s=atom_marker_size, label='N')
    ax.scatter(CA_coords[:, 0], CA_coords[:, 1], CA_coords[:, 2],
               color='orange', s=atom_marker_size, label='CA')
    ax.scatter(C_coords[:, 0], C_coords[:, 1], C_coords[:, 2],
               color='green', s=atom_marker_size, label='C')
    
    # Plot bonds using tokens if provided.
    token_lookup = {}
    cmap = cm.get_cmap('tab10')
    if tokens is not None:
        for token in tokens:
            start, bt, length = token
            token_coords = coords[start: start + length + 1]
            # Draw each bond segment in the token.
            for i in range(length):
                segment = token_coords[i:i+2]
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                        color=cmap((sum(bt) if isinstance(bt, tuple) else bt) % 10), 
                        label=f"Token {bt}" if i == 0 else None,
                        linewidth=2)
        # Highlight the boundary atoms between tokens.
        for token in tokens:
            start, bt, length = token
            if bt not in token_lookup:
                token_lookup[bt] = [{'N': 'blue', 'CA': 'orange', 'C': 'green'}[atom_types[index]]
                                     for index in range(start, start+length+1)]
            left_atom = coords[start]
            right_atom = coords[start + length]
            ax.scatter(left_atom[0], left_atom[1], left_atom[2],
                       color='black', s=boundary_marker_size, zorder=10)
            ax.scatter(right_atom[0], right_atom[1], right_atom[2],
                       color='black', s=boundary_marker_size, zorder=10)
    else:
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color='red', linewidth=2)
    
    # Compute shortest bond length.
    bond_lengths = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    min_bond_length = bond_lengths.min()
    r = 0.3 * min_bond_length  # for bond-bond joint arc
    
    # Annotate each bond-bond joint with its angle.
    if vis_bond_angle:
        for i in range(1, n_atoms - 1):
            joint = coords[i]
            v1 = coords[i-1] - joint
            v2 = coords[i+1] - joint
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue
            u1 = v1 / norm1
            u2 = v2 / norm2
            dot_val = np.dot(u1, u2)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angle = np.arccos(dot_val)
            
            # Compute perpendicular vector in the plane.
            w = u2 - np.dot(u1, u2) * u1
            norm_w = np.linalg.norm(w)
            if norm_w == 0:
                continue
            w = w / norm_w
            
            t_vals = np.linspace(0, angle, 50)
            arc_points = np.array([joint + r * (np.cos(t) * u1 + np.sin(t) * w)
                                    for t in t_vals])
            ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                    color='black', linewidth=2)
            mid_point = joint + r * (np.cos(angle/2) * u1 + np.sin(angle/2) * w)
            txt = ax.text(mid_point[0], mid_point[1], mid_point[2],
                        f"{angle:.2f}", color='black', fontweight='bold', fontsize=angle_font_size)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # --- Torsion Loop: Draw rotating loop representing the dihedral angle ---
    # Dihedral angles require 4 consecutive points; use bond between p1 and p2.
    if vis_dihedral and n_atoms >= 4:
        for i in range(1, n_atoms - 2):
            p0 = coords[i-1]
            p1 = coords[i]
            p2 = coords[i+1]
            p3 = coords[i+2]
            torsion_angle = dihedral(p0, p1, p2, p3)
            # Use bond between p1 and p2.
            bond_vec = p2 - p1
            bond_len = np.linalg.norm(bond_vec)
            if bond_len == 0:
                continue
            v = bond_vec / bond_len
            midpoint = (p1 + p2) / 2.0
            
            # Determine an arbitrary vector perpendicular to v.
            arbitrary = np.array([1, 0, 0])
            if np.abs(v[0]) > 0.9:
                arbitrary = np.array([0, 1, 0])
            u = np.cross(v, arbitrary)
            if np.linalg.norm(u) == 0:
                continue
            u = u / np.linalg.norm(u)
            w = np.cross(v, u)
            
            arc_radius = 0.2 * bond_len  # radius for torsion loop
            t_vals = np.linspace(0, torsion_angle, 50)
            arc_points = np.array([midpoint + arc_radius * (np.cos(t) * u + np.sin(t) * w)
                                   for t in t_vals])
            ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                    color='magenta', linewidth=2)
            arc_mid = midpoint + arc_radius * (np.cos(torsion_angle/2) * u + np.sin(torsion_angle/2) * w)
            txt = ax.text(arc_mid[0], arc_mid[1], arc_mid[2], f"{torsion_angle:.2f}",
                          color='magenta', fontweight='bold', fontsize=angle_font_size)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    
    if title:
        ax.set_title(title, fontsize=4 * marker_scale)
    
    # Zoom in on the coordinates.
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()    
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    new_range = max_range * zoom_factor
    if xlim:
        ax.set_xlim(*xlim)
    else:
        xlim = (x_mid - new_range/2, x_mid + new_range/2)        
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ylim = (y_mid - new_range/2, y_mid + new_range/2)
    if zlim:
        ax.set_zlim(*zlim)
    else:
        zlim = (z_mid - new_range/2, z_mid + new_range/2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)    
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    # Sorting legend entries (assumes helper legend_key_to_tuple is defined elsewhere)    
    for handle, label in sorted(zip(handles, labels), key=lambda it: legend_key_to_tuple(it[1])):
        if label not in unique:
            unique[label] = handle
    handles = list(unique.values())
    labels = list(unique.keys())
    if tokens is not None:
        for i, label in enumerate(labels):
            bt = legend_key_to_tuple(label)
            if bt == (0,):
                continue
            if len(bt) == 1:
                bt = bt[0]                
                color = cmap(bt % 10)
            else:
                color = cmap(sum(bt) % 10)
            handles[i] = BackboneFragment(bt, token_lookup[bt], color)
        ncol = (len(handles)+n_per_row-1)//n_per_row
        handles, labels = column_fill(handles, labels, ncol)
        ax.legend(handles, labels,
                  handler_map={BackboneFragment: HandlerBackboneFragment()}, 
                  ncols=ncol, 
                  bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.,
                  loc='upper left')
    else:
        handles = list(unique.values())
        labels = list(unique.keys())
        ncol = (len(handles)+n_per_row-1)//n_per_row
        handles, labels = column_fill(handles, labels, ncol)
        ax.legend(unique.values(), unique.keys(), 
        ncols=ncol, 
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.,
        loc='upper left')
    plt.subplots_adjust(right=0.50)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches=bbox_inches)
    plt.close(fig)
    print("Backbone plot saved to:", output_path)
    return xlim, ylim, zlim

def plot_times(times):
    """
    Plots a list of times (from time.perf_counter()) and returns the matplotlib figure and axes.
    
    Parameters:
      times : list or array-like
          A list of time stamps (in seconds) logged using time.perf_counter().
    
    Returns:
      fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
          The figure and axes objects for further customization.
    """
    # Create a figure and axes.
    fig, ax = plt.subplots()
    
    # Plot the times. x-axis is simply the index.
    ax.plot(range(len(times)), times, marker='o', linestyle='-')
    ax.set_yscale('log')
    # Label the axes
    
    return fig, ax


def sorted_bar_plot(values, title="Sorted Bar Plot", ylabel="Value", save_path=None):
    """
    Creates a bar plot from a list of values, sorted in descending order,
    and saves or displays the plot.
    
    Parameters:
        values (list or array-like): The numerical values to plot.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        save_path (str, optional): If provided, the plot is saved to this file path.
    """
    # Sort the values in descending order
    sorted_values = sorted(values, reverse=True)
    indices = list(range(len(sorted_values)))  # Use indices as x positions

    plt.figure(figsize=(8, 5))
    plt.bar(indices, sorted_values, edgecolor='black')
    plt.title(title)
    plt.ylabel(ylabel)
    # Remove x-axis labels
    plt.xticks([])  
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



def plot_feature_importance(mean_attn_scores, labels, output_path):
    # 1) sanity check
    assert len(mean_attn_scores) == len(labels), "scores and labels must be same length"
    
    # 2) create figure
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # 3) horizontal bar plot
    y_pos = range(len(labels))
    ax.barh(y_pos, mean_attn_scores, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # highest scores at top
    ax.set_xlabel("Mean Attention Score")
    ax.set_title("Feature Importance")
    
    # 4) layout, show, and save
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def pareto_efficient_indices(points, sense=("min", "min")):
    """
    Return indices of Pareto efficient points.
    points: array-like shape (n, 2) for (x, y).
    sense: tuple per axis: "min" or "max".
    """
    if len(points) == 0:
        return []
    pts = np.array(points, dtype=float).copy()
    if sense[0] == "max":
        pts[:, 0] = -pts[:, 0]
    if sense[1] == "max":
        pts[:, 1] = -pts[:, 1]
    n = pts.shape[0]
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_eff[i]:
            continue
        dominates = np.all(pts <= pts[i] + 1e-12, axis=1) & np.any(pts < pts[i] - 1e-12, axis=1)
        dominates[i] = False
        if np.any(dominates):
            is_eff[i] = False
    return np.where(is_eff)[0].tolist()


def pareto_efficient_indices_nd(points, sense):
    """
    Return indices of Pareto-efficient rows in `points` (n x m).
    `sense`: list/tuple of length m with "min" or "max" per objective.
    """
    P = np.asarray(points, float).copy()
    for j, s in enumerate(sense):
        if s == "max":
            P[:, j] = -P[:, j]  # convert 'max' to minimizing
    n = P.shape[0]
    eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not eff[i]:
            continue
        dominated = (np.all(P <= P[i] + 1e-12, axis=1) &
                     np.any(P <  P[i] - 1e-12, axis=1))
        dominated[i] = False
        if dominated.any():
            eff[i] = False
    return np.where(eff)[0].tolist()

def set_log_xlim_with_pad(ax, xs, pad_frac=0.10):
    xs = np.asarray(xs, float)
    if xs.size == 0: return
    xmin, xmax = xs.min(), xs.max()
    ax.set_xlim(xmin/(1+pad_frac), xmax*(1+pad_frac))

def human_count(n):
    if n is None: return None
    n = float(n)
    if n >= 1e9: return f"{n/1e9:.0f}B"
    if n >= 1e6: return f"{n/1e6:.0f}M"
    if n >= 1e3: return f"{n/1e3:.0f}K"
    return f"{int(n)}"

def draw_pareto_cloud(ax, xs, ys, color, band_frac=0.06, alpha=0.18, z=1):
    """Shaded 'cloud' around a frontier (sorted by x)."""
    if len(xs) == 0: return
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # de-duplicate x (keep median y)
    uniq_x, inv = np.unique(xs, return_inverse=True)
    agg_y = np.array([np.median(ys[inv==i]) for i in range(len(uniq_x))])
    yrng = max(np.ptp(agg_y), 1e-6)
    band = band_frac * yrng
    upper = agg_y + band
    lower = agg_y - band
    ax.fill_between(uniq_x, lower, upper, color=mcolors.to_rgba(color, alpha),
                    step=None, linewidth=0, zorder=z)

def main():
    parser = argparse.ArgumentParser(
        description="Two-panel Pareto plots with method curves, Pareto Fronts, and method-colored borders."
    )
    parser.add_argument("--out", required=True, help="Output figure path (png/pdf/svg)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-labels", action="store_true", help="Hide method labels")
    args = parser.parse_args()

    # ---------------- Updated data ----------------
    # (family, codebook, train_size, BPR, train_rmsd, train_lddt, cameo_rmsd, cameo_lddt, casp_rmsd, casp_lddt)
    rows = [
        ("VQ-VAE",   128,   None, 395.92, 1.58, 0.73, 3.70, 0.57, 4.86, 0.46),
        ("VQ-VAE",   256,   None, 397.40, 1.50, 0.73, 3.65, 0.57, 4.85, 0.46),
        ("VQ-VAE",   512,  40000, 399.36, 1.53, 0.73, 3.72, 0.57, 4.71, 0.46),
        ("VQ-VAE",  1024,   None, 402.28, 1.40, 0.78, 3.69, 0.60, 5.00, 0.47),

        ("AminoASeed", 128,  None, 399.76, 0.82, 0.84, 2.92, 0.66, 4.03, 0.57),
        ("AminoASeed", 256,  None, 401.24, 0.78, 0.85, 2.90, 0.69, 3.56, 0.60),
        ("AminoASeed", 512, 40000, 403.20, 0.77, 0.85, 2.94, 0.69, 3.80, 0.59),
        ("AminoASeed",1024,  None, 406.11,   None,  None,   None,  None,  None,  None),  # placeholder

        ("ESM3",     4096,   None, 2274.12,  None,  None,  0.91, 0.96, 1.29, 0.93),
        ("FoldSeek",   20,   None, 4.32,     None,  None,   None,  None, None, None),
        ("ProToken",  512, 552000, 132.90,   None,  None,  0.62, 0.92, 0.59, 0.90),

        ("PT-BPE",    600,  40000, 36.02, 1.66, 0.73, 1.77, 0.72, 1.53, 0.72),
        ("PT-BPE",   2500,  40000, 41.11, 1.41, 0.75, 1.57, 0.74, 1.51, 0.73),
        ("PT-BPE",   6000,  40000, 45.44, 1.37, 0.76, 1.52, 0.74, 1.54, 0.72),
        ("PT-BPE",  21000,  40000, 47.62, 1.21, 0.77, 1.40, 0.75, 1.55, 0.72),     # placeholder
    ]

    # Fallback per-family sizes for label parentheses
    fam_train_sizes = {
        "PT-BPE": 40000, "VQ-VAE": 40000, "AminoASeed": 40000,
        "ProToken": 552000, "ESM3": 236_000_000
    }

    # Group by family (method)
    by_fam = {}
    for fam, cb, trn, bpr, tr, tl, cr, cl, pr, pl in rows:
        by_fam.setdefault(fam, []).append(dict(
            codebook=cb, train_sz=trn, bpr=bpr,
            train_rmsd=tr, train_lddt=tl,
            cameo_rmsd=cr, cameo_lddt=cl, casp_rmsd=pr, casp_lddt=pl
        ))

    # Method label (family + training set size)
    def fam_label(fam):
        sz = next((r["train_sz"] for r in by_fam[fam] if r["train_sz"] is not None), None)
        if sz is None: sz = fam_train_sizes.get(fam)
        return f"{fam} ({human_count(sz)})" if sz is not None else fam

    fam_list = list(by_fam.keys())

    # Method edge colors (borders) + label colors
    tableau = list(mcolors.TABLEAU_COLORS.values())
    fam_edge = {fam: tableau[i % len(tableau)] for i, fam in enumerate(fam_list)}
    markeredgewidth = 0
    # Split face colors (test)
    split_face = {"CAMEO": "gold", "CASP14": "silver"}

    # Emphasize PT-BPE
    EMPHASIS_FAM = "PT-BPE"
    EMP_LINE_WIDTH = 2.1
    BASE_LINE_WIDTH = 1.3
    EMP_MARKER_SCALE = 1.25

    # Build per-split series per family
    series = {split: {fam: {"x": [], "r": [], "l": []} for fam in fam_list}
              for split in ("CAMEO", "CASP14")}
    train_pts = {fam: {"x_r": [], "y_r": [], "x_l": [], "y_l": []} for fam in fam_list}

    # Pareto inputs (only if both test splits exist)
    bpr_r, cameo_rmsd, casp_rmsd = [], [], []
    bpr_l, cameo_lddt, casp_lddt = [], [], []

    for fam in fam_list:
        for rec in by_fam[fam]:
            b = rec["bpr"]
            if b is None:  # placeholder
                continue
            # Train triangles
            if rec["train_rmsd"] is not None:
                train_pts[fam]["x_r"].append(b); train_pts[fam]["y_r"].append(rec["train_rmsd"])
            if rec["train_lddt"] is not None:
                train_pts[fam]["x_l"].append(b); train_pts[fam]["y_l"].append(rec["train_lddt"])
            # Test points
            if rec["cameo_rmsd"] is not None:
                series["CAMEO"][fam]["x"].append(b)
                series["CAMEO"][fam]["r"].append(rec["cameo_rmsd"])
                series["CAMEO"][fam]["l"].append(rec["cameo_lddt"])
            if rec["casp_rmsd"] is not None:
                series["CASP14"][fam]["x"].append(b)
                series["CASP14"][fam]["r"].append(rec["casp_rmsd"])
                series["CASP14"][fam]["l"].append(rec["casp_lddt"])
            # Pareto if both splits exist for this configuration
            if (rec["cameo_rmsd"] is not None) and (rec["casp_rmsd"] is not None):
                bpr_r.append(b); cameo_rmsd.append(rec["cameo_rmsd"]); casp_rmsd.append(rec["casp_rmsd"])
            if (rec["cameo_lddt"] is not None) and (rec["casp_lddt"] is not None):
                bpr_l.append(b); cameo_lddt.append(rec["cameo_lddt"]); casp_lddt.append(rec["casp_lddt"])

    bpr_r, cameo_rmsd, casp_rmsd = map(lambda a: np.asarray(a, float), (bpr_r, cameo_rmsd, casp_rmsd))
    bpr_l, cameo_lddt, casp_lddt = map(lambda a: np.asarray(a, float), (bpr_l, cameo_lddt, casp_lddt))

    # Pareto (3D)
    eff_r_idx = pareto_efficient_indices_nd(np.c_[bpr_r, cameo_rmsd, casp_rmsd],
                                            sense=("min","min","min")) if bpr_r.size else []
    eff_l_idx = pareto_efficient_indices_nd(np.c_[bpr_l, cameo_lddt, casp_lddt],
                                            sense=("min","max","max")) if bpr_l.size else []

    # ---------- Figure ----------
    plt.rcParams.update({
        "figure.dpi": args.dpi, "font.size": 10.5,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10
    })
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    for ax in (ax_r, ax_l):
        ax.set_xscale("log")
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="both", lw=0.6, ls=":", alpha=0.35)
        ax.grid(True, which="minor", axis="x", lw=0.4, ls=":", alpha=0.2)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_xlabel("BPR")
    halo = [PathEffects.Stroke(linewidth=2.4, foreground="white"), PathEffects.Normal()]

    # ------- Left: RMSD vs BPR -------
    ax_r.set_ylabel("RMSD")

    # Pareto Fronts (test only)
    if len(eff_r_idx):
        order = np.argsort(bpr_r[eff_r_idx])
        draw_pareto_cloud(ax_r, bpr_r[eff_r_idx][order], cameo_rmsd[eff_r_idx][order],
                          split_face["CAMEO"], band_frac=0.06, alpha=0.18, z=1)
        draw_pareto_cloud(ax_r, bpr_r[eff_r_idx][order], casp_rmsd[eff_r_idx][order],
                          split_face["CASP14"], band_frac=0.06, alpha=0.18, z=1)

    # Train triangles (smaller; edge = method color, hollow fill)
    for fam in fam_list:
        if len(train_pts[fam]["x_r"]):
            ax_r.scatter(train_pts[fam]["x_r"], train_pts[fam]["y_r"],
                         s=55, marker="^", facecolor=fam_edge[fam], edgecolor=fam_edge[fam],
                         linewidth=markeredgewidth, alpha=0.95, zorder=3)

    # Method curves + test circles (edge = method color, face = split)
    for split in ("CAMEO", "CASP14"):
        for fam in fam_list:
            x = np.array(series[split][fam]["x"], float)
            y = np.array(series[split][fam]["r"], float)
            if x.size == 0: continue
            o = np.argsort(x); x, y = x[o], y[o]
            lw = EMP_LINE_WIDTH if fam == EMPHASIS_FAM else BASE_LINE_WIDTH
            z = 3 if fam == EMPHASIS_FAM else 2
            # ax_r.plot(x, y, color=split_face[split], lw=lw, alpha=0.9, zorder=z)
            size = 45*(EMP_MARKER_SCALE if fam == EMPHASIS_FAM else 1.0)  # smaller markers
            ax_r.scatter(x, y, s=size, marker="o" if split == "CAMEO" else "s", facecolor=fam_edge[fam],
                         edgecolor=fam_edge[fam], linewidth=markeredgewidth, alpha=0.96, zorder=z+0.1)

    # Labels (one per method, colored by method)
    if not args.no_labels:
        for fam in fam_list:
            xs = []; ys = []
            for split in ("CAMEO","CASP14"):
                xs += list(series[split][fam]["x"])
                ys += list(series[split][fam]["r"])
            if not xs: continue
            i = int(np.argmax(xs))
            ax_r.annotate(fam_label(fam), (xs[i], ys[i]), xytext=(6, 8),
                          textcoords="offset points", ha="left", va="bottom",
                          color=fam_edge[fam], fontsize=10,
                          path_effects=halo, zorder=4)

    # Legends: split (fill color semantics) + methods (edge color semantics)
    split_handles_r = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor="white",
               markeredgecolor='black', markersize=7, label='CAMEO (test)'),
        Line2D([0],[0], marker='s', color='none', markerfacecolor="white",
               markeredgecolor='black', markersize=7, label='CASP14 (test)'),
        Line2D([0],[0], marker='^', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=7, label='Train'),
        Line2D([0],[0], color=mcolors.to_rgba(split_face["CAMEO"],0.35), lw=6, label='Pareto Front (CAMEO, ○)'),
        Line2D([0],[0], color=mcolors.to_rgba(split_face["CASP14"],0.35), lw=6, label='Pareto Front (CASP14, □)'),
    ]
    lg1 = ax_r.legend(handles=split_handles_r, title="RMSD", loc="upper left",
                      frameon=True, framealpha=0.9, fancybox=True, edgecolor="0.85")
    ax_r.add_artist(lg1)

    # # --- Methods legend (RMSD panel) ---
    # method_handles = [
    #     Line2D([0],[0], marker='o', color='none', markerfacecolor='none',
    #         markeredgecolor=fam_edge[f], markeredgewidth=2, markersize=8,
    #         label=fam_label(f))
    #     for f in fam_list
    # ]
    # ax_r.legend(
    #     handles=method_handles,
    #     title="Methods (edge color)\nParentheses = train data size",
    #     loc="upper left", frameon=True, framealpha=0.9,
    #     fancybox=True, edgecolor="0.85", ncol=2
    # )

    # # --- Methods legend (LDDT panel) ---
    # method_handles_l = [
    #     Line2D([0],[0], marker='o', color='none', markerfacecolor='none',
    #         markeredgecolor=fam_edge[f], markeredgewidth=2, markersize=8,
    #         label=fam_label(f))
    #     for f in fam_list
    # ]
    # ax_l.legend(
    #     handles=method_handles_l,
    #     title="Methods (edge color)\nParentheses = train data size",
    #     loc="upper left", frameon=True, framealpha=0.9,
    #     fancybox=True, edgecolor="0.85", ncol=2
    # )

    # Limits
    all_x_r = [xx for s in ("CAMEO","CASP14") for f in fam_list for xx in series[s][f]["x"]]
    set_log_xlim_with_pad(ax_r, np.array(all_x_r, float), pad_frac=0.12)
    all_y_r = ([y for f in fam_list for y in train_pts[f]["y_r"]] +
               [y for s in ("CAMEO","CASP14") for f in fam_list for y in series[s][f]["r"]])
    if all_y_r:
        ymin, ymax = min(all_y_r), max(all_y_r)
        rng = max(ymax - ymin, 1e-6)
        ax_r.set_ylim(max(0, ymin - 0.12*rng), ymax + 0.18*rng)
    ax_r.set_title("RMSD vs BPR")

    # ------- Right: LDDT vs BPR -------
    ax_l.set_ylabel("LDDT")

    # Pareto Fronts (test only)
    if len(eff_l_idx):
        order = np.argsort(bpr_l[eff_l_idx])
        draw_pareto_cloud(ax_l, bpr_l[eff_l_idx][order], cameo_lddt[eff_l_idx][order],
                          split_face["CAMEO"], band_frac=0.06, alpha=0.18, z=1)
        draw_pareto_cloud(ax_l, bpr_l[eff_l_idx][order], casp_lddt[eff_l_idx][order],
                          split_face["CASP14"], band_frac=0.06, alpha=0.18, z=1)

    # Train triangles
    for fam in fam_list:
        if len(train_pts[fam]["x_l"]):
            ax_l.scatter(train_pts[fam]["x_l"], train_pts[fam]["y_l"],
                         s=55, marker="^", edgecolor=fam_edge[fam], facecolor=fam_edge[fam],
                         linewidth=markeredgewidth, alpha=0.95, zorder=3)

    # Method curves + test circles
    for split in ("CAMEO", "CASP14"):
        for fam in fam_list:
            x = np.array(series[split][fam]["x"], float)
            y = np.array(series[split][fam]["l"], float)
            if x.size == 0: continue
            o = np.argsort(x); x, y = x[o], y[o]
            lw = EMP_LINE_WIDTH if fam == EMPHASIS_FAM else BASE_LINE_WIDTH
            z = 3 if fam == EMPHASIS_FAM else 2
            # ax_l.plot(x, y, color=split_face[split], lw=lw, alpha=0.9, zorder=z)
            size = 45*(EMP_MARKER_SCALE if fam == EMPHASIS_FAM else 1.0)
            ax_l.scatter(x, y, s=size, marker="o" if split == "CAMEO" else "s", facecolor=fam_edge[fam],
                         edgecolor=fam_edge[fam], linewidth=markeredgewidth, alpha=0.96, zorder=z+0.1)

    # Labels
    if not args.no_labels:
        for fam in fam_list:
            xs = []; ys = []
            for split in ("CAMEO","CASP14"):
                xs += list(series[split][fam]["x"])
                ys += list(series[split][fam]["l"])
            if not xs: continue
            i = int(np.argmax(xs))
            ax_l.annotate(fam_label(fam), (xs[i], ys[i]), xytext=(6, 8),
                          textcoords="offset points", ha="left", va="bottom",
                          color=fam_edge[fam], fontsize=10,
                          path_effects=halo, zorder=4)

    # Legends
    split_handles_l = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor="white",
               markeredgecolor='black', markersize=7, label='CAMEO (test)'),
        Line2D([0],[0], marker='s', color='none', markerfacecolor="white",
               markeredgecolor='black', markersize=7, label='CASP14 (test)'),
        Line2D([0],[0], marker='^', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=7, label='Train'),
        Line2D([0],[0], color=mcolors.to_rgba(split_face["CAMEO"],0.35), lw=6, label='Pareto Front (CAMEO, ○)'),
        Line2D([0],[0], color=mcolors.to_rgba(split_face["CASP14"],0.35), lw=6, label='Pareto Front (CASP14, □)'),
    ]
    lg2 = ax_l.legend(handles=split_handles_l, title="LDDT", loc="best",
                      frameon=True, framealpha=0.9, fancybox=True, edgecolor="0.85")
    ax_l.add_artist(lg2)

    # Limits
    all_x_l = [xx for s in ("CAMEO","CASP14") for f in fam_list for xx in series[s][f]["x"]]
    set_log_xlim_with_pad(ax_l, np.array(all_x_l, float), pad_frac=0.12)

    all_y_l = ([y for f in fam_list for y in train_pts[f]["y_l"]] +
               [y for s in ("CAMEO","CASP14") for f in fam_list for y in series[s][f]["l"]])
    if all_y_l:
        ymin, ymax = min(all_y_l), max(all_y_l)
        rng = max(ymax - ymin, 1e-6)
        ax_l.set_ylim(max(0, ymin - 0.12*rng), min(1.05, ymax + 0.18*rng))
    ax_l.set_title("LDDT vs BPR")

    # Share same BPR span
    xmin = min(ax_r.get_xlim()[0], ax_l.get_xlim()[0])
    xmax = max(ax_r.get_xlim()[1], ax_l.get_xlim()[1])
    ax_r.set_xlim(xmin, xmax); ax_l.set_xlim(xmin, xmax)
    fig.text(
        0.5, 0.01,
        "Note: method labels show training dataset size in parentheses (e.g., 40K = 40,000).",
        ha="center", va="bottom", fontsize=9, color="0.35"
    )
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.out}")

    
if __name__ == "__main__":
    # plot_losses(sys.argv[1], out_fname=sys.argv[2], simple=True)
    main()