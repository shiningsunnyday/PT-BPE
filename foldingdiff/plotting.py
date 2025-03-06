"""
Utility functions for plotting
"""
import os, sys
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerBase
import matplotlib.cm as cm
import seaborn as sns

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
                        color=orig_handle.color, lw=2) for i in range(n - 1)]

        # Combine all artists.
        artists = circles + lines

        # Apply the transformation.
        for artist in artists:
            artist.set_transform(trans)
        return artists


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


def save_circular_histogram(angles, path=None, bins=10, title=None):
    """
    Creates a circular (rose) histogram from a list of angle values (in radians, range -pi to pi)
    and saves the figure at the specified file path.
    
    Parameters:
        angles (list or array-like): Angle values in radians, expected in the range [-pi, pi].
        path (str): The file path where the figure will be saved.
        bins (int, optional): Number of bins for the histogram. Defaults to 10.
    """
    # Convert angles from [-pi, pi] to [0, 2*pi]
    angles = np.array(angles)
    angles = (angles + 2*np.pi) % (2*np.pi)
    
    # Compute histogram data: counts and bin edges over [0, 2pi]
    counts, bin_edges = np.histogram(angles, bins=bins, range=(0, 2*np.pi))
    widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + widths / 2
    if path is None:
        return bin_centers, widths, counts

    # Create a polar subplot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # Create a bar plot for each bin in the polar coordinate system
    ax.bar(bin_centers, counts, width=widths, bottom=0.0, edgecolor='black', align='center')    
    ax.set_title(title if title else "Circular Histogram (Rose Diagram)")
    plt.savefig(path, bbox_inches='tight')
    print("Histogram saved at", os.path.abspath(path))
    plt.close()


def legend_key_to_tuple(label):
    tup = []
    for sublabel in label.split():
        if sublabel.isdigit():
            tup.append(int(sublabel))
        else:
            tup.append(sublabel)
    return tuple(tup)


def plot_backbone(coords, output_path, atom_types, tokens=None, title=""):
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
          Each token will be drawn in black (you can change the color scheme as desired)
          and each individual bond segment is annotated with its token label (bt).
          If None, the entire backbone is drawn in red.
      title : str, optional
          Title for the plot.
    """
    coords = np.asarray(coords)
    
    # Create a larger 3D plot for zooming in.
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert atom_types to a numpy array for boolean indexing.
    atom_types_arr = np.array(atom_types)
    
    # Scatter plot the atoms by type.
    N_coords = coords[atom_types_arr == 'N']
    CA_coords = coords[atom_types_arr == 'CA']
    C_coords = coords[atom_types_arr == 'C']
    
    ax.scatter(N_coords[:, 0], N_coords[:, 1], N_coords[:, 2],
               color='blue', s=20, label='N')
    ax.scatter(CA_coords[:, 0], CA_coords[:, 1], CA_coords[:, 2],
               color='orange', s=20, label='CA')
    ax.scatter(C_coords[:, 0], C_coords[:, 1], C_coords[:, 2],
               color='green', s=20, label='C')
    
    # Plot bonds using tokens if provided.
    token_lookup = {}
    if tokens is not None:
        cmap = cm.get_cmap('tab10')
        for token in tokens:
            start, bt, length = token
            token_coords = coords[start: start + length + 1]
            # Draw each bond segment in the token.
            for i in range(length):
                segment = token_coords[i:i+2]
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                        color=cmap(bt%10), 
                        label=f"Token {bt}",
                        linewidth=2)
                # Calculate the midpoint of the segment for annotation.
                # midpoint = segment.mean(axis=0)
                # txt = ax.text(midpoint[0], midpoint[1], midpoint[2],
                #               f'{bt}', color='black', fontweight='bold', fontsize=6)
                # txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        # Highlight the boundary atoms between tokens.
        boundary_marker_size = 50  # adjust as needed
        for token in tokens:
            start, bt, length = token
            if bt not in token_lookup:
                token_lookup[bt] = [{'N':'blue','CA':'orange','C':'green'}[atom_types[index]] for index in range(start, start+length+1)]
            # The boundary atoms for the token are at index start and start+length.
            left_atom = coords[start]
            right_atom = coords[start + length]
            ax.scatter(left_atom[0], left_atom[1], left_atom[2],
                       color='black', s=boundary_marker_size, zorder=10)
            ax.scatter(right_atom[0], right_atom[1], right_atom[2],
                       color='black', s=boundary_marker_size, zorder=10)                
    else:
        # Plot the entire backbone as one red line.
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color='red', linewidth=2)
    
    # Compute the shortest bond length (for all consecutive atoms)
    bond_lengths = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    min_bond_length = bond_lengths.min()
    r = 0.3 * min_bond_length  # radius for the arc
    
    # Annotate each bond-bond joint with its angle.
    # For each internal atom (skipping the first and last)
    for i in range(1, coords.shape[0]-1):
        joint = coords[i]
        v1 = coords[i-1] - joint  # direction toward the previous atom
        v2 = coords[i+1] - joint  # direction toward the next atom
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
        u1 = v1 / norm1
        u2 = v2 / norm2
        dot_val = np.dot(u1, u2)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        angle = np.arccos(dot_val)
        
        # Compute a perpendicular vector in the plane of u1 and u2.
        w = u2 - np.dot(u1, u2)*u1
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            continue
        w = w / norm_w
        
        # Generate arc points along the circle of radius r centered at the joint.
        t_vals = np.linspace(0, angle, 50)
        arc_points = np.array([joint + r*(np.cos(t)*u1 + np.sin(t)*w) for t in t_vals])
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                color='black', linewidth=2)
        
        # Compute the midpoint of the arc (at t = angle/2).
        mid_point = joint + r*(np.cos(angle/2)*u1 + np.sin(angle/2)*w)
        
        # Annotate the angle (in radians, formatted to 2 decimals) at the arc midpoint.
        txt = ax.text(mid_point[0], mid_point[1], mid_point[2],
                      f"{angle:.2f}", color='black', fontweight='bold', fontsize=4)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Set axis labels.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Adjust the view angle.
    ax.view_init(elev=30, azim=45)
    
    # Optionally, set a title.
    if title:
        ax.set_title(title)
    
    # Compute coordinate bounds and set tighter limits to "zoom in".
    x_min, x_max = -10, 20 # coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = -15, 20 # coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = -20, 15 # coords[:, 2].min(), coords[:, 2].max()
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    zoom_factor = 0.7  # adjust this value as needed; smaller => more zoom
    new_range = max_range * zoom_factor
    ax.set_xlim(x_mid - new_range/2, x_mid + new_range/2)
    ax.set_ylim(y_mid - new_range/2, y_mid + new_range/2)
    ax.set_zlim(z_mid - new_range/2, z_mid + new_range/2)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in sorted(zip(handles, labels), key=lambda it: legend_key_to_tuple(it[1])):
        if label not in unique:
            unique[label] = handle
    handles = list(unique.values())
    labels = list(unique.keys())
    if tokens is not None:
        # For each token, create one dummy handle using our BackboneFragment class.
        for i, label in enumerate(labels):
            token_id_match = re.match('Token (\d+)', label)
            if token_id_match is None:
                continue
            bt = int(token_id_match.groups()[0])
            color = cmap(bt%10)
            handles[i] = BackboneFragment(bt, token_lookup[bt], color)
        ax.legend(handles, labels,
                  handler_map={BackboneFragment: HandlerBackboneFragment()})
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print("Backbone plot saved to:", output_path)



def save_histogram(values, path, bins=10, title="", return_ax=False):
    """
    Creates a histogram from a list of values and saves it to the specified file path.
    
    Parameters:
        values (list or array-like): The numerical values to plot.
        path (str): The file path where the figure will be saved.
        bins (int, optional): Number of bins for the histogram. Defaults to 10.
        title (str, optional): Title for the histogram. Defaults to "Histogram" if not provided.
        return_ax (bool, optional): If True, returns the matplotlib Axes object. Defaults to False.
    
    Returns:
        ax (matplotlib.axes.Axes): The Axes object (only if return_ax is True).
    """
    # Create a figure and an axes.
    fig, ax = plt.subplots()
    
    # Plot histogram using the axes object.
    ax.hist(values, bins=bins, edgecolor='black')
    
    # Set title, xlabel, and ylabel.
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
        
    # Return the axes if requested; otherwise, close the figure.
    if return_ax:
        return fig, ax
    else:
        # Save the figure.
        fig.savefig(path)        
        plt.close(fig)


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


if __name__ == "__main__":
    plot_losses(sys.argv[1], out_fname=sys.argv[2], simple=True)
