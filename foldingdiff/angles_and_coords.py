"""
Code to convert from angles between residues to XYZ coordinates. 
"""
import functools
import gzip
import os
import logging
import glob
from collections import namedtuple, defaultdict
from itertools import groupby
from typing import *
import warnings
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence

from foldingdiff import nerf
import torch

EXHAUSTIVE_ANGLES = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
EXHAUSTIVE_DISTS = ["0C:1N", "N:CA", "CA:C"]
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
    "THR", "TRP", "TYR", "VAL"
}
# --- 3‑letter ➜ 1‑letter lookup table (IUPAC) -------------------------------
_AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common “unknown / modified” fall‑backs
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
    "XAA": "X", "UNK": "X"
}
STANDARD_SIDECHAIN_ORDER = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "VAL": ["CB", "CG1", "CG2"]
}
MINIMAL_ANGLES = ["phi", "psi", "omega"]
MINIMAL_DISTS = []


def canonical_distances_and_dihedrals(
    fname: str,
    distances: List[str] = MINIMAL_DISTS,
    angles: List[str] = MINIMAL_ANGLES,
) -> Optional[pd.DataFrame]:    
    """Parse the pdb file for the given values"""
    assert os.path.isfile(fname)
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true_div.*")
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        print(fname, "has multiple models")
        # return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure(model=1)
    # First get the dihedrals
    try:        
        phi, psi, omega = struc.dihedral_backbone(source_struct)
        calc_angles = {"phi": phi, "psi": psi, "omega": omega}
        if (psi==psi).sum() < len(psi)-1:
            phi, psi, omega = struc.dihedral_backbone(source_struct)
    except struc.BadStructureError:
        logging.debug(f"{fname} contains a malformed structure - skipping")
        return None

    # Get any additional angles
    non_dihedral_angles = [a for a in angles if a not in calc_angles]
    # Gets the N - CA - C for each residue
    # https://www.biotite-python.org/apidoc/biotite.structure.filter_backbone.html
    backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
    for a in non_dihedral_angles:
        if a == "tau" or a == "N:CA:C":
            # tau = N - CA - C internal angles
            r = np.arange(3, len(backbone_atoms), 3)
            idx = np.hstack([np.vstack([r, r + 1, r + 2]), np.zeros((3, 1))]).T
        elif a == "CA:C:1N":  # Same as C-N angle in nerf
            # This measures an angle between two residues. Due to the way we build
            # proteins out later, we do not need to meas
            r = np.arange(0, len(backbone_atoms) - 3, 3)
            idx = np.hstack([np.vstack([r + 1, r + 2, r + 3]), np.zeros((3, 1))]).T
        elif a == "C:1N:1CA":
            r = np.arange(0, len(backbone_atoms) - 3, 3)
            idx = np.hstack([np.vstack([r + 2, r + 3, r + 4]), np.zeros((3, 1))]).T
        else:
            raise ValueError(f"Unrecognized angle: {a}")
        try:
            calc_angles[a] = struc.index_angle(backbone_atoms, indices=idx.astype(int))
        except IndexError:
            logging.debug(f"index_angle raised IndexError - skipping")
            return None

    # At this point we've only looked at dihedral and angles; check value range
    for k, v in calc_angles.items():
        if not (np.nanmin(v) >= -np.pi and np.nanmax(v) <= np.pi):
            logging.warning(f"Illegal values for {k} in {fname} -- skipping")
            return None

    # Get any additional distances
    for d in distances:
        if (d == "0C:1N") or (d == "C:1N"):
            # Since this is measuring the distance between pairs of residues, there
            # is one fewer such measurement than the total number of residues like
            # for dihedrals. Therefore, we pad this with a null 0 value at the end.
            r = np.arange(0, len(backbone_atoms) - 3, 3)
            idx = np.hstack([np.vstack([r + 2, r + 3]), np.zeros((2, 1))]).T
        elif d == "N:CA":
            # We start resconstructing with a fixed initial residue so we do not need
            # to predict or record the initial distance. Additionally we pad with a
            # null value at the end
            r = np.arange(3, len(backbone_atoms), 3)
            idx = np.hstack([np.vstack([r, r + 1]), np.zeros((2, 1))]).T
            assert len(idx) == len(calc_angles["phi"])
        elif d == "CA:C":
            # We start reconstructing with a fixed initial residue so we do not need
            # to predict or record the initial distance. Additionally, we pad with a
            # null value at the end.
            r = np.arange(3, len(backbone_atoms), 3)
            idx = np.hstack([np.vstack([r + 1, r + 2]), np.zeros((2, 1))]).T
            assert len(idx) == len(calc_angles["phi"])
        else:
            raise ValueError(f"Unrecognized distance: {d}")
        calc_angles[d] = struc.index_distance(backbone_atoms, indices=idx.astype(int))

    return pd.DataFrame({k: calc_angles[k].squeeze() for k in distances + angles})


def create_new_chain_nerf(
    out_fname: str,
    dists_and_angles: pd.DataFrame,
    angles_to_set: Optional[List[str]] = None,
    dists_to_set: Optional[List[str]] = None,
    center_coords: bool = True,
) -> str:
    """
    Create a new chain using NERF to convert to cartesian coordinates. Returns
    the path to the newly create file if successful, empty string if fails.
    """
    if angles_to_set is None and dists_to_set is None:
        angles_to_set, dists_to_set = [], []
        for c in dists_and_angles.columns:
            # Distances are always specified using one : separating two atoms
            # Angles are defined either as : separating 3+ atoms, or as names
            if c.count(":") == 1:
                dists_to_set.append(c)
            else:
                angles_to_set.append(c)
        logging.debug(f"Auto-determined setting {dists_to_set, angles_to_set}")
    else:
        assert angles_to_set is not None
        assert dists_to_set is not None

    # Check that we are at least setting the dihedrals
    required_dihedrals = ["phi", "psi", "omega"]
    assert all([a in angles_to_set for a in required_dihedrals])

    nerf_build_kwargs = dict(
        phi_dihedrals=dists_and_angles["phi"],
        psi_dihedrals=dists_and_angles["psi"],
        omega_dihedrals=dists_and_angles["omega"],
    )
    for a in angles_to_set:
        if a in required_dihedrals:
            continue
        assert a in dists_and_angles
        if a == "tau" or a == "N:CA:C":
            nerf_build_kwargs["bond_angle_ca_c"] = dists_and_angles[a]
        elif a == "CA:C:1N":
            nerf_build_kwargs["bond_angle_c_n"] = dists_and_angles[a]
        elif a == "C:1N:1CA":
            nerf_build_kwargs["bond_angle_n_ca"] = dists_and_angles[a]
        else:
            raise ValueError(f"Unrecognized angle: {a}")

    for d in dists_to_set:
        assert d in dists_and_angles.columns
        if d == "0C:1N":
            nerf_build_kwargs["bond_len_c_n"] = dists_and_angles[d]
        elif d == "N:CA":
            nerf_build_kwargs["bond_len_n_ca"] = dists_and_angles[d]
        elif d == "CA:C":
            nerf_build_kwargs["bond_len_ca_c"] = dists_and_angles[d]
        else:
            raise ValueError(f"Unrecognized distance: {d}")

    nerf_builder = nerf.NERFBuilder(**nerf_build_kwargs)
    coords = (
        nerf_builder.centered_cartesian_coords
        if center_coords
        else nerf_builder.cartesian_coords
    )
    if np.any(np.isnan(coords)):
        logging.warning(f"Found NaN values, not writing pdb file {out_fname}")
        return ""

    assert coords.shape == (
        int(dists_and_angles.shape[0] * 3),
        3,
    ), f"Unexpected shape: {coords.shape} for input of {len(dists_and_angles)}"
    return write_coords_to_pdb(coords, out_fname)


def rotate_vector(v, k, angle):
    return (v * np.cos(angle) +
            np.cross(k, v) * np.sin(angle) +
            k * np.dot(k, v) * (1 - np.cos(angle)))


def update_backbone_positions(N_init, CA_init, C_init, L_CA_C, L_N_CA, new_theta):
    """
    Update the positions of the backbone atoms (N, CA, C) of the first residue.
    
    Procedure:
      1. C remains fixed.
      2. CA_new is placed on the original CA→C line at distance L_CA_C from C.
      3. Compute the original bond vector from the new CA (i.e. use N_init - CA_new)
         and determine the current N–CA–C angle measured at CA_new.
      4. Determine the rotation dtheta = new_theta - (current angle) about the axis defined 
         by the normal to the plane spanned by (N_init - CA_new) and (C_init - CA_new).
      5. Rotate the N vector accordingly, rescale to L_N_CA, and add to CA_new to get N_new.
    
    Parameters:
      N_init : array-like, shape (3,)
          Initial coordinates of the N atom.
      CA_init : array-like, shape (3,)
          Initial coordinates of the CA atom.
      C_init : array-like, shape (3,)
          Initial coordinates of the C atom (remains fixed).
      L_CA_C : float
          New CA–C bond length.
      L_N_CA : float
          New N–CA bond length.
      new_theta : float
          New N–CA–C bond angle in radians.
    
    Returns:
      N_new, CA_new, C_new : tuple of numpy arrays
          Updated coordinates for the N, CA, and C atoms.
    """
    import numpy as np

    # Convert inputs to numpy arrays
    N_init = np.array(N_init, dtype=float)
    CA_init = np.array(CA_init, dtype=float)
    C_init = np.array(C_init, dtype=float)
    
    # Step 1: Update CA position: place CA_new on the C->CA_init line at distance L_CA_C from C.
    v = CA_init - C_init
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("C_init and CA_init are identical.")
    v = v / norm_v
    CA_new = C_init + L_CA_C * v

    # Step 2: Recenter the N vector with respect to the new CA position.
    vec_N = N_init - CA_new
    vec_C = C_init - CA_new
    norm_N = np.linalg.norm(vec_N)
    norm_C = np.linalg.norm(vec_C)
    if norm_N == 0 or norm_C == 0:
        raise ValueError("Invalid geometry: one of the bond vectors is zero-length.")
    
    # Compute the current N–CA–C angle at the updated CA position.
    cos_theta = np.dot(vec_N, vec_C) / (norm_N * norm_C)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    current_theta = np.arccos(cos_theta)
    
    # Compute the rotation angle needed: dtheta = new_theta - current_theta.
    dtheta = new_theta - current_theta
    
    # Step 3: Define the rotation axis: the normal to the plane defined by vec_N and vec_C.
    axis = np.cross(vec_N, vec_C)
    norm_axis = np.linalg.norm(axis)
    if norm_axis == 0:
        raise ValueError("The points are collinear; cannot define a unique plane.")
    axis = axis / norm_axis
    
    # Step 4: Rotate vec_N by dtheta around the computed axis using Rodrigues' formula.
    rotated_vec = rotate_vector(vec_N, axis, -dtheta)
    
    # Step 5: Rescale the rotated vector to have length L_N_CA and compute N_new.
    rotated_vec = rotated_vec / np.linalg.norm(rotated_vec) * L_N_CA
    N_new = CA_new + rotated_vec
    
    # C remains unchanged.
    C_new = C_init.copy()
    
    return N_new, CA_new, C_new

def write_coords_to_pdb(coords: np.ndarray, out_fname: str) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    assert len(coords) % 3 == 0, f"Expected 3N coords, got {len(coords)}"
    atoms = []
    for i, (n_coord, ca_coord, c_coord) in enumerate(
        (coords[j : j + 3] for j in range(0, len(coords), 3))
    ):
        atom1 = struc.Atom(
            n_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 1,
            res_name="GLY",
            atom_name="N",
            element="N",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom2 = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 2,
            res_name="GLY",
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom3 = struc.Atom(
            c_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 3,
            res_name="GLY",
            atom_name="C",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atoms.extend([atom1, atom2, atom3])
    full_structure = struc.array(atoms)

    # Add bonds
    full_structure.bonds = struc.BondList(full_structure.array_length())
    indices = list(range(full_structure.array_length()))
    for a, b in zip(indices[:-1], indices[1:]):
        full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)

    # Annotate secondary structure using CA coordinates
    # https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
    # https://academic.oup.com/bioinformatics/article/13/3/291/423201
    # a = alpha helix, b = beta sheet, c = coil
    # ss = struc.annotate_sse(full_structure, "A")
    # full_structure.set_annotation("secondary_structure_psea", ss)

    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)
    return out_fname


@functools.lru_cache(maxsize=8192)
def get_pdb_length(fname: str) -> int:
    """
    Get the length of the chain described in the PDB file
    """
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    structure = PDBFile.read(fname)
    if structure.get_model_count() > 1:
        return -1
    chain = structure.get_structure()[0]
    backbone = chain[struc.filter_backbone(chain)]
    l = int(len(backbone) / 3)
    return l


def is_valid_atom_array(lst):
    if len(lst) % 3 != 0:
        return False
    # if lst[0] < 0: # negative residue idx
    #     return False
    for i in range(len(lst) // 3):
        if not (lst[i*3] == lst[i*3+1] == lst[i*3+2]):
            return False
        if not (i == 0 or lst[i*3] > lst[(i-1)*3]):
            return False
    return True


def extract_backbone_residue_idxes(
    fname: str, atoms: Collection[Literal["N", "CA", "C"]] = ["CA"]
) -> Optional[np.ndarray]:
    """Extract the atom idxes of the alpha carbons"""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        structure = PDBFile.read(f)
    # if structure.get_model_count() > 1:
    #     return None
    chain = structure.get_structure(1, extra_fields=["atom_id"])
    backbone = chain[struc.filter_backbone(chain)]
    ca = [c for c in backbone if c.atom_name in atoms]
    idxes = [c.res_id for c in backbone if c.atom_name in atoms]
    # has 3 atoms per residue and residue id is increasing
    if not is_valid_atom_array(idxes):
        return None
    return idxes



def extract_backbone_coords(
    fname: str, atoms: Collection[Literal["N", "CA", "C"]] = ["CA"]
) -> Optional[np.ndarray]:
    """Extract the coordinates of the alpha carbons"""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        structure = PDBFile.read(f)
    # if structure.get_model_count() > 1:
    #     return None
    chain = structure.get_structure(1)
    backbone = chain[struc.filter_backbone(chain)]
    ca = [c for c in backbone if c.atom_name in atoms]
    coords = [c.coord for c in ca]
    if len(coords) == 0:
        return None
    coords = np.vstack(coords)
    return coords


def aa_seq_from_backbone(backbone_atoms):
    """
    Extract a 1‑letter amino‑acid sequence from a backbone‑only AtomArray.
    Works for single‑model PDB domain files such as those in CATH.
    """
    # If the PDB lacks insertion codes Biotite leaves out the annotation.
    has_ic = "ins_code" in backbone_atoms.get_annotation_categories()
    ins_codes = backbone_atoms.ins_code if has_ic else [""] * backbone_atoms.array_length()

    seq_by_res = OrderedDict()  # preserves residue order
    for ch_id, res_id, ic, res_name in zip(
            backbone_atoms.chain_id,
            backbone_atoms.res_id,
            ins_codes,
            backbone_atoms.res_name):

        key = (ch_id, int(res_id), ic)
        if key in seq_by_res:
            continue                           # already stored this residue
        seq_by_res[key] = _AA3_TO_AA1.get(res_name.strip().upper(), "X")

    return "".join(seq_by_res.values())
    


def extract_c_beta_coords(fname):
    """
    Return an (N,3) float array of Cβ coordinates—one row per residue—
    in the same order as extract_aa_seq. Missing Cβs get [nan, nan, nan].
    """
    # Open PDB (or .pdb.gz) and read first model
    opener = gzip.open if str(fname).endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        pdb = PDBFile.read(f)
    if pdb.get_model_count() > 1:
        print(f"Warning: {fname} has multiple models; using model 1")
    atom_stack = pdb.get_structure(model=1)
    # If we got a stack, take the first (only) frame
    atoms = atom_stack if isinstance(atom_stack, struc.AtomArray) else atom_stack[0]

    # 1) Replicate extract_aa_seq’s residue ordering
    bb_mask = struc.filter_backbone(atoms)
    backbone = atoms[bb_mask]
    has_ic = "ins_code" in backbone.get_annotation_categories()
    ins_codes = backbone.ins_code if has_ic else [""] * backbone.array_length()

    # Ordered dict to preserve (chain, res_id, ins_code) in appearance order
    seq_keys = OrderedDict()
    for ch, res_id, ic in zip(backbone.chain_id, backbone.res_id, ins_codes):
        key = (ch, int(res_id), ic)
        if key not in seq_keys:
            seq_keys[key] = None

    # 2) For each residue key, find its CB atom in the full atom array
    #    and record coordinates
    cb_coords = []
    # categorize quickly
    cat = atoms.get_annotation_categories()
    for (ch, res_id, ic) in seq_keys:
        m = (atoms.chain_id == ch) & (atoms.res_id == res_id)
        if has_ic:
            m &= (atoms.ins_code == ic)
        # select CB only
        m &= (atoms.atom_name == "CB")
        if np.any(m):
            idx = np.argmax(m)  # first occurrence
            coord = atoms.coord[idx]
        else:
            coord = np.array([np.nan, np.nan, np.nan], dtype=float)
        cb_coords.append(coord)

    return np.vstack(cb_coords)


    

def extract_aa_seq(fname):
    """
    Return the amino‑acid sequence (1‑letter string) found in `pdb_file`.

    Parameters
    ----------
    pdb_file : str | pathlib.Path | io.TextIOBase
        Path to a PDB file or an already opened file handle.
    chain_id : str | None
        Desired chain ID (e.g. "A").  If None, all chains are concatenated
        in the order they first appear.

    Notes
    -----
    • Only ATOM records are inspected (as required by the PDB format).  
    • Alternate location identifiers, insertion codes, and multiple models
      are ignored—the first occurrence of each (chain, residue number)
      wins.  That behaviour fits most single‑model PDBs such as CATH
      domain files (e.g. “1opcA00.pdb”).
    """
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        print(fname, "has multiple models")
        # return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure(model=1)
    # keep caller’s line intact – grab first model as an AtomArray
    backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
    return aa_seq_from_backbone(backbone_atoms)


def _norm(v): 
    n = np.linalg.norm(v); 
    return v / (n + 1e-12)
    
def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize the last dim with a small eps for numerical stability."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def frame_from_triad(N, CA, C):
    """
    Return (R, t) at residue i using triad (N, CA, C).
    R columns are x,y,z; origin at CA.
    """
    x = _norm(C - CA)
    u = _norm(N - CA)
    z = _norm(np.cross(x, u))   # right-handed; preserves chirality
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    t = CA.copy()
    return R, t


def frame_from_triad_torch(N: torch.Tensor,
                     CA: torch.Tensor,
                     C: torch.Tensor,
                     eps: float = 1e-8):
    """
    Compute (R, t) from a triad (N, CA, C).

    Args
    ----
    N, CA, C : (*, 3) tensors (same batch shape *), dtype/device arbitrary.
    eps      : small value to avoid div-by-zero in normalization.

    Returns
    -------
    R : (*, 3, 3) rotation matrices; columns are x, y, z.
    t : (*, 3)    translation (just CA).
    """
    # unit x-axis (points from CA to C)
    x = _normalize(C - CA, eps)

    # helper vector
    u = _normalize(N - CA, eps)

    # right-handed z-axis, then y-axis
    z = _normalize(torch.cross(x, u, dim=-1), eps)
    y = torch.cross(z, x, dim=-1)

    # stack column-wise to form rotation matrix
    R = torch.stack((x, y, z), dim=-1)   # (*, 3, 3)
    t = CA.clone()                       # (*, 3)

    return R, t


def rot_geodesic(RA, RB):
    """Angle of RA^T RB in radians."""
    R = RA.T @ RB
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(c))



def extract_side_chain_coords(fname: str) -> Optional[List[Tuple[str, List[Tuple[str, Optional[np.ndarray]]]]]]:
    """
    Extract the coordinates of all side chain atoms, grouped by residue,
    and express them in a canonical local frame defined by the backbone atoms.
    
    For each residue:
      - The canonical frame is defined as:
          origin = CA coordinate
          x-axis = normalized vector from CA to C
          y-axis = normalized cross product of (N - CA) and x-axis
          z-axis = cross product of x-axis and y-axis
      - Side-chain atoms are transformed into this frame.
      - The side-chain atoms are then re-ordered according to a pre-defined 
        canonical ordering (for that amino acid type). If an expected atom 
        is missing, None is inserted.
    
    Returns:
      A list of tuples, one per residue, of the form:
         (res_name, [(atom_name, transformed_coord), ...])
      where transformed_coord is a 3-element numpy array in the canonical frame,
      or None if the atom is missing.
    """
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname, "rt") as f:
        structure = PDBFile.read(f)
    
    # Only handle single-model structures
    # if structure.get_model_count() > 1:
    #     return None
    
    # Assuming the domain file represents a single chain structure
    chain = structure.get_structure(1)
    
    sidechain_by_residue = []
    
    # Iterate through residues using your residue iterator
    for residue in struc.residue_iter(chain):
        # Assumption: all atoms in residue share the same res_name and res_id.
        res_name = residue.res_name[0].strip()  # e.g., "ARG", "ALA", etc.
        if res_name not in STANDARD_RESIDUES:
            continue  # or handle non-standard residues as needed
        
        # Build a dictionary for backbone atoms (N, CA, C)
        backbone = {atom.atom_name: atom for atom in residue if atom.atom_name in {"N", "CA", "C"}}
        if not {"N", "CA", "C"}.issubset(backbone):
            continue  # Skip if any backbone atom is missing
        
        N_coord = backbone["N"].coord
        CA_coord = backbone["CA"].coord
        C_coord = backbone["C"].coord
        
        # Define the canonical frame:
        # x-axis: from CA to C (normalized)
        x_axis = C_coord - CA_coord
        norm_x = np.linalg.norm(x_axis)
        if norm_x == 0:
            continue
        x_axis = x_axis / norm_x
        
        # y-axis: normalized cross product of (N - CA) and x_axis
        y_axis = np.cross((N_coord - CA_coord), x_axis)
        norm_y = np.linalg.norm(y_axis)
        if norm_y == 0:
            continue
        y_axis = y_axis / norm_y
        
        # z-axis: cross product of x_axis and y_axis
        z_axis = np.cross(x_axis, y_axis)
        
        # Build the rotation matrix; columns are the new basis vectors.
        R = np.column_stack((x_axis, y_axis, z_axis))
        
        # For side-chain atoms, re-order them according to canonical ordering.
        canonical_order = STANDARD_SIDECHAIN_ORDER[res_name]
        ordered_sidechain_atoms = []
        for atom_name in canonical_order:
            found_atom = None
            for atom in residue:
                if atom.atom_name == atom_name:
                    found_atom = atom
                    break
            if found_atom is not None:
                relative = found_atom.coord - CA_coord
                # Transform to canonical frame: dot product with each basis vector
                new_coord = np.dot(relative, R)
                ordered_sidechain_atoms.append((atom_name, new_coord))
            else:
                ordered_sidechain_atoms.append((atom_name, None))
        
        sidechain_by_residue.append((res_name, ordered_sidechain_atoms))
    
    return sidechain_by_residue


SideChainAtomRelative = namedtuple(
    "SideChainAtom", ["name", "element", "bond_dist", "bond_angle", "dihedral_angle"]
)

def circular_rmse(angles, ref):
    """
    Calculate the circular RMSE of a list of angles relative to a given reference angle.
    
    Parameters:
        angles (array-like): List of angles in radians (e.g., in the range [-pi, pi]).
        ref (float): The reference angle in radians.
        
    Returns:
        float: The circular RMSE.
    """
    angles = np.array(angles)
    # Compute the difference between each angle and the reference angle,
    # adjusting for circular wrap-around.
    diff = (angles - ref + np.pi) % (2 * np.pi) - np.pi
    # Compute RMSE from these circular differences.
    rmse = np.sqrt(np.mean(diff**2))
    return rmse   


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Gets the angle between u and v"""
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    unit_vector = lambda vector: vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def collect_aa_sidechain_angles(
    ref_fname: str,
) -> Dict[str, List[SideChainAtomRelative]]:
    """
    Collect the sidechain distances/angles/dihedrals for all amino acids such that
    we can reconstruct an approximate version of them from the backbone coordinates
    and these relative distances/angles/dihedrals

    Returns a dictionary that maps each amino acid residue to a list of SideChainAtom
    objects
    """
    opener = gzip.open if ref_fname.endswith(".gz") else open
    with opener(ref_fname, "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        raise ValueError
    chain = structure.get_structure()[0]
    retval = defaultdict(list)
    for _, res_atoms in groupby(chain, key=lambda a: a.res_id):
        res_atoms = struc.array(list(res_atoms))
        # Residue name, 3 letter -> 1 letter
        try:
            residue = ProteinSequence.convert_letter_3to1(res_atoms[0].res_name)
        except KeyError:
            logging.warning(
                f"{ref_fname}: Skipping unknown residue {res_atoms[0].res_name}"
            )
            continue
        if residue in retval:
            continue
        backbone_mask = struc.filter_backbone(res_atoms)
        a, b, c = res_atoms[backbone_mask].coord  # Backbone
        for sidechain_atom in res_atoms[~backbone_mask]:
            d = sidechain_atom.coord
            retval[residue].append(
                SideChainAtomRelative(
                    name=sidechain_atom.atom_name,
                    element=sidechain_atom.element,
                    bond_dist=np.linalg.norm(d - c, 2),
                    bond_angle=angle_between(d - c, b - c),
                    dihedral_angle=struc.dihedral(a, b, c, d),
                )
            )
    logging.info(
        "Collected {} amino acid sidechain angles from {}".format(
            len(retval), os.path.abspath(ref_fname)
        )
    )
    return retval


@functools.lru_cache(maxsize=32)
def build_aa_sidechain_dict(
    reference_pdbs: Optional[Collection[str]] = None,
) -> Dict[str, List[SideChainAtomRelative]]:
    """
    Build a dictionary that maps each amino acid residue to a list of SideChainAtom
    that specify how to build out that sidechain's atoms from the backbone
    """
    if not reference_pdbs:
        reference_pdbs = glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/*.pdb")
        )

    ref_file_counter = 0
    retval = {}
    for pdb in reference_pdbs:
        try:
            sidechain_angles = collect_aa_sidechain_angles(pdb)
            retval.update(sidechain_angles)  # Overwrites any existing key/value pairs
            ref_file_counter += 1
        except ValueError:
            continue
    logging.info(f"Built sidechain dictionary with {len(retval)} amino acids from {ref_file_counter} files")
    return retval


def add_sidechains_to_backbone(
    backbone_pdb_fname: str,
    aa_seq: str,
    out_fname: str,
    reference_pdbs: Optional[Collection[str]] = None,
) -> str:
    """
    Add the sidechains specified by the amino acid sequence to the backbone
    """
    opener = gzip.open if backbone_pdb_fname.endswith(".gz") else open
    with opener(backbone_pdb_fname, "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        raise ValueError
    chain = structure.get_structure()[0]

    aa_library = build_aa_sidechain_dict(reference_pdbs)

    atom_idx = 1  # 1-indexed
    full_atoms = []
    for res_aa, (_, backbone_atoms) in zip(
        aa_seq, groupby(chain, key=lambda a: a.res_id)
    ):
        backbone_atoms = struc.array(list(backbone_atoms))
        assert len(backbone_atoms) == 3
        for b in backbone_atoms:
            b.atom_id = atom_idx
            atom_idx += 1
            b.res_name = ProteinSequence.convert_letter_1to3(res_aa)
            full_atoms.append(b)
        # Place each atom in the sidechain
        a, b, c = backbone_atoms.coord
        for rel_atom in aa_library[res_aa]:
            d = nerf.place_dihedral(
                a,
                b,
                c,
                rel_atom.bond_angle,
                rel_atom.bond_dist,
                rel_atom.dihedral_angle,
            )
            atom = struc.Atom(
                d,
                chain_id=backbone_atoms[0].chain_id,
                res_id=backbone_atoms[0].res_id,
                atom_id=atom_idx,
                res_name=ProteinSequence.convert_letter_1to3(res_aa),
                atom_name=rel_atom.name,
                element=rel_atom.element,
                hetero=backbone_atoms[0].hetero,
            )
            atom_idx += 1
            full_atoms.append(atom)
    sink = PDBFile()
    sink.set_structure(struc.array(full_atoms))
    sink.write(out_fname)
    return out_fname
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_reverse_dihedral()
    # backbone = collect_aa_sidechain_angles(
    #     os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/1CRN.pdb")
    # )
    # print(build_aa_sidechain_dict())
