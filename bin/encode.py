import pyrosetta
from pyrosetta import pose_from_pdb
import nglview as nv
from ipywidgets import HBox
from tqdm import tqdm
import mdtraj as md
from Bio.PDB import PDBParser
import os
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
from foldingdiff.tokenizer import *
import scipy.io
import numpy as np
import subprocess

def visualize():
    # Initialize PyRosetta
    pyrosetta.init()

    # Load the PDB file
    pdb_filename = "/n/home02/msun415/foldingdiff/data/cath/dompdb/152lA00.pdb"  # Change this to your actual file
    pose = pose_from_pdb(pdb_filename)

    # Residue index to modify (change as needed)
    residue_index = 10  # Change to the residue you want to modify

    # Get initial torsion angles
    initial_phi = pose.phi(residue_index)
    initial_psi = pose.psi(residue_index)

    print(f"Before modification - Phi: {initial_phi:.2f}, Psi: {initial_psi:.2f}")

    # Save the original structure
    before_pdb = "before.pdb"
    pose.dump_pdb(before_pdb)

    # Modify the torsion angle
    pose.set_phi(residue_index, initial_phi + 50)  # Increase phi by 20 degrees
    pose.set_psi(residue_index, initial_psi)  # Decrease psi by 15 degrees

    # Get modified torsion angles
    modified_phi = pose.phi(residue_index)
    modified_psi = pose.psi(residue_index)

    print(f"After modification - Phi: {modified_phi:.2f}, Psi: {modified_psi:.2f}")

    # Save the modified structure
    after_pdb = "after.pdb"
    pose.dump_pdb(after_pdb)

    # Create two separate NGLView widgets
    view_before = nv.show_structure_file(before_pdb)
    view_after = nv.show_structure_file(after_pdb)

    # Set titles
    view_before._set_size('400px', '400px')
    view_after._set_size('400px', '400px')

    # Display side by side
    HBox([view_before, view_after])    

def parse_pdb(pdb_file):
    # Create a PDB parser object
    parser = PDBParser(QUIET=True)

    # Path to your PDB file (e.g., '12asA00.pdb')
    structure = parser.get_structure("protein", pdb_file)

    # We'll store coordinates for each residue as a tuple: (N, CA, C)
    backbone_coords = []

    # Iterate over all residues in all chains
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check that the residue has the backbone atoms we need.
                if all(atom_name in residue for atom_name in ['N', 'CA', 'C']):
                    # Extract coordinates
                    N_coord = residue['N'].get_coord()
                    CA_coord = residue['CA'].get_coord()
                    C_coord = residue['C'].get_coord()
                    backbone_coords.append((N_coord, CA_coord, C_coord))

    # Now, backbone_coords is a list of tuples, each containing three numpy arrays of shape (3,).
    # For a protein with N residues, you have N entries, corresponding to 3 x 3D coordinates.
    for i, (N_coord, CA_coord, C_coord) in enumerate(backbone_coords, start=1):
        print(f"Residue {i}:")
        print(f"  N:  {N_coord}")
        print(f"  CA: {CA_coord}")
        print(f"  C:  {C_coord}")

    return backbone_coords

def call_freqgeo(G):
    scipy.io.savemat('/n/home02/msun415/foldingdiff/data/cath/graphs.mat', {"G": G})

    breakpoint()
    try:
        result = subprocess.run(
            ["matlab", "-batch", "testmexfreqgeo"],
            cwd="/n/home02/msun415/foldingdiff/freqgeo-1.0/src",
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print("MATLAB finished successfully.")
        print("Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("MATLAB failed with exit code", e.returncode)
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        
    res = scipy.io.loadmat("/n/home02/msun415/foldingdiff/freqgeo-1.0/src/mexfreqgeo_output.mat")
    count = res['count']
    graphs = res['graphs']
    occurence = res['occurence']

    for i in range(len(graphs[0])):
        g = graphs[0, i][0, 0]
        nodelabels, nodepos, edges = g
        n = len(nodelabels)
        breakpoint()    

cath_folder = "/n/home02/msun415/foldingdiff/data/cath/dompdb/"  # Change this to your actual file
all_coords = []
files = os.listdir(cath_folder)
files = sorted(files, key=len)
for f in tqdm(files[:10]):
    if f:
        print(f)
        all_coords.append(parse_pdb(os.path.join(cath_folder, f)))

dataset = FullCathCanonicalCoordsDataset('/n/home02/msun415/foldingdiff/data/cath/dompdb', use_cache=False, debug=True, zero_center=False)

vocab = {}
for i in range(9):
    breakpoint()
    tokens = Tokenizer(dataset.structures[i])
    # call_freqgeo(G)
