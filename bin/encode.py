import pyrosetta
from pyrosetta import pose_from_pdb
import nglview as nv
from ipywidgets import HBox
from tqdm import tqdm
import tempfile
import nglview as nv
import imageio
from Bio.PDB import PDBParser, DSSP
import os
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
from foldingdiff.bpe import *
from foldingdiff.plotting import *
import scipy.io
import numpy as np
import subprocess
import argparse
from datetime import datetime

def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_{timestamp}.log")
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().info("Logger initialized.")

def get_logger():
    """Helper to retrieve the global logger."""
    return LOGGER

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

def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff BPE Script")
    parser.add_argument("--save-dir", type=str, default="plots/bpe", 
                        help="Directory to save output files (images, pdb files, plots, etc.).")
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory where log files will be saved.")
    parser.add_argument("--toy", type=int, default=10, 
                            help="Number of PDB files.")                        
    return parser.parse_args()

def amino_acid_sequence(fname):
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',  'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',  'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}       
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0] 
    seq = [d3to1[k] for k in struc.get_residues(source_struct)[1]]
    return seq

# Function to group continuous segments with the same secondary structure type
def group_segments(res_ss_list):
    segments = []
    if not res_ss_list:
        return segments
    # Sort by residue number
    res_ss_list.sort(key=lambda x: x[0])
    current_ss = res_ss_list[0][1]
    start = res_ss_list[0][0]
    end = start
    for res, ss in res_ss_list[1:]:
        if ss == current_ss and res == end + 1:
            # Continue the segment
            end = res
        else:
            segments.append((current_ss, start, end))
            current_ss = ss
            start = res
            end = res
    segments.append((current_ss, start, end))
    return segments

def find_secondary_structures(fname):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(fname).stem, fname)
    model = structure[0]  # assuming you want the first model
    dssp = DSSP(model, fname)
    ss_segments = defaultdict(list)  # key: chain id, value: list of (ss_type, start, end)
    for key, dssp_data in dssp.property_dict.items():
        chain_id = key[0]
        res_id = key[1][1]  # residue number
        ss = dssp_data[2]  # secondary structure assignment (e.g., H, E, etc.)
        
        # For simplicity, treat blank assignments (often loops) as 'C' (coil)
        if ss == " ":
            ss = "C"
        
        # Append tuple (residue number, ss) for later grouping
        ss_segments[chain_id].append((res_id, ss))    
    all_segments = []
    # Group segments for each chain and print results
    print("Secondary Structure Segments (from DSSP):")
    for chain_id, res_ss_list in ss_segments.items():
        segments = group_segments(res_ss_list)
        print(f"Chain {chain_id}:")
        for ss_type, start, end in segments:
            print(f"  {ss_type}: residues {start} to {end}")        
        all_segments.append(segments)
    return all_segments


def main():
    args = parse_args()
    setup_logger(args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info("Script started.")
    
    # Use args.save_dir for saving outputs.
    # Input folder remains the same for now.
    cath_folder = "/n/home02/msun415/foldingdiff/data/cath/dompdb/"
    # all_coords = []
    
    # files = os.listdir(cath_folder)
    # files = sorted(files, key=len)
    # for f in tqdm(files[:10]):
    #     if f:
    #         logger.info("Processing file: %s", f)
    #         all_coords.append(parse_pdb(os.path.join(cath_folder, f)))

    dataset = FullCathCanonicalCoordsDataset(cath_folder, 
                                               use_cache=False, debug=True, zero_center=False, toy=args.toy)    
    find_secondary_structures(dataset.fnames[0])
    for i, struc in enumerate(dataset.structures):
        if (struc['angles']['psi']==struc['angles']['psi']).sum() < len(struc['angles']['psi'])-1:
            breakpoint()
    bpe = BPE(dataset.structures, bins=100, save_dir=args.save_dir)
    # bpe_debug = BPE(dataset.structures, bins=100, save_dir=args.save_dir)
    bpe.initialize()
    # bpe_debug.initialize()
    bpe.bin()
    # bpe_debug.old_bin()
    for t in range(1000):
        for i, tokenizer in enumerate(bpe.tokenizers):
            if i > 0:
                continue
            # Save visualization using the output directory provided
            visual_path = os.path.join(args.save_dir, f"backbone_{i}_iter={t}.png")
            tokenizer.visualize(visual_path)        
        bpe.step()
        # bpe_debug.old_step()
        # for k in bpe._geo_dict:
        #     if k not in bpe_debug._geo_dict:
        #         breakpoint()
        #     elif set(bpe._geo_dict[k]) != set(bpe_debug._geo_dict[k]):
        #         breakpoint()
        # for k in bpe_debug._geo_dict:
        #     if k not in bpe._geo_dict:
        #         breakpoint()
        #     elif set(bpe_debug._geo_dict[k]) != set(bpe._geo_dict[k]):
        #         breakpoint()
    # save
    pickle.dump(bpe, open('bpe.pkl', 'wb+'))
    logger.info("Script finished.")



if __name__ == "__main__":
    main()
