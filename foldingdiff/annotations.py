from Bio.PDB import PDBParser, DSSP
from collections import defaultdict

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