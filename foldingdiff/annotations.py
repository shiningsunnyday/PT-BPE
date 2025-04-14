from Bio.PDB import PDBParser, DSSP
from collections import defaultdict
from pathlib import Path

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
    try:
        dssp = DSSP(model, fname)
    except:
        print(fname, "failed")
        raise        
    ss_segments = defaultdict(list)  # key: chain id, value: list of (ss_type, start, end)
    standard_residues = set()
    for chain in model:
        for residue in chain.get_residues():
            hetfield, resseq, icode = residue.get_id()
            if hetfield.strip() == "":  # standard amino acid
                standard_residues.add((chain.id, resseq, icode))    
    for key, dssp_data in dssp.property_dict.items():
        chain_id, res_id_tuple = key  # res_id_tuple is typically (hetfield, resseq, icode)
        hetfield, resseq, icode = res_id_tuple
        if (chain_id, resseq, icode) not in standard_residues: # cause hetfield is useless
            continue        
        ss = dssp_data[2]  # secondary structure assignment (e.g., H, E, etc.)        
        # For simplicity, treat blank assignments (often loops) as 'C' (coil)
        if ss == " ":
            ss = "C"        
        # Append tuple (residue number, ss) for later grouping
        ss_segments[chain_id].append((resseq, ss))  
    all_segments = []
    # Group segments for each chain and print results
    print("Secondary Structure Segments (from DSSP):")
    for chain_id, res_ss_list in ss_segments.items():
        segments = group_segments(res_ss_list)
        print(f"Chain {chain_id}:")
        for ss_type, start, end in segments:
            print(f"  {ss_type}: residues {start} to {end}")        
        all_segments.append(segments)
    if len(all_segments) > 1:
        breakpoint()
    else:
        return all_segments[0]
    return all_segments