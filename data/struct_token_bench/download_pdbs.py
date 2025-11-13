import os
import io
from pathlib import Path
from biotite.database import rcsb
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import biotite.structure.io.pdb as pdbio
import biotite.structure as struc

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    type=str,
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="pdb_files",
    help="Directory to save the downloaded PDB files.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="Number of parallel workers for downloading.",
)
args = parser.parse_args()


with open(args.data_file, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

pdb_ids = []
for item in data:
    if "ConformationalSwitchDataset" in args.data_file:
        pdb_ids.append((item["prot1_pdb_id"], item["prot1_chain_id"]))
        pdb_ids.append((item["prot2_pdb_id"], item["prot2_chain_id"]))
    elif "pdb_id" in item and "chain_id" in item:
        pdb_ids.append((item["pdb_id"], item["chain_id"]))
    else:
        assert "pdb_path" in item
        p = Path(item["pdb_path"])
        pdb_id, chain_id = p.stem.split("_")
        assert Path(item["pdb_path"]).parent == args.output_dir

os.makedirs(args.output_dir, exist_ok=True)

def download_and_save(pdb_id, chain_id):
    if os.path.exists(os.path.join(args.output_dir, f"{pdb_id}_{chain_id}.pdb")):
        return f"{pdb_id}_{chain_id}", "already exists"
    try:
        f: io.StringIO = rcsb.fetch(pdb_id, "pdb")  # Format: "pdb", "cif", "xml", etc.
        structure = pdbio.PDBFile.read(f).get_structure(model=1)
        structure_chain = structure[structure.chain_id == chain_id]
        if len(structure_chain) == 0:
            return f"{pdb_id}_{chain_id}", "no chain found"
        file_path = os.path.join(args.output_dir, f"{pdb_id}_{chain_id}.pdb")
        with open(file_path, "w") as out_file:
            pdb_out = pdbio.PDBFile()
            pdb_out.set_structure(structure_chain)
            pdb_out.write(out_file)
        return f"{pdb_id}_{chain_id}", "success"
    except Exception as e:
        return f"{pdb_id}_{chain_id}", f"failed: {e}"

# Parallel download
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(download_and_save, pdb_id, chain_id): (pdb_id, chain_id)
        for pdb_id, chain_id in pdb_ids
    }
    for future in as_completed(futures):
        pdb_chain_id, status = future.result()
        print(f"{pdb_chain_id}: {status}")
