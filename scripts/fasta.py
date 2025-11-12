from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    """Build a basic CLI parser."""
    parser = argparse.ArgumentParser(
        usage="Wrapper for running FASTA+annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdb_file", type=str, help="Input dir containing .pdb files"
    )
    parser.add_argument(
        "--out_file", type=str, help="Output FASTA file"
    )
    return parser


def main():
    """Run script."""
    args = build_parser().parse_args()
    pdb_file = args.pdb_file

    parser = PDBParser(QUIET=True)
    pdb = Path(pdb_file).stem
    structure = parser.get_structure(f"{pdb}", pdb_file)
    model = structure[0]

    # Collect residues from the specified chain
    residues = []
    for chain in model:
        for res in chain:
            # Skip non-standard residues
            if res.get_id()[0] == " ":
                # Convert to single-letter code
                if 'CA' in res:
                    residues.append(res['CA'].parent.resname)

    # Convert 3-letter codes to single-letter
    three_to_one = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
        'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
        'TYR':'Y','VAL':'V','MSE':'M'  # example: MSE is selenomethionine
    }
    seq_str = ''.join(three_to_one.get(x, 'X') for x in residues)
    seq_record = SeqRecord(Seq(seq_str), id=f"{pdb}_domain", description=f"CATH domain {pdb}")
    SeqIO.write(seq_record, args.out_file, "fasta")    

if __name__ == "__main__":
    main()


