#!/usr/bin/env python3

import torch
import esm
from typing import List
import argparse

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
model.set_chunk_size(128)

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
# Multimer prediction can be done with chains separated by ':'

def get_plddt_esmfold_batched(
    aa_seqs: List[str],
    batch_size: int = 2,
) -> List[torch.Tensor]:
    """
    Run ESMFold v1 in GPU batches and return, for each sequence, 
    a FloatTensor of shape (L,) with per-residue mean pLDDT in [0,1].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    mean_plddt_list: List[torch.Tensor] = []

    for i in range(0, len(aa_seqs), batch_size):
        batch_seqs = aa_seqs[i : i + batch_size]

        # 1) inference on the batch
        with torch.no_grad():
            out = model.infer(batch_seqs)

        # 2) load into torch and normalize to [0,1]
        #    out["plddt"]           : numpy array (batch, L_max) in [0,100]
        #    out["atom37_atom_exists"]: numpy array (batch, L_max, N_atoms)
        plddt = (
            out["plddt"]
            .to(device)
            .float()
            / 100.0
        )  # (B, L_max)        
        atom37_exists = (
            out["atom37_atom_exists"]
            .to(device)
            .float()
        )  # (B, L_max, N_atoms)
        # 3) compute weighted average per residue
        weighted = plddt * atom37_exists         # (B, L_max, N_atoms)
        sum_weighted = weighted.sum(dim=-1)                    # (B, L_max)
        n_atoms = atom37_exists.sum(dim=-1)                    # (B, L_max)
        mean_batch = sum_weighted / n_atoms                    # (B, L_max)
        assert mean_batch.shape[1] == max([len(aa) for aa in batch_seqs])
        # 4) chop back to each sequence length and move to CPU
        for j, seq in enumerate(batch_seqs):
            L = len(seq)
            mean_plddt_list.append(mean_batch[j, :L].cpu())

    return mean_plddt_list

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--in-file")
  parser.add_argument("--out-file")
  args = parser.parse_args()  
  seqs = [l.rstrip('\n') for l in open(args.in_file).readlines()]
  # seqs = [sequence, sequence[:100]]
  out = get_plddt_esmfold_batched(seqs)
  torch.save(out, args.out_file)
  # get_plddt_with_esmfold([sequence])
