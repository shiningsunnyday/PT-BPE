#!/usr/bin/env python3

import torch
import esm
from typing import List
import argparse
from esm.esmfold.v1.misc import batch_encode_sequences
model = esm.pretrained.esmfold_v1()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.eval().to(device)

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

# sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
sequence = "TPVTEDRFGILTEKYRIPVPILPGLPMNNHGNYVQTYGIGLKELWVIDEKKWKPGRVDHTVGWPLDRHTYGGSFLYHLNEGEPLLALGFVVGLDYQNPYLSPFREFQRWKHHPSIKPTLEGGKRIAYGARALNEGGFQRNIRPSCHGILGVYGGMIYTGIFYWIFRGMEPWTLKHKGSDSDQLKPAKDCTPIEYPKPDGQISF"
# Multimer prediction can be done with chains separated by ':'

def get_embeddings_esmfold_batched(
    aa_seqs: List[str],
    batch_size: int = 8,
) -> List[torch.Tensor]:
	"""
	Compute per-residue embeddings for a list of sequences using ESMFold v1 in GPU batches.

	Returns a list of length len(aa_seqs), where each element is a FloatTensor of shape (L_i, D).
	"""
	n_layer = model.esm.num_layers
	embeddings: List[torch.Tensor] = []

	# process in batches
	for i in range(0, len(aa_seqs), batch_size):
		batch_seqs = aa_seqs[i:i+batch_size]
		# tokenize: returns (aatype, mask, residx, linker_mask, chain_index)
		aatype, mask, residx, linker_mask, chainf_index = batch_encode_sequences(batch_seqs)
		aatype = aatype.to(device)	
		with torch.no_grad():
		    out = model.esm(aatype, repr_layers=[n_layer], return_contacts=False)	
		# out["representations"][n_layer] is (B, L, D)
		layer_feats = out["representations"][n_layer]  # (batch, L, D)
		assert layer_feats.shape[1] == max([len(aa) for aa in batch_seqs])
		# strip BOS/EOS for each sequence in batch
		for j, feats in enumerate(layer_feats):
			l = len(batch_seqs[j])
			embeddings.append(feats.cpu()[:l])

	return embeddings


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--in-file")
	parser.add_argument("--out-file")
	args = parser.parse_args()
	seqs = [l.rstrip('\n') for l in open(args.in_file).readlines()]
	# seqs = [sequence, sequence[:100]]
	out = get_embeddings_esmfold_batched(seqs)
	torch.save(out, args.out_file)    
