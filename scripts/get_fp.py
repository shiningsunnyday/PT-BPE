#!/usr/bin/env python3




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--in-file")
	parser.add_argument("--out-file")
	args = parser.parse_args()
	seqs = [l.rstrip('\n') for l in open(args.in_file).readlines()]
	# seqs = [sequence, sequence[:100]]
	out = get_embeddings_esmfold_batched(seqs)
	torch.save(out, args.out_file)    
