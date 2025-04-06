from foldingdiff.potential_model import *
from foldingdiff.tokenizer import Tokenizer
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff BPE Script")
    parser.add_argument("--cath_folder", default="/n/home02/msun415/foldingdiff/data/cath/dompdb/")
    parser.add_argument("--toy", type=int, default=10, 
                            help="Number of PDB files.")
    parser.add_argument("--debug", action='store_true')    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cuda", default="cpu")
    parser.add_argument("--save_dir", default="/n/home02/msun415/foldingdiff/ckpts")
    parser.add_argument("--plot_dir", default="/n/home02/msun415/foldingdiff/plots/learn")
    return parser.parse_args()


def semi_crf_dp_and_map(out, N):
    """
    out[i][l]: the (log-)score for a segment starting at i, of length l
               i.e. from x_{i} to x_{i + l - 1}, inclusive.
    N: length of the sequence.

    Returns:
      log_a: list of length (N+1) - log_a[k] is log-partition up to k
      map_a: list of length (N+1) - map_a[k] is the best (max) segmentation score up to k
      best_lens: list of length (N+1) storing the best segment length chosen at k
    """

    # log_a[k] for log-partition
    log_a = [float("-inf")] * (N + 1)
    log_a[0] = 0.0  # alpha(0) = 1 => log(1) = 0

    # map_a[k] for Viterbi (best) segmentation score
    map_a = [float("-inf")] * (N + 1)
    map_a[0] = 0.0  # best score for empty prefix = 0

    # backpointer: best_lens[k] = the segment length that yields the best segmentation at k
    best_lens = [-1] * (N + 1)

    # DP from k=1 to k=N
    for k in range(1, N + 1):
        # For log_a, gather candidate log-scores to do log-sum-exp
        alpha_candidates = []
        # For map_a, track the maximum
        best_score = float("-inf")
        best_l = None

        for l in range(1, k + 1):
            # Score from k-l to k-1 is out[k-l][l]
            cand_part = log_a[k - l] + out[k - l][l]   # for log-partition
            alpha_candidates.append(cand_part)

            cand_viterbi = map_a[k - l] + out[k - l][l]  # for MAP
            if cand_viterbi > best_score:
                best_score = cand_viterbi
                best_l = l

        # log_a[k] = log-sum-exp of all alpha_candidates
        log_a[k] = torch.logsumexp(torch.stack(alpha_candidates), dim=0)

        # map_a[k] = maximum of all viterbi candidates
        map_a[k] = best_score
        best_lens[k] = best_l

    return log_a, map_a, best_lens

def backtrace_map_segmentation(best_lens, N):
    """
    Given backpointer (best_lens), recover the MAP segmentation from k=N down to k=0.

    best_lens[k] tells which segment length was chosen at position k.
    """

    segmentation = []
    k = N
    while k > 0:
        l = best_lens[k]
        start_idx = k - l  # segment from [start_idx .. (k-1)]
        segmentation.append((start_idx, k))  # or store (start_idx, k-1)
        k = start_idx

    # The segments are recovered in reverse order, so reverse them
    segmentation.reverse()
    return segmentation


def obtain_span(t, a, b):
    geo = t.token_geo(a, b)
    geo = {k: (x for x in geo[k]) for k in geo}
    # bd
    # bd, ba, bd
    # bd, ba, bd, (ba, da, bd)_n
    i = a%3
    j = a%3
    k = a%3
    span = []       
    bdt = t.BOND_TYPES[i]        
    span.append(next(geo[bdt]))
    i += 1        
    if b > 1:
        bat = t.BOND_ANGLES[j%3]
        span.append(next(geo[bat]))
        j += 1
        bdt = t.BOND_TYPES[i%3]
        span.append(next(geo[bdt]))
        i += 1
    for idx in range(2, b):
        bat = t.BOND_ANGLES[j%3]
        span.append(next(geo[bat]))
        j += 1
        bdt = t.DIHEDRAL_ANGLES[k%3]
        span.append(next(geo[bdt]))
        k += 1
        bdt = t.BOND_TYPES[i%3]
        span.append(next(geo[bdt]))
        i += 1
    return span


def main():
    args = parse_args()
    dataset = FullCathCanonicalCoordsDataset(args.cath_folder, use_cache=False, debug=args.debug, zero_center=False, toy=args.toy, secondary=False)
    model = LongSequenceGRU()    
    model.to(args.cuda)
    opt = optim.Adam(model.parameters())    
    dataset = [Tokenizer(x) for x in dataset.structures]
    best_scores = [float("-inf") for _ in range(len(dataset))]
    best_loss = float("inf")
    for epoch in range(args.epochs):
        total_loss = 0.
        for _idx, t in enumerate(dataset):
            opt.zero_grad()
            loss = 0.          
            N = 3*t.n-1
            out = [[None for _ in range(N+1)] for _ in range(N)]
            # compute
            for i in range(N):
                span = obtain_span(t, i, N-i)
                span_tensor = torch.tensor(span, dtype=torch.float32).reshape(1, -1, 1).to(args.cuda)
                out_all_steps, _ = model(span_tensor, [len(span)])
                out_all_steps = out_all_steps.flatten()
                for j in range(i+1, N+1):
                    if j-i == 1: # len(span) == 1                        
                        out[i][j-i] = out_all_steps[0] # bonds i to j
                    elif j-i == 2: # len(span) == 2
                        out[i][j-i] = out_all_steps[2]
                    else: # len(span) == 3
                        out[i][j-i] = out_all_steps[2+3*(j-i-2)]                                    
            log_a, map_a, best_lens = semi_crf_dp_and_map(out, N)
            best_seg = backtrace_map_segmentation(best_lens, N)            
            t.bond_to_token = {start: (start, idx, end-start) for idx, (start, end) in enumerate(best_seg)}            
            loss += -log_a[N]
            a_loss = loss.item()
            penalty_loss = 0.1*l1_penalty(model)
            loss += penalty_loss
            loss.backward()
            print(f"-log a: {a_loss}, l1 penalty: {penalty_loss.item()}")
            opt.step()
            if (map_a[N]-log_a[N]) > best_scores[_idx]:
                prob = torch.exp(map_a[N]-log_a[N])
                t.visualize(os.path.join(args.plot_dir, f"epoch={epoch}_{_idx}_{prob.item()}.png"), vis_dihedral=False)
                best_scores[_idx] = map_a[N]-log_a[N]
                print("Max prob:", prob)
            total_loss += loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model, os.path.join(args.save_dir, f"{epoch}_loss={best_loss}.pt"))
    

if __name__ == "__main__":
    breakpoint()
    main()
