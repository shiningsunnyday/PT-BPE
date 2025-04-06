from foldingdiff.potential_model import *
from foldingdiff.tokenizer import Tokenizer
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff BPE Script")
    parser.add_argument("--cath_folder", default="/n/home02/msun415/foldingdiff/data/cath/dompdb/")
    parser.add_argument("--toy", type=int, default=10, 
                            help="Number of PDB files.")
    parser.add_argument("--debug", action='store_true')    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cuda", default="cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = FullCathCanonicalCoordsDataset(args.cath_folder, use_cache=False, debug=args.debug, zero_center=False, toy=args.toy, secondary=False)
    model = LongSequenceGRU()    
    model.to(args.cuda)
    opt = optim.Adam(model.parameters())    
    dataset = [Tokenizer(x) for x in dataset.structures]

    def obtain_span(a, b):
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

    for epoch in range(args.epochs):
        for t in dataset:  
            opt.zero_grad()
            loss = 0.          
            N = t.n
            log_a = [0.] + [float("-inf") for _ in range(N)] # a(0)=1

            out = [[None for _ in range(N+1)] for _ in range(N)]
            # precompute
            for i in range(N):
                span = obtain_span(i, N)
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
                        

            for k in tqdm(range(1, N+1)):

                scores = []                
                for l in range(1, k+1):                     
                    # span = obtain_span(k-l, l) # bonds k-l to l
                    # span_tensor = torch.tensor(span, dtype=torch.float32).reshape(1, -1, 1).to(args.cuda)
                    # assert (out[k-l][l] - model.fc(model(span_tensor, [len(span)])[1]).item()).abs() < 1e-5
                    scores.append(log_a[k-l] + out[k-l][l])

                log_a[k] = torch.logsumexp(torch.stack(scores), dim=0)
            loss += -log_a[N]
            a_loss = loss.item()
            penalty_loss = 0.01*l1_penalty(model)
            loss += penalty_loss
            loss.backward()
            print(a_loss, penalty_loss)
            opt.step()
    

if __name__ == "__main__":
    breakpoint()
    main()
