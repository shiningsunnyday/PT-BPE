from foldingdiff.potential_model import *
from foldingdiff.modelling import *
from foldingdiff.tokenizer import Tokenizer
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
import torch
import gc
import psutil
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
from torch import optim
import numpy as np
from tqdm import tqdm
import time
import argparse
import os
import logging
import inspect

# Configure logging (you can customize format and level here)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def parse_args():    
    parser = argparse.ArgumentParser(description="FoldingDiff BPE Script")
    parser.add_argument("--cath_folder", default="./data/cath/dompdb/")
    parser.add_argument("--toy", type=int, default=10, 
                            help="Number of PDB files.")
    parser.add_argument("--pad", type=int, default=512, help="Max protein size")
    parser.add_argument("--debug", action='store_true')    
    parser.add_argument("--visualize", action='store_true')    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cuda", default="cpu")
    # run artifacts
    parser.add_argument("--auto", action='store_true', help='auto set folders')
    parser.add_argument("--save_dir", default="./ckpts")
    parser.add_argument("--plot_dir", default="./plots/learn")
    # model params
    parser.add_argument("--model", default="bert")
    # hparams
    parser.add_argument("--gamma", type=float, default=0.)
    return parser.parse_args()


def semi_crf_dp_and_map(out, N, gamma):
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
            seg_score = out[k-l][l]
            seg_score_with_len = seg_score + gamma*l
            # Score from k-l to k-1 is out[k-l][l]
            cand_part = log_a[k - l] + seg_score_with_len   # for log-partition
            alpha_candidates.append(cand_part)

            cand_viterbi = map_a[k - l] + seg_score_with_len  # for MAP
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


def obtain_span(t, a, b, bert=False):
    geo = t.token_geo(a, b)
    geo = {k: (x for x in geo[k]) for k in geo}
    # bd
    # bd, ba, bd
    # bd, ba, bd, (ba, da, bd)_n
    i = a%3
    j = a%3
    k = a%3
    span = [] 
    types = []      
    bdt = t.BOND_TYPES[i]
    # recall that feats are [BOND_TYPES, BOND_ANGLES, DIHEDRAL_ANGLES]
    # index 2 is BOND_TYPES[a%3]
    # index 1 is DIHEDRAL_ANGLES[(a%3+1)%3]
    # index 0 is BOND_ANGLES[(a%3+2)%3]
    # so BOND_TYPES[0] is 2+3*(3-(a%3))
    # so DIHEDRAL_ANGLES[0] is 1+3*(3-(a%3+1)%3)
    # so BOND_ANGLES[0] is 3*(3-(a%3+2)%3)
    if bert: #(ba, da, bd)_(n)
        b0 = 2+3*(3-(a%3))%3
        ba0 = 3*(3-(a%3+2)%3)%3
        da0 = 1+3*(3-(a%3+1)%3)%3
        permute = [b0, (b0+3)%9, (b0+6)%9, ba0, (ba0+3)%9, (ba0+6)%9, da0, (da0+3)%9, (da0+6)%9]
        span.append(0.)
        types.append(t.BOND_ANGLES[(j+2)%3])
        span.append(0.)
        types.append(t.DIHEDRAL_ANGLES[(k+1)%3])
        span.append(next(geo[bdt]))
        types.append(bdt)
    i += 1        
    if b > 1:
        bat = t.BOND_ANGLES[j%3]
        span.append(next(geo[bat]))
        types.append(bat)
        j += 1
        if bert:
            span.append(0.)
            types.append(None)
        bdt = t.BOND_TYPES[i%3]
        span.append(next(geo[bdt]))
        types.append(bdt)
        i += 1
    for idx in range(2, b):
        bat = t.BOND_ANGLES[j%3]
        span.append(next(geo[bat]))
        types.append(bat)
        j += 1
        bdt = t.DIHEDRAL_ANGLES[k%3]
        span.append(next(geo[bdt]))        
        types.append(bdt)
        k += 1
        bdt = t.BOND_TYPES[i%3]
        span.append(next(geo[bdt]))
        types.append(bdt)
        i += 1    
    if bert:
        # not a mult of 9? pad 0's
        if len(span)%9:
            span += [0 for _ in range(9-len(span)%9)]
        inv_permute = np.argsort(permute)
        return np.array(span).reshape(-1, 9)[:, permute], inv_permute
    else:
        return np.array(span)



def precompute(model, N, t, bert=False):
    out = [[None for _ in range(N+1)] for _ in range(N)]
    # compute
    if bert:
        span, inv_permute = obtain_span(t, 0, N, bert=bert)
    span_tensor = torch.tensor(span, dtype=torch.float32)
    last_dim = 9 if bert else 1
    span_tensor = span_tensor.reshape(1, -1, last_dim).to(model.device)
    dummy_attention_mask = torch.ones(1, len(span), dtype=torch.long).to(model.device)
    timestep = torch.ones(1, dtype=torch.long).to(model.device)
    encoded = model(inputs=span_tensor, timestep=timestep, attention_mask=dummy_attention_mask)[0]
    if not (encoded==encoded).all():
        breakpoint()
    # out_all_steps = out_all_steps.flatten()    
    for i in range(N):
        if i == 0:
            substr_embedding = encoded
        else:
            if bert:
                substr_embedding = encoded[i//3:]
            else:
                substr_embedding = torch.cat((encoded[3*i-1:3*i+1], encoded[3*i+2:])) # rid extra dihedral angle
        for j in range(i+1, N+1):
            if j-i == 1: # len(span) == 1          
                if bert:
                    span_embedding = substr_embedding[:(j-i+2)//3]
                else:
                    span_embedding = substr_embedding[:1]              
            elif j-i == 2: # len(span) == 2
                if bert:
                    span_embedding = substr_embedding[:(j-i+2)//3]
                else:
                    span_embedding = substr_embedding[:2]
            else: # len(span) == 3
                if bert:
                    span_embedding = substr_embedding[:(j-i+2)//3]
                else:
                    span_embedding = substr_embedding[:2+3*(j-i-2)]
            span_embedding_agg = span_embedding.mean(axis=0)
            score = model.fc(span_embedding_agg)
            if score != score:
                breakpoint()
            out[i][j-i] = score            
    return out


def get_model(args):
    if args.model == "transformer":
        model = AngleTransformer(d_model=32, nhead=4, num_layers=2, max_len=max_len)
    elif args.model == "bert":
        ft_is_angular = [False, False, False, True, True, True, True, True, True]  # e.g., for [phi, psi, omega, tau]
        ft_names = Tokenizer.BOND_TYPES+Tokenizer.BOND_ANGLES+Tokenizer.DIHEDRAL_ANGLES
        time_encoding = "gaussian_fourier"  # or "sinusoidal"
        decoder = "mlp"  # or "linear"
        config = BertConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
            output_attentions=False,
            output_hidden_states=False,
            out_dim=32
        )
        # 3. Instantiate the model.
        model = BertForDiffusionBase(
            config=config,
            ft_is_angular=ft_is_angular,
            ft_names=ft_names,
            time_encoding=time_encoding,
            decoder=decoder,
            scalar_out=True
        )        
    else:
        raise NotImplementedError
    return model


def main():
    args = parse_args()
    logging.info(args)
    if args.auto:
        cur_time = time.time()
        setattr(args, 'plot_dir', f'./plots/learn/{cur_time}')
        setattr(args, 'save_dir', f'./ckpts/{cur_time}')
        os.makedirs(args.plot_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        logging.info(f"{args.plot_dir}")
    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logging.info(f"Using device: {args.cuda}")    
    dataset = FullCathCanonicalCoordsDataset(args.cath_folder, use_cache=False, debug=args.debug, zero_center=False, toy=args.toy, secondary=False, trim_strategy="discard")
    dataset = [Tokenizer(x) for x in dataset.structures]
    max_len = max([3*(3*t.n-1)-2 for t in dataset])
    model = get_model(args)
    model.to(args.cuda)
    model.to(args.cuda)
    first_param = next(model.parameters())
    logging.info(f"Model device: {first_param.device}")    
    opt = optim.Adam(model.parameters())        
    best_scores = [float("-inf") for _ in range(len(dataset))]
    probs = [0. for _ in range(len(dataset))]
    best_loss = float("inf")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.
        for _idx, t in tqdm(enumerate(dataset), desc=f"epoch {epoch}"):
            opt.zero_grad()
            loss = 0.         
            N = 3*t.n-1             
            out = precompute(model, N, t, bert=args.model=="bert")
            log_a, map_a, best_lens = semi_crf_dp_and_map(out, N, gamma=args.gamma)
            best_seg = backtrace_map_segmentation(best_lens, N)            
            t.bond_to_token = {start: (start, idx, end-start) for idx, (start, end) in enumerate(best_seg)}            
            loss += -log_a[N]
            a_loss = loss.item()
            penalty_loss = 0.01*l1_penalty(model)
            loss += penalty_loss
            start_time = time.perf_counter()            
            loss.backward()
            logging.info(f"loss backward took {time.perf_counter() - start_time:.4f} seconds")
            # print(f"-log a: {a_loss}, l1 penalty: {penalty_loss.item()}")
            opt.step()

            # Log GPU memory usage:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                logging.info(f"Epoch {epoch}, Iteration {_idx} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
            
            # (Optional) Log CPU memory usage:            
            process = psutil.Process(os.getpid())
            cpu_mem = process.memory_info().rss / (1024 ** 2)  # in MB
            logging.info(f"Epoch {epoch}, Iteration {_idx} - CPU memory usage: {cpu_mem:.2f} MB")           
            prob = torch.exp(map_a[N]-log_a[N])
            if prob.item() > probs[_idx]:
                probs[_idx] = prob.item()
                t.visualize(os.path.join(args.plot_dir, f"epoch={epoch}_{_idx}_{probs[_idx]}.png"), vis_dihedral=False)                        
            # if (map_a[N]-log_a[N]) > best_scores[_idx]:                
            #     # t.visualize(os.path.join(args.plot_dir, f"epoch={epoch}_{_idx}_{prob.item()}.png"), vis_dihedral=False)                
            #     best_scores[_idx] = map_a[N]-log_a[N]
            #     probs[_idx] = prob
            #     print("Max prob:", prob)
            total_loss += loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            logging.info(f"total loss: {total_loss}")
            torch.save(model, os.path.join(args.save_dir, f"{epoch}_loss={best_loss}.pt"))
   

if __name__ == "__main__":
    main()
