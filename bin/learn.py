from foldingdiff.potential_model import *
from foldingdiff.modelling import *
from foldingdiff.tokenizer import Tokenizer
from foldingdiff.plotting import plot_feature_importance
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
from pathlib import Path

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
    parser.add_argument("--model", default="bert", choices=["bert", "transformer", "feats"])
    # hparams
    parser.add_argument("--gamma", type=float, default=0.)
    parser.add_argument("--max-seg-len", type=int, default=1e10, help="Max length of segment")
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
        # inv_permute = np.argsort(permute)
        # return np.array(span).reshape(-1, 9)[:, permute], inv_permute
        return np.array(span).reshape(-1, 9)[:, permute]
    else:
        return np.array(span)


def get_model(args, max_len):
    if args.model == "transformer":
        encoder = AngleTransformer(d_model=32, nhead=4, num_layers=2, max_len=max_len)
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
        encoder = BertForDiffusionBase(
            config=config,
            ft_is_angular=ft_is_angular,
            ft_names=ft_names,
            time_encoding=time_encoding,
            decoder=decoder,
            scalar_out=True
        )        
    else:
        encoder = None
    # === residue‑feat extractor and segment potential ===
    res_feat = BackboneResidueFeaturizer()
    agg      = SegmentFeatureAggregator(res_feat.out_dim, res_feat.labels)
    seg_mlp  = SegmentPotentialMLP(agg.out_dim, hidden=64)
    # Wrap everything in a single object (lightweight example)
    model = SemiCRFModel(
        encoder=encoder,
        res_featurizer=res_feat,
        seg_aggregator=agg,
        seg_potential=seg_mlp,
        length_bias=args.gamma,          # γ from your DP
        max_seg_len=args.max_seg_len,
        device=args.cuda,
        num_workers=0 if args.debug else 20
    )
    return model.to(args.cuda)


# ---------------------------------------------------------------------
# main training function – rewritten for SemiCRFModel
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    logging.info(args)

    # ---------------- output folders ---------------------------------
    if args.auto:
        cur_time   = time.time()
        args.plot_dir = f'./plots/learn/{cur_time}'
        args.save_dir = f'./ckpts/{cur_time}'
        os.makedirs(args.plot_dir, exist_ok=True)
        os.makedirs(args.save_dir,  exist_ok=True)
        logging.info(f"plots : {args.plot_dir}")
        logging.info(f"ckpts : {args.save_dir}")

    logging.info(f"CUDA available : {torch.cuda.is_available()}")
    logging.info(f"Using device   : {args.cuda}")

    # ---------------- load data --------------------------------------
    raw_ds = FullCathCanonicalCoordsDataset(
        args.cath_folder, use_cache=False, debug=args.debug,
        zero_center=False, toy=args.toy, secondary=False,
        trim_strategy="discard"
    )
    dataset = [Tokenizer(x) for x in raw_ds.structures]

    # maximum token length for Transformer positional encodings
    max_len = max([3 * (3 * t.n - 1) - 2 for t in dataset])

    # ---------------- build model ------------------------------------
    model = get_model(args, max_len=max_len)           # returns SemiCRFModel
    model.to(args.cuda)
    logging.info(f"Model allocated on {next(model.parameters()).device}")

    opt = optim.Adam(model.parameters())

    best_loss = float("inf")
    model.train()

    # ---------------- precompute if needed ------------------------------------
    if args.model == "feats":
        model.compute_feats(dataset)

    # -----------------------------------------------------------------
    for epoch in range(args.epochs):
        total_loss = 0.0
        for idx, t in tqdm(enumerate(dataset), desc=f"epoch {epoch}"):
            opt.zero_grad()

            # --------- build “angles” tensor the same way as before ---
            if args.model == "feats": # residue level
                N = t.n
                assert t.n == len(t.aa), "number of residues != length of amino acid sequence"
                coords = t.compute_coords()
                prot_id = Path(t.fname).stem
                # --------- call SemiCRFModel.precompute -------------------
                out, attn_scores = model.precompute(
                    prot_id       = prot_id,
                    aa_seq        = t.aa,              # Tokenizer stores AA sequence
                    coords_tensor = coords
                )                                       # out[i][l] ready for DP
            else:
                N = 3 * t.n - 1 # bond level
        
                bert_flag = args.model == "bert"
                span = obtain_span(t, 0, N, bert=bert_flag)

                in_feat = 9 if bert_flag else 1
                span_tensor = (
                    torch.tensor(span, dtype=torch.float32)
                        .reshape(1, -1, in_feat)
                        .to(args.cuda)
                )
                attention_mask = torch.ones(1, span_tensor.size(1),
                                            dtype=torch.long, device=args.cuda)
                timestep = torch.ones(1, 1, dtype=torch.long, device=args.cuda)

                # --------- call SemiCRFModel.precompute -------------------
                out, attn_scores = model.precompute(
                    aa_seq        = t.aa,              # Tokenizer stores AA sequence
                    angles_tensor = span_tensor,
                    timestep      = timestep,
                    attention_mask= attention_mask
                )                                       # out[i][l] ready for DP

            # ---------- forward + Viterbi on semi‑CRF -----------------
            log_a, map_a, best_lens = semi_crf_dp_and_map(out, N, gamma=args.gamma)
            best_seg = backtrace_map_segmentation(best_lens, N)
            attn_stack = torch.stack([attn_scores[start][end-start] for start, end in best_seg], axis=0)
            attn_agg = attn_stack.mean(axis=0)            
            
            if args.model == "feats":
                # store segmentation in Tokenizer (as before)
                t.bond_to_token = {3*start: (3*start, 3*seg_idx, min(3*(end-start), 3*t.n-1-3*start))
                                for seg_idx, (start, end) in enumerate(best_seg)}                
            else:
                # store segmentation in Tokenizer (as before)
                t.bond_to_token = {start: (start, seg_idx, end - start)
                                for seg_idx, (start, end) in enumerate(best_seg)}

            # --------------- loss & optimisation ----------------------
            loss   = -log_a[N]                       # negative log‑partition
            loss   = loss + 0.01 * l1_penalty(model) # optional L1 reg
            loss.backward()
            opt.step()

            total_loss += loss.item()

            # --------------- (optional) diagnostics -------------------
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserv = torch.cuda.memory_reserved() / 1024**2
                logging.info(f"E{epoch} I{idx}  GPU MB  alloc {alloc:.1f}  reserv {reserv:.1f}")

            cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            logging.info(f"E{epoch} I{idx}  CPU MB {cpu_mem:.1f}")

            # segmentation probability
            prob = torch.exp(map_a[N] - log_a[N]).item()
            if prob > 0.5:   # arbitrary threshold for visualisation
                path = Path(os.path.join(args.plot_dir, f"epoch={epoch}_idx={idx}_p={prob:.3f}.png"))
                attn_path = path.with_name(path.stem + "_attn" + path.suffix)
                t.visualize(path,
                                vis_dihedral=False)
                plot_feature_importance(attn_agg.detach().cpu().numpy(), model.aggregator.per_res_labels, attn_path)

        # ---------------- checkpoint if improved ---------------------
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"epoch={epoch}_loss={best_loss:.4f}.pt"))
            logging.info(f"[epoch {epoch}] new best total loss {best_loss:.4f}")



if __name__ == "__main__":
    breakpoint()
    main()
