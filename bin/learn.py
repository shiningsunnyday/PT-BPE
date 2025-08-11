from foldingdiff.potential_model import *
# from foldingdiff.modelling import *
# from foldingdiff.old_potential_model import *
from foldingdiff.tokenizer import Tokenizer
from foldingdiff.plotting import plot_feature_importance
from foldingdiff.datasets import FullCathCanonicalCoordsDataset
from foldingdiff.utils import validate_args_match
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import gc
import psutil
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from tqdm import tqdm
import time
import argparse
import os
import logging
import inspect
import pickle
import json
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
    default_max_seg_len = 10000000000
    parser = argparse.ArgumentParser(description="FoldingDiff BPE Script")
    # general params
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--visualize", action='store_true')  
    parser.add_argument("--auto", action='store_true', help='auto set folders')    
    parser.add_argument("--save-dir")  
    # data params
    parser.add_argument("--data-dir", type=str, default="cath", choices=[
         'cath', 'homo', 'ec', "bindint", "bindbio", "repeat", "catint", "catbio", "conserved", "test"
         ], help="Which dataset.")    
    parser.add_argument("--toy", type=int, default=10, 
                            help="Number of PDB files.")
    parser.add_argument("--pad", type=int, default=512, help="Max protein size")
    # training params
    parser.add_argument("--epochs", type=int, default=10)    
    parser.add_argument("--cuda", default="cpu")
    # feat preprocessing params
    parser.add_argument("--config", help='file to config compute_feats')
    parser.add_argument("--batch_size", help='batch size for feat computation checkpointing', type=int, default=100)    
    # model params
    parser.add_argument("--model", default="bert", choices=["bert", "transformer", "feats"])
    parser.add_argument("--mode", help="potential function mode, unary: 1D semi-CRF, binary: 2D, recursive: decomposition", choices=["unary", "binary", "recursive"])
    # hparams
    parser.add_argument("--gamma", type=float, default=0.)
    parser.add_argument("--l1", type=float, default=0.01)
    parser.add_argument("--max-seg-len", type=int, default=default_max_seg_len, help="Max length of segment")
    args = parser.parse_args()
    if args.mode == "unary" and args.max_seg_len != default_max_seg_len:
        parser.error(f"--max-seg-len should not be passed for unary mode")
    return args


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


def semi_crf_dp_and_map_2d(
    unary_scores: torch.Tensor,
    edge_scores: torch.Tensor,
    N: int,
    Lmax: int,
    gamma: float = 0.0
):
    """
    2D semi-CRF forward: returns
      - log_alpha[k, l]: log-partition up to position k with last segment length l
      - map_alpha[k, l]: Viterbi max-score up to k with last segment length l
      - backpointer[k, l]: the argmax previous length for map_alpha[k, l]

    Args:
      unary_scores: Tensor[N, Lmax+1]    unary log-potentials φ(x_{i:i+l})
      edge_scores:  Tensor[N, Lmax+1, Lmax+1]
                     edge_scores[i, lp, l] = ψ(segment ending at i of length lp,
                                                  segment starting at i of length l)
      N:            sequence length
      Lmax:         maximum segment length
      gamma:        length bonus per segment (added to unary)
    """
    device = unary_scores.device
    # Initialize DP tables
    log_alpha = torch.full((N+1, Lmax+1), float('-inf'), device=device)
    map_alpha = torch.full((N+1, Lmax+1), float('-inf'), device=device)
    backpointer = torch.zeros((N+1, Lmax+1), dtype=torch.long)

    # Base case: empty prefix → no segments
    log_alpha[0, 0] = 0.0
    map_alpha[0, 0] = 0.0

    for k in range(1, N+1):
        max_l = min(Lmax, k)
        for l in range(1, max_l+1):
            i = k - l
            # unary + length bonus
            u = unary_scores[i, l] + gamma * l

            # consider all possible previous segment lengths lp
            prev_max_lp = min(Lmax, i)
            prev_log = log_alpha[i, :prev_max_lp+1]    # (lp=0..prev_max_lp)
            prev_map = map_alpha[i, :prev_max_lp+1]
            e = edge_scores[i, :prev_max_lp+1, l]      # same range for lp

            # log-partition update
            candidates_log = prev_log + u + e
            log_alpha[k, l] = torch.logsumexp(candidates_log, dim=0)

            # Viterbi (MAP) update
            candidates_map = prev_map + u + e
            best_val, best_lp = torch.max(candidates_map, dim=0)
            map_alpha[k, l] = best_val
            backpointer[k, l] = best_lp

    return log_alpha, map_alpha, backpointer


def hierarchical_inside_and_map(
    unary_scores: torch.Tensor,
    split_scores: torch.Tensor,
    N: int,
    Lmax: int = None,
    gamma: float = 0.0
):
    """
    Returns:
      dp_inside[i,j]: log‐partition for span [i,j)
      dp_map[i,j]:    Viterbi best score for span [i,j)
      backptr[i,j]:   best split k (or -1 if leaf)

    Caps *only* the leaf option at Lmax; splits can build arbitrarily large spans.
    """
    device = unary_scores.device

    # Allocate tables
    dp_inside = torch.full((N+1, N+1), -float('inf'), device=device)
    dp_map    = torch.full((N+1, N+1), -float('inf'), device=device)
    backptr   = torch.full((N+1, N+1), -1,        dtype=torch.long, device=device)

    # Base case: empty spans
    for i in range(N+1):
        dp_inside[i, i] = 0.0
        dp_map[i, i]    = 0.0

    # Fill spans of width d=1..N
    for d in range(1, N+1):
        for i in range(0, N - d + 1):
            j = i + d

            # 1) LEAF option (only if d<=Lmax)
            if Lmax is None or d <= Lmax:
                leaf_score   = unary_scores[i, d] + gamma * d
                best_map_val = leaf_score
                best_k       = -1
                inside_terms = [leaf_score]
            else:
                best_map_val = -float('inf')
                best_k       = -1
                inside_terms = []

            # 2) SPLIT options (always allowed, no child-length cap)
            for k in range(i+1, j):
                # score for splitting (i,j) → (i,k) + (k,j)
                split_val = split_scores[i, k-i, j-k]

                # Viterbi (MAP) update
                cand_map = split_val + dp_map[i, k] + dp_map[k, j]
                if cand_map > best_map_val:
                    best_map_val = cand_map
                    best_k       = k

                # Inside‐sum update
                inside_terms.append(
                    split_val
                    + dp_inside[i, k]
                    + dp_inside[k, j]
                )

            # Write out DP cells
            dp_map[i, j]    = best_map_val
            backptr[i, j]   = best_k
            dp_inside[i, j] = torch.logsumexp(
                                  torch.stack(inside_terms),
                                  dim=0
                               )

    return dp_inside, dp_map, backptr


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


def backtrace_map_segmentation_2d(
    backpointer: torch.Tensor,
    map_alpha: torch.Tensor,
    N: int
):
    """
    Recover MAP segmentation from the 2D Viterbi tables.

    Args:
      backpointer: LongTensor[N+1, Lmax+1] from semi_crf_dp_and_map_2d
      map_alpha:   Tensor[N+1, Lmax+1]       from semi_crf_dp_and_map_2d
      N:           sequence length

    Returns:
      segmentation: List of (start, end) tuples for the best path.
    """
    # 1) pick best final length at position N
    final_scores = map_alpha[N]                  # shape (Lmax+1)
    l = torch.argmax(final_scores).item()

    segs = []
    k = N
    while k > 0 and l > 0:
        start = k - l
        segs.append((start, k))
        prev_l = backpointer[k, l].item()
        k = start
        l = prev_l

    return list(reversed(segs))


def backtrace_hierarchy(
    backptr: torch.Tensor,
    i: int,
    j: int
):
    """
    Recursively backtrace the best hierarchical decomposition for span [i,j).

    Args:
      backptr: LongTensor from hierarchical_dp_and_map
      i, j:    span boundaries

    Returns:
      List of tuples (start, end) for leaf spans in the final tree, in pre-order.
    """
    k = backptr[i, j].item()
    # If k == -1, it's a leaf
    if k < 0:
        return [(i, j)]

    # Otherwise split into two children
    left = backtrace_hierarchy(backptr, i, k)
    right = backtrace_hierarchy(backptr, k, j)
    return left + right


def construct_hierarchy(
    t: Tokenizer,
    backptr: torch.Tensor,
    i: int,
    j: int
):
    """
    Recursively build the best hierarchical decomposition for span [i,j).

    Args:
      t: tokenizer
      backptr: LongTensor from hierarchical_dp_and_map
      i, j:    span boundaries
    """
    k = backptr[i, j].item()
    # If k == -1, it's a leaf
    if k < 0:
        # leaf
        if t.bond_to_token is None:
            t.bond_to_token = {}
        t.bond_to_token[3*i] = (3*i, 0, 3*(j-i))
        return [(i, j)]

    # Otherwise split into two children
    left = construct_hierarchy(t, backptr, i, k)
    right = construct_hierarchy(t, backptr, k, j)
    t.bond_to_token.pop(3*k)
    t.bond_to_token[3*i] = (3*i, 0, 3*(j-i))




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


def get_model(args, device, max_len, config=None):
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
    # Wrap everything in a single object (lightweight example)
    if args.mode == "unary":
        model = SemiCRFModel(
            config=config,
            encoder=encoder,
            length_bias=args.gamma,          # γ from your DP
            max_seg_len=args.max_seg_len,
            device=device
        )
    else:
        model = SemiCRF2DModel(
            config=config,
            length_bias=args.gamma,          # γ from your DP
            max_seg_len=args.max_seg_len,
            device=device
        )    
    return model.to(device)


class FeatDataset(Dataset):
    def __init__(self, tokenizers, config, save_dir):
        self.tokenizers = tokenizers
        self.config = config
        self.save_dir = save_dir
        for t in self.tokenizers:
            prot_id = Path(t.fname).stem
            assert os.path.exists(os.path.join(save_dir, f"{prot_id}.pkl"))

    
    def __len__(self):
        return len(self.tokenizers)


    def __getitem__(self, idx):
        t = self.tokenizers[idx]
        prot_id = Path(t.fname).stem
        path = os.path.join(self.save_dir, f"{prot_id}.pkl")
        try:
            feats = pickle.load(open(path, "rb"))
        except Exception as e:
            # this will go straight to stderr and flush immediately
            print(f"Failed to unpickle {path}: {e}", file=sys.stderr, flush=True)
            raise
        for feat_type in feats:
            if feat_type not in self.config or not self.config[feat_type]["enabled"]:
                feats.pop(feat_type)
        return idx, t, feats


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]

def find_latest_checkpoint(save_dir: str) -> Path:
    save_path = Path(save_dir)
    # Grab all files matching your naming pattern
    cks = list(save_path.glob("epoch=*_*loss=*.pt"))
    if not cks:
        raise FileNotFoundError(f"No checkpoints found in {save_dir}")

    # Option 1: sort by epoch number embedded in the filename
    def epoch_num(p: Path):
        m = re.search(r"epoch=(\d+)", p.name)
        return int(m.group(1)) if m else -1

    latest_by_epoch = max(cks, key=epoch_num)

    # Option 2: alternatively, sort by file‐modified time
    # latest_by_epoch = max(cks, key=lambda p: p.stat().st_mtime)

    return latest_by_epoch, epoch_num(latest_by_epoch)

# ---------------------------------------------------------------------
# main training function – rewritten for SemiCRFModel
# ---------------------------------------------------------------------
def main(args) -> None:
    logging.info(args)

    # ---------------- output folders ---------------------------------
    if args.save_dir:
        save_dir = Path(args.save_dir)
        name = save_dir.name
        assert os.path.exists(save_dir)
        plot_dir = f'./plots/learn/{name}'
        os.makedirs(plot_dir, exist_ok=True)        
        setattr(args, 'plot_dir', plot_dir)
        resume = True
    elif args.auto:
        cur_time = time.time()
        setattr(args, 'plot_dir', f'./plots/learn/{cur_time}')
        setattr(args, 'save_dir', f'./ckpts/{cur_time}')
        os.makedirs(args.plot_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)        
        logging.info(f"plots : {args.plot_dir}")
        logging.info(f"ckpts : {args.save_dir}")        
        resume = False
    else:
        raise NotImplementedError
    
    args_path = os.path.join(args.save_dir, 'args.json')    
    if resume:
        assert os.path.exists(args_path)
        print(f"loading args from {args_path}")
        loaded_args = json.load(open(args_path))
        if not args.debug:
            validate_args_match(
                current   = args,
                loaded    = loaded_args,
                skip      = ["debug", "visualize", "auto", "config", \
                "epochs", "gamma", "l1", "max-seg-len"],   # fields you don’t need to compare
            )
    else:        
        json.dump(args.__dict__, open(args_path, 'w+'))        
    logging.info(f"CUDA available : {torch.cuda.is_available()}")    

    # ---------------- load data --------------------------------------
    raw_ds = FullCathCanonicalCoordsDataset(
        args.data_dir, use_cache=False, debug=args.debug,
        zero_center=False, toy=args.toy, pad=args.pad, secondary=False,
        trim_strategy="discard"
    )
    dataset = [Tokenizer(x) for x in raw_ds.structures]

    # maximum token length for Transformer positional encodings
    max_len = max([3 * (3 * t.n - 1) - 2 for t in dataset])

    # ---------------- build device ------------------------------------
    # --- 1) Detect distributed or not ---
    is_ddp = False
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        is_ddp = True
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # single‐GPU (or CPU) fallback
        device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")     

    # ---------------- build model ------------------------------------
    logging.info(f"Using device   : {device}")       
    if args.model == "feats":        
        if args.config:
            config = json.load(open(args.config))
            config_args_path = os.path.join(args.save_dir, 'config.json')
            if resume:   
                assert os.path.exists(config_args_path)             
                loaded_config = json.load(open(config_args_path))
                try:
                    validate_args_match(config, loaded_config)
                except AssertionError as e:
                    logging.error(e)
            else:
                json.dump(config, open(config_args_path, 'w+'))        
            model = get_model(args, device, max_len=max_len, config=config)           # returns SemiCRFModel
        else:
            model = get_model(args, device, max_len=max_len)
    else:
        model = get_model(args, device, max_len=max_len)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    logging.info(f"Model allocated on {next(model.parameters()).device}")
    opt = optim.Adam(model.parameters())
    best_loss = float("inf")
    model.train()

    # ---------------- precompute if needed ------------------------------------
    if args.model == "feats":
        # compute feats in batches            
        if is_ddp:
            obj = model.module
        else:
            obj = model
        obj.compute_batch_feats(dataset, config, save_dir=args.save_dir, batch_size=args.batch_size)        
        dataset = FeatDataset(dataset, config, args.save_dir)
        # dataset = DataLoader(dataset, collate_fn=collate_fn, batch_size=1, num_workers=4, prefetch_factor=1, persistent_workers=True, pin_memory=True)
        if is_ddp:
            sampler = DistributedSampler(dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True        
        dataset = DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0 if args.debug else 4,
            prefetch_factor=None if args.debug else 2,
            persistent_workers=False if args.debug else True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    # -----------------------------------------------------------------
    try:
        checkpoint_path, epoch = find_latest_checkpoint(args.save_dir)   
        ckpt = torch.load(checkpoint_path, map_location=model.device)         
        if 'model_state' in ckpt:
            obj.load_state_dict(ckpt['model_state'])
            opt.load_state_dict(ckpt['optim_state'])
            best_loss = ckpt['best_loss']            
            assert ckpt['epoch'] == epoch
            start_epoch = ckpt['epoch'] + 1
        else:
            obj.load_state_dict(ckpt)
            best_loss = float('inf')
            start_epoch = epoch + 1
    except FileNotFoundError:
        best_loss = float('inf')
        start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        for _, (idx, t, feats) in tqdm(enumerate(dataset), desc=f"epoch {epoch}"):
            opt.zero_grad()

            # --------- build “angles” tensor the same way as before ---
            if args.model == "feats": # residue level
                N = t.n
                assert t.n == len(t.aa), "number of residues != length of amino acid sequence"
                coords = t.compute_coords()
                # --------- call SemiCRFModel.precompute -------------------
                if args.mode == "unary":
                    out, attn_scores = obj.precompute(
                        feats         = feats,
                        aa_seq        = t.aa,              # Tokenizer stores AA sequence
                        coords_tensor = coords
                    )    
                elif args.mode == "binary":
                    try:
                        unary_out, unary_attn_scores, edge_out, edge_attn_scores = obj.precompute(
                            feats         = feats,
                            aa_seq        = t.aa              # Tokenizer stores AA sequence
                        )  
                    except Exception as e:
                        print(e)
                        print(t.fname)
                        raise                                                       # out[i][l] ready for DP
                else:
                    try:
                        unary_out, unary_attn_scores, edge_out, edge_attn_scores = obj.precompute(
                            feats         = feats,
                            aa_seq        = t.aa              # Tokenizer stores AA sequence
                        )  
                    except Exception as e:
                        print(e)
                        print(t.fname)
                        raise    
            else:
                N = 3 * t.n - 1 # bond level        
                bert_flag = args.model == "bert"
                span = obtain_span(t, 0, N, bert=bert_flag)

                in_feat = 9 if bert_flag else 1
                span_tensor = (
                    torch.tensor(span, dtype=torch.float32)
                        .reshape(1, -1, in_feat)
                        .to(device)
                )
                attention_mask = torch.ones(1, span_tensor.size(1),
                                            dtype=torch.long, device=device)
                timestep = torch.ones(1, 1, dtype=torch.long, device=device)

                # --------- call SemiCRFModel.precompute -------------------
                out, attn_scores = obj.precompute(
                    aa_seq        = t.aa,              # Tokenizer stores AA sequence
                    angles_tensor = span_tensor,
                    timestep      = timestep,
                    attention_mask= attention_mask
                )                                       # out[i][l] ready for DP

            # ---------- forward + Viterbi on semi‑CRF -----------------
            if args.mode == "unary":
                log_a, map_a, best_lens = semi_crf_dp_and_map(out, N, gamma=args.gamma)
                best_seg = backtrace_map_segmentation(best_lens, N)
                attn_stack = torch.stack([attn_scores[start][end-start] for start, end in best_seg], axis=0)
                attn_agg = attn_stack.mean(axis=0)            
            elif args.mode == "binary":
                log_alpha, map_alpha, backpointer = semi_crf_dp_and_map_2d(unary_out, edge_out, N, args.max_seg_len, args.gamma)
                best_seg = backtrace_map_segmentation_2d(backpointer, map_alpha, N)
                attn_stack = torch.stack([unary_attn_scores[start][end-start] for start, end in best_seg], axis=0)
                attn_agg = attn_stack.mean(axis=0)                                            
            else:
                dp_inside, dp_map, backptr = hierarchical_inside_and_map(unary_out, edge_out, N, args.max_seg_len, args.gamma)
                best_tree = backtrace_hierarchy(backptr, 0, N)
                logging.info("best tree", best_tree)
            
            if args.model == "feats":
                if args.mode == "recursive":
                    # navigate hierarchy
                    construct_hierarchy(t, backptr, 0, N)
                else:
                    # store segmentation in Tokenizer (as before)
                    t.bond_to_token = {3*start: (3*start, 3*seg_idx, min(3*(end-start), 3*t.n-1-3*start))
                                    for seg_idx, (start, end) in enumerate(best_seg)}                                
            else:
                # store segmentation in Tokenizer (as before)
                t.bond_to_token = {start: (start, seg_idx, end - start)
                                for seg_idx, (start, end) in enumerate(best_seg)}

            # --------------- loss & optimisation ----------------------
            if args.mode == "unary":
                loss   = -log_a[N]                       # negative log‑partition
            elif args.mode == "binary":
                # 1) The log‐partition Z(x) = log ∑ₗ exp(log_alpha[N, ℓ])
                logZ = torch.logsumexp(log_alpha[N], dim=0)          # scalar

                # 2) The best MAP score = maxₗ map_alpha[N, ℓ]
                best_map, _ = torch.max(map_alpha[N], dim=0)         # scalar
                loss = -logZ                
            else:
                # compute log Z here and the loss
                logZ = dp_inside[0, N]  
                loss = -logZ           

            loss   = loss + args.l1 * l1_penalty(model) # optional L1 reg
            loss.backward()
            opt.step()

            total_loss += loss.item()

            # --------------- (optional) diagnostics -------------------
            is_main = (not is_ddp) or dist.get_rank() == 0
            if is_main:
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1024**2
                    reserv = torch.cuda.memory_reserved() / 1024**2
                    logging.info(f"E{epoch} I{idx}  GPU MB  alloc {alloc:.1f}  reserv {reserv:.1f}")
                
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
                logging.info(f"E{epoch} I{idx}  CPU MB {cpu_mem:.1f}")

            # segmentation probability
            if args.mode == "unary":                
                prob = torch.exp(map_a[N] - log_a[N]).item()
            elif args.mode == "binary":
                # 3) Probability of that MAP segmentation:
                prob = torch.exp(best_map - logZ)                # scalar in [0,1]                
            else:
                # Use dp for the MAP segmentation probability
                best_dp = dp_map[0, N]
                prob = torch.exp(best_dp - logZ).item()
            logging.info(f"epoch {epoch} idx {idx} prob {prob}")
            if prob > 0.5:   # arbitrary threshold for visualisation
                path = Path(os.path.join(args.save_dir, f"epoch={epoch}_idx={idx}_p={prob:.3f}.png"))
                attn_path = path.with_name(path.stem + "_attn" + path.suffix)
                t.visualize(path, vis_dihedral=False)
                plot_feature_importance(attn_agg.detach().cpu().numpy(), obj.aggregator.per_res_labels, attn_path)
                if args.mode == "recursive":
                    path = Path(os.path.join(args.save_dir, f"epoch={epoch}_idx={idx}_p={prob:.3f}.png"))
                    t.bond_to_token.tree.visualize(path, horizontal_gap=0.5, font_size=6)

        # ---------------- checkpoint if improved ---------------------
        if (not is_ddp) or dist.get_rank() == 0:
            if total_loss < best_loss:
                best_loss = total_loss
                checkpoint = {
                    'epoch':         epoch,
                    'model_state':   obj.state_dict(),
                    'optim_state':   opt.state_dict(),
                    'best_loss':     best_loss,
                    # optionally: 'scheduler_state': scheduler.state_dict(),
                    # you can add any other metadata you like
                }

                filename = os.path.join(
                    args.save_dir,
                    f"epoch={epoch}_loss={best_loss:.4f}.pt"
                )
                torch.save(checkpoint, filename)
                logging.info(f"[epoch {epoch}] saved new best checkpoint (loss={best_loss:.4f})")

    if is_ddp:
        dist.destroy_process_group()    


if __name__ == "__main__":
    args = parse_args()
    main(args)        
