import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
from pathlib import Path
import wandb
from collections import Counter
import random
import logging
from tqdm import tqdm
from foldingdiff.datasets import extract_backbone_coords
from foldingdiff.tokenizer import *
from foldingdiff.metrics import *
# -----------------------------
# 0) ARGPARSE
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Protein Motif LM Training")
    parser.add_argument(
        "--wandb_team", type=str, default="msun415",
        help="Weights & Biases team name"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="protein-motif-lm",
        help="Weights & Biases project name"
    )
    parser.add_argument("--task", choices=["remote-homology-detection", # per-protein
                                            "structural-flexibility-prediction", # per-residue regression
                                            "BindInt",
                                            "BindBio",
                                            "CatInt",
                                            "CatBio",
                                            "conserved-site-prediction",
                                            "repeat-motif-prediction",
                                            "epitope-region-prediction", # per-residue classification
    ], default="remote-homology-detection")    
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to pickle file containing tokenized sequences"
    )
    parser.add_argument(
        "--labels_path", type=str,
        help="Path to csv file containing processed probe labels"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run minimal reproducible example for debugging purposes only"
    )
    parser.add_argument(
        "--inference", action="store_true",
        help="Skip training and run final eval on test splits"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to a saved model checkpoint for inference mode"
    )    

   # sampling‐specific args:
    parser.add_argument("--num_samples", type=int, default=2,
                        help="How many uncdonditional sequences to sample")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling softmax temperature")    

    # model & training hyperparameters
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--num_layers",   type=int,   default=8)
    parser.add_argument("--hidden_dims",  type=int,   nargs='+', default=[32])
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--num_heads",    type=int,   default=8)
    parser.add_argument("--d_ff",         type=int,   default=1024)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--eval_interval",type=int,   default=500)
    parser.add_argument("--probe_layer",  type=int,   default=4)
    parser.add_argument("--probe_epochs", type=int,   default=5)
    parser.add_argument("--probe_lr",     type=float, default=1e-3)

    return parser.parse_args()

# -----------------------------
# 1) DATASET
# -----------------------------
class ProteinDataset(Dataset):
    def __init__(self, bpe, labels_path, task):
        """
        lables_path: csv for labels to structures in ckpt_path
        """        
        self.task = task
        self.vocab_size = bpe.vocab_size
        self.label_set = set()
        logging.info(f"VOCAB SIZE: {self.vocab_size}")
        if labels_path:
            df = pd.read_csv(labels_path)              
            self.do_probe = True
        else:
            df = None
            self.do_probe = False
        self.seqs = []
        self.fnames = []
        self.probe = []
        self.probe_split = []
        count = 0
        for t in bpe.tokenizers:
            tokenized = t.tokenize()
            self.fnames.append(t.fname)
            seq = bpe.quantize(tokenized)
            if self.do_probe:
                pdb_chain = Path(t.fname).stem.split('_')
                if len(pdb_chain) != 2:
                    raise ValueError(f"{t.fname} should be pdbid_chainid")
                pdb_id = pdb_chain[0]
                chain_id = pdb_chain[1]
                mask = (df.pdb_id == pdb_id) & (df.chain_id == chain_id)
                result = df.loc[mask]
                if len(result) >= 1:
                    row = result.iloc[0]
                    if self.task == "remote-homology-detection":
                        label = row["fold_label"]
                        self.label_set.add(label)
                        split = row["split"]
                    else:
                        label = eval(row["residue_label"])
                        self.label_set |= set(label)
                        if len(label) == t.n:                        
                            motif_label = []
                            for (start, _, length) in t.bond_to_token.values():
                                counts = Counter(label[start//3: (start+length+1)//3])
                                key, _ = counts.most_common(1)[0]
                                motif_label.append(key)
                                if start+length < 3*t.n-1:
                                    motif_label.append(None)
                                    motif_label.append(None)
                                    motif_label.append(None)                    
                            label = motif_label
                            split = row["split"]
                        else:
                            label = [None for _ in seq]
                            split = None
                else:
                    label = [None for _ in seq]
                    split = None
                self.probe.append(label)
                self.probe_split.append(split)
                count += (split is not None)
            self.seqs.append(seq)        
        max_len = len(sorted(self.seqs, key=len)[int(0.95*len(self.seqs))])
        self.max_len = max_len
        logging.info(f"MAX LEN: {max_len}")
        logging.info(f"{count}/{len(bpe.tokenizers)} structures have labels")
        logging.info(f"LABEL SET: {self.label_set}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][: self.max_len]
        pad_len = self.max_len - len(seq)
        input_ids = seq + [0] * pad_len
        attention_mask = [1] * len(seq) + [0] * pad_len
        labels = input_ids[1:] + [0]
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "idx": idx
        }
        if self.do_probe:
            if self.task == "remote-homology-detection":
                item["probe_labels"] = self.probe[idx]
            else:
                probe_seq = self.probe[idx][: self.max_len]
                pad_probe = probe_seq + [None] * pad_len
                probe_labels = [val if (val is not None) else 0 for val in pad_probe]
                probe_mask = [(val is not None) for val in pad_probe]
                item["probe_labels"] = torch.tensor(probe_labels, dtype=torch.long)
                item["probe_mask"] = torch.tensor(probe_mask, dtype=torch.bool)
        return item


def split_dataset(ds, seed=0):
    n = len(ds)
    idxes = list(range(n))
    random.seed(seed)
    random.shuffle(idxes)
    train_idxes, valid_idxes, test_idxes = idxes[:int(0.8*n)], idxes[int(0.8*n): int(0.9*n)], idxes[int(0.9*n):]
    return Subset(ds, train_idxes), Subset(ds, valid_idxes), Subset(ds, test_idxes)


def split_probe_dataset(ds):
    train_idx = [i for i in range(len(ds)) if ds.probe_split[i] == "train"]
    valid_idx = [i for i in range(len(ds)) if ds.probe_split[i] == "valid"]
    test_idx = [i for i in range(len(ds)) if (ds.probe_split[i] and ds.probe_split[i] not in ["train", "valid"])]
    return Subset(ds, train_idx), Subset(ds, valid_idx), Subset(ds, test_idx)


def make_collate_fn(global_max_len: int):
    """
    Returns a `collate_fn` for DataLoader that:
      1) Computes the true lengths from attention_mask in the batch
      2) Truncates each tensor to the batch_max length (<= global_max_len)
      3) Stacks into batched tensors
    """
    def collate_fn(batch):
        # batch is a list of dicts, each with keys:
        # "input_ids", "attention_mask", "labels" (and optionally "probe_labels")
        # all are 1D LongTensors of length `global_max_len`.
        
        # 1) find true sequence lengths via attention_mask
        seq_lens = [int(item["attention_mask"].sum().item()) for item in batch]
        batch_max = min(max(seq_lens), global_max_len)
        
        # 2) truncate & stack
        input_ids      = torch.stack([item["input_ids"][:batch_max]      for item in batch], dim=0)
        attention_mask = torch.stack([item["attention_mask"][:batch_max] for item in batch], dim=0)
        labels         = torch.stack([item["labels"][:batch_max]         for item in batch], dim=0)
        idxes = [item["idx"] for item in batch]
        out = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "idxes": idxes
        }
        
        # 3) optionally carry through probe_labels
        if "probe_labels" in batch[0]:
            if isinstance(batch[0]["probe_labels"], Iterable):
                probe_labels = torch.stack([item["probe_labels"][:batch_max] for item in batch], dim=0)
                out["probe_labels"] = probe_labels
                probe_mask = torch.stack([item["probe_mask"][:batch_max] for item in batch], dim=0)
                out["probe_mask"] = probe_mask  
            else:
                out["probe_labels"] = torch.tensor([item["probe_labels"] for item in batch])
        return out

    return collate_fn

# -----------------------------
# 2) MODEL
# -----------------------------
class ProteinLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(
                x,
                src_mask = causal_mask,
                src_key_padding_mask = ~attention_mask.bool()
            )
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, x

# -----------------------------
# 3) PROBE CLASSIFIER
# -----------------------------
class ProbeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = None,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        A simple MLP probe.

        Args:
          input_dim:    Size of the LM hidden states (d_model).
          num_classes:  Number of output classes.
          hidden_dims:  List of hidden‐layer sizes. If None, defaults to [input_dim//2].
          activation:   One of {"relu","gelu","tanh","leaky_relu"}.
          dropout:      Dropout probability between layers.
        """
        super().__init__()

        # default hidden size
        if hidden_dims is None:
            hidden_dims = [max(input_dim // 2, 1)]

        # Map string → activation class
        acts = {
            "relu":      nn.ReLU,
            "gelu":      nn.GELU,
            "tanh":      nn.Tanh,
            "leaky_relu":nn.LeakyReLU,
        }
        Act = acts.get(activation.lower(), nn.ReLU)

        # Build layer sizes: [in → h1 → h2 → … → out]
        dims = [input_dim] + hidden_dims + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # for all but final layer, add activation+dropout
            if i < len(dims) - 2:
                layers.append(Act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (..., input_dim)  e.g. shape (N, d_model)
        returns logits (..., num_classes)
        """
        return self.net(h)

# -----------------------------
# 4) EVAL HELPERS
# -----------------------------
def evaluate_next_token(model, dataloader, device, criterion):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval next token"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            nt = mask.sum().item()
            total_loss += loss.item() * nt
            total_tokens += nt
    return total_loss / total_tokens

def evaluate_probe(model,
                   train_loader,
                   val_loader,
                   device,
                   args,
                   num_labels,
                   per_residue=False):
    model.eval()

    # 1) Gather training features & labels (same as before)...
    if per_residue:
        feats, labs, masks = [], [], []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="gather train features"):
                ids        = batch["input_ids"].to(device)
                attn_mask  = batch["attention_mask"].to(device)
                p_labels   = batch["probe_labels"].to(device)
                p_mask     = batch["probe_mask"].to(device)

                _, hidden = model(ids, attn_mask)      # (B, L, D)
                B, L, D    = hidden.size()

                feats.append(hidden.reshape(B*L, D))
                labs.append(p_labels.reshape(B*L))
                masks.append(p_mask.reshape(B*L).bool())

        feat_train = torch.cat(feats, dim=0)
        lab_train  = torch.cat(labs,  dim=0)
        mask_train = torch.cat(masks, dim=0)

        # only keep positions with labels
        feat_train = feat_train[mask_train]
        lab_train  = lab_train[mask_train]

    else:
        feats, labs = [], []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="gather train features"):
                ids        = batch["input_ids"].to(device)
                attn_mask  = batch["attention_mask"].to(device)
                p_labels   = batch["probe_labels"].to(device)  # (B,)

                _, hidden = model(ids, attn_mask)             # (B, L, D)
                masked    = hidden * attn_mask.unsqueeze(-1)  # zero out padding
                lengths   = attn_mask.sum(dim=1, keepdim=True)  # (B,1)
                avg_h     = masked.sum(dim=1) / lengths         # (B, D)

                feats.append(avg_h)
                labs.append(p_labels)

        feat_train = torch.cat(feats, dim=0)  # (N, D)
        lab_train  = torch.cat(labs,  dim=0)  # (N,)

    # 2) Train linear probe
    probe = ProbeClassifier(args.d_model, 
                            num_labels, 
                            hidden_dims=args.hidden_dims, 
                            dropout=args.dropout).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    crit  = nn.CrossEntropyLoss()

    probe.train()
    for _ in tqdm(range(args.probe_epochs), desc="training probe"):
        opt.zero_grad()
        logits_p = probe(feat_train)
        loss_p   = crit(logits_p, lab_train)
        loss_p.backward()
        opt.step()

    # 3) Evaluate on validation split
    probe.eval()
    # buffers for metrics
    if per_residue:
        all_probs, all_labels = [], []
    else:
        all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="probe inference"):
            ids        = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            p_labels   = batch["probe_labels"].to(device)

            _, hidden = model(ids, attn_mask)
            if per_residue:
                p_mask = batch["probe_mask"].to(device)
                B, L, D = hidden.size()
                feats_v = hidden.reshape(B*L, D)
                labs_v  = p_labels.reshape(B*L)
                mask_v  = p_mask.reshape(B*L).bool()

                logits_v = probe(feats_v[mask_v])             # (n_labeled, C)
                probs_v  = F.softmax(logits_v, dim=-1).cpu().numpy()
                labs_np  = labs_v[mask_v].cpu().numpy()

                all_probs.append(probs_v)
                all_labels.append(labs_np)

            else:
                # protein-level
                masked   = hidden * attn_mask.unsqueeze(-1)
                lengths  = attn_mask.sum(dim=1, keepdim=True)
                avg_h    = masked.sum(dim=1) / lengths  # (B, D)
                logits_p = probe(avg_h)                              # (B, C)
                preds    = logits_p.argmax(dim=-1).cpu().numpy()
                labs_np  = p_labels.cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labs_np)

    metrics = {}
    if per_residue:
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)  # shape (N, C)
        y_pred = y_prob.argmax(axis=1)

        metrics["accuracy"] = (y_pred == y_true).mean()
        # multi-class AUROC (one-vs-rest, macro average)
        metrics["auroc"]     = roc_auc_score(
            y_true, y_prob if num_labels > 2 else y_prob[:, 1], 
            multi_class="ovo", 
            average="macro"
        )
    else:
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        metrics["accuracy"]  = (y_pred == y_true).mean()
        metrics["macro_f1"]  = f1_score(y_true, y_pred, average="macro")

    return metrics

def sample_unconditional(
    model,
    device,
    bpe,
    max_len: int,
    length_prior: list[int],
    start_prior: list[int],
    num_samples: int = 1,
    temperature: float = 1.0    
):
    """
    Unconditional sampling of `num_samples` sequences of total length K
    (where K is drawn from length_prior and satisfies K%4==1), *without* any
    extra BOS token.  The first token is sampled from start_prior.
    """
    import logging
    assert bpe.res_init, "BPE must be initialized"

    logging.info(f"Starting sample_unconditional with num_samples={num_samples}, device={device}")

    # 1) filter legal lengths
    legal_lengths = [K for K in length_prior if (K % 4 == 1 and K <= max_len)]
    logging.info(f"Legal lengths: {legal_lengths}")
    assert legal_lengths, "No K in length_prior satisfies K%4==1"

    # 2) precompute your 4 ranges
    n         = len(bpe._tokens)
    omega_off = bpe.cum_bin_count('omega')
    cac1n_off = bpe.cum_bin_count('C:1N:1CA')
    phi_off   = bpe.cum_bin_count('phi')
    logging.info(f"n={n} omega_off={omega_off} cac1n_off={cac1n_off} phi_off={phi_off}")
    ranges = {
        0: (0,   n),
        1: (n + omega_off,
            n + omega_off + len(bpe._bin_counts[1]['omega'])),
        2: (n + phi_off,
            n + phi_off + len(bpe._bin_counts[1]['phi'])),
        3: (n + cac1n_off,
            n + cac1n_off + len(bpe._bin_counts[1]['CA:C:1N'])),
    }
    logging.info(f"Ranges: {ranges}")
    term_motifs = np.array([i < len(bpe._tokens) \
                   and Tokenizer.num_bonds(list(bpe._tokens.values())[i]) == 2 \
                    for i in range(bpe.vocab_size)])
    logging.info(f"term_motifs shape: {term_motifs.shape}, bpe.vocab_size: {bpe.vocab_size}")
    vocab_size = model.token_emb.num_embeddings
    logging.info(f"Model embedding vocab_size: {vocab_size}")
    model.eval()
    samples = []    
    attempts = 0
    max_attempts_per_sample = 100  # To avoid infinite loops on totally broken setups

    with torch.no_grad():
        for sample_idx in range(num_samples):
            inner_attempt = 0
            while True:
                inner_attempt += 1
                attempts += 1
                if inner_attempt > max_attempts_per_sample:
                    logging.error(f"[Sample {sample_idx}] Reached {max_attempts_per_sample} failed attempts -- giving up!")
                    raise RuntimeError(f"Could not produce valid sample {sample_idx} after {max_attempts_per_sample} tries.")
                try:
                    # a) pick your length
                    K = random.choice(legal_lengths)
                    logging.info(
                        f"[Sample {sample_idx}:{inner_attempt}] Picked length K={K}"
                    )

                    # b) sample the very first token from your empirical start_prior
                    first_tok = random.choice(start_prior)
                    seq = torch.tensor([[first_tok]], dtype=torch.long, device=device)

                    # c) generate the remaining K-1 tokens, catching NaN and index errors
                    for j in range(1, K):
                        attn_mask = torch.ones_like(seq)
                        try:
                            logits, _ = model(seq, attn_mask)
                        except Exception as e:
                            logging.info(
                                f"[Sample {sample_idx}:{inner_attempt}][Step {j}] Error in model forward: {e}"
                            )
                            raise  # Will be caught by outer except

                        logits = logits[0, -1]  # last‐position logits

                        lo, hi = ranges[j % 4]
                        assert 0 <= lo <= hi <= vocab_size, f"Invalid mask bounds: lo={lo}, hi={hi}, vocab_size={vocab_size}"
                        block = torch.full((vocab_size,), float("-inf"), device=device)
                        block[lo:hi] = 0.0
                        if j < K-1:
                            block[term_motifs] = float("-inf")
                        else:
                            block[~term_motifs] = float("-inf")
                        logits = logits + block

                        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)

                        # Check for NaNs or all-0s
                        if not torch.isfinite(probs).all() or probs.sum() == 0:
                            logging.info(
                                f"[Sample {sample_idx}:{inner_attempt}][Step {j}] Probabilities invalid (NaN or all zero)."
                            )
                            raise RuntimeError("Probabilities NaN or degenerate")

                        nxt = torch.multinomial(probs, 1)  # shape (1,)
                        seq = torch.cat([seq, nxt.unsqueeze(0)], dim=1)

                    # If we got here, sample completed successfully
                    samples.append(seq.squeeze(0).tolist())
                    logging.info(
                        f"[Sample {sample_idx}:{inner_attempt}] Finished, sequence length: {len(samples[-1])} (SUCCESS)"
                    )
                    break  # Break retry loop for this sample

                except Exception as e:
                    logging.warning(f"[Sample {sample_idx}:{inner_attempt}] Sampling errored (retrying): {e}")
                    continue  # Retry this sample attempt

    # (optional) decode & visualize
    tokenizers = []
    for i, sample in enumerate(samples):
        logging.info(f"[Decode] Sample {i}, length: {len(sample)}")
        quant = bpe.dequantize(sample)
        repl  = bpe.recover(quant)
        t     = bpe.recover_structure(repl, quant)
        tokenizers.append(t)
        # try:
        #     t.visualize(f"{i}.png")
        # except Exception as e:
        #     logging.warning(f"[Decode] Visualization failed for sample {i}: {e}")

    return tokenizers

# -----------------------------
# 5) MAIN TRAIN/EVAL LOOP
# -----------------------------
def main(args):
    if args.debug:
        # override for minimal debug run
        args.d_model    = 64
        args.num_layers = 1
        args.num_heads  = 2
        args.d_ff       = 256
        args.max_len    = 32
        args.batch_size = 4
        args.epochs     = 1
        args.eval_interval = 1

    # 1) W&B init (disable if debug)
    wandb.init(
        entity=args.wandb_team,
        project=args.wandb_project,
        config=vars(args),
        mode="disabled" if args.debug else None
    )

    # 2) Load & split
    save_dir = Path(args.checkpoint_path).parent
    bpe = pickle.load(open(args.checkpoint_path, 'rb'))
    full_ds = ProteinDataset(
        bpe,
        args.labels_path,
        task=args.task
    )
    # derived metrics: vocab_size / max_len
    vocab_size = full_ds.vocab_size
    max_len = full_ds.max_len
    num_labels = len(full_ds.label_set)

    # 3) Debug mode: take only first 10
    # next-token splits: train / val / test_ntp
    train_ds, val_ds, test_ntp_ds = split_dataset(full_ds, args.seed)
    # probing splits: probe_train / probe_val / probe_test
    if args.labels_path:
        probe_train_ds, probe_val_ds, probe_test_ds = split_probe_dataset(full_ds)        

    if args.debug:
        idx10 = list(range(10))
        train_ds       = Subset(train_ds, idx10)
        val_ds         = Subset(val_ds,   idx10)
        test_ntp_ds    = Subset(test_ntp_ds, idx10)
        if args.labels_path:
            probe_train_ds = Subset(probe_train_ds, idx10)
            probe_val_ds   = Subset(probe_val_ds,   idx10)
            probe_test_ds  = Subset(probe_test_ds,  idx10)    

    # 4) DataLoaders
    collate_fn         = make_collate_fn(max_len)
    train_loader       = DataLoader(train_ds,       batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader         = DataLoader(val_ds,         batch_size=args.batch_size, collate_fn=collate_fn)
    test_ntp_loader    = DataLoader(test_ntp_ds,    batch_size=args.batch_size, collate_fn=collate_fn)

    if args.labels_path:
        probe_train_loader = DataLoader(probe_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        probe_val_loader   = DataLoader(probe_val_ds,   batch_size=args.batch_size, collate_fn=collate_fn)
        probe_test_loader  = DataLoader(probe_test_ds,  batch_size=args.batch_size, collate_fn=collate_fn)

    # 5) Model / optimizer / loss
    device = torch.device("cuda" if (not args.debug and torch.cuda.is_available()) else "cpu")
    model = ProteinLM(
        vocab_size, args.d_model, args.num_layers,
        args.num_heads, args.d_ff, max_len
    ).to(device)

    # Inference mode?  Load weights and skip training
    if args.inference:
        assert args.model_path, "--model_path required in inference mode"
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"loaded model from {args.model_path}")
        length_prior = [len(seq) for seq in full_ds.seqs]
        start_prior = [seq[0] for seq in full_ds.seqs]  
        model.eval()
        crit = nn.CrossEntropyLoss(ignore_index=0)       

        # a) Sample unconditional
        tokenizers = sample_unconditional(
            model=model,
            device=device,
            bpe=bpe,
            length_prior=length_prior,
            start_prior=start_prior,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_len=max_len
        )        
        train_pdb_files = [full_ds.fnames[item["idx"]] for item in train_ds]
        full_coords_pfunc = functools.partial(extract_backbone_coords, atoms=["N", "CA", "C"])
        pool = mp.Pool(processes=mp.cpu_count())
        train_coords = pool.map(full_coords_pfunc, tqdm(train_pdb_files, desc="extract coords train"))
        sampled_dfs = [t._angles_and_dists[["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]] for t in tokenizers]
        outdir = Path(args.model_path).parent
        gen_pdb_files = write_preds_pdb_folder(sampled_dfs, outdir / "sampled_pdb")
        generated_coords = pool.map(full_coords_pfunc, tqdm(gen_pdb_files, desc="extract coords gen"))
        metrics = compute_metrics(gen_pdb_files, train_pdb_files, generated_coords, train_coords)
        wandb.log(metrics)

        # b) Next-token on **test** split
        ntp_loss = evaluate_next_token(model, test_ntp_loader, device, crit)
        ntp_ppl  = torch.exp(torch.tensor(ntp_loss))
        wandb.log({"test/ntp_loss": ntp_loss, "test/ntp_ppl": ntp_ppl})         
                
        # c) Zero-shot probe on **probe_test** (trained on probe_train)
        if args.labels_path:
            metrics = evaluate_probe(
                model,
                probe_train_loader,
                probe_test_loader,
                device,
                args,
                num_labels = num_labels,
                per_residue=(args.task != "remote-homology-detection")
            )
            wandb.log({f"test/probe_{key}": metrics[key] for key in metrics})
        return

    # 6) Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    step = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            ids  = batch["input_ids"].to(device)
            m    = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            logits, _ = model(ids, m)
            loss = crit(logits.view(-1, vocab_size), lbls.view(-1))
            loss.backward()
            optimizer.step()

            step += 1
            wandb.log({"train/loss": loss.item(), "step": step})

            if step % args.eval_interval == 0:
                # a) next-token on **val**
                val_loss = evaluate_next_token(model, val_loader, device, crit)
                val_ppl  = torch.exp(torch.tensor(val_loss))
                wandb.log({
                    "val/ntp_loss": val_loss,
                    "val/ntp_ppl":  val_ppl,
                    "step": step
                })

                # b) zero-shot probe on **probe_val**
                if args.labels_path:
                    metrics = evaluate_probe(
                        model,
                        probe_train_loader,
                        probe_val_loader,
                        device,
                        args,
                        num_labels = num_labels,
                        per_residue=(args.task != "remote-homology-detection")
                    )
                    wandb.log({f"val/probe_{key}": metrics[key] for key in metrics} | {"step": step})
                model.train()

        # end‐of‐epoch checkpoint
        ckpt = f"{save_dir}/ckpt_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        wandb.save(ckpt)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()    
    if args.debug:
        breakpoint()
    main(args)
