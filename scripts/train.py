import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import pandas as pd
from pathlib import Path
import wandb
import random

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
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to pickle file containing tokenized sequences"
    )
    parser.add_argument(
        "--labels_path", type=str, required=True,
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
        "--model_ckpt", type=str, default=None,
        help="Path to a saved model checkpoint for inference mode"
    )    

    # model & training hyperparameters
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--vocab_size",   type=int,   default=10_000)
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--num_layers",   type=int,   default=8)
    parser.add_argument("--num_heads",    type=int,   default=8)
    parser.add_argument("--d_ff",         type=int,   default=1024)
    parser.add_argument("--max_len",      type=int,   default=512)
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
    def __init__(self, ckpt_path, labels_path, max_len=512):
        """
        lables_path: csv for labels to structures in ckpt_path
        """
        bpe = pickle.load(open(ckpt_path, 'rb'))        
        self.vocab_size = bpe.vocab_size
        if labels_path:
            df = pd.read_csv(labels_path)              
        else:
            df = None
        self.seqs = []
        self.probe = []
        self.probe_split = []
        for t in bpe.tokenizers:
            tokenized = t.tokenize()
            seq = bpe.quantize(tokenized)
            if df is not None:
                pdb_chain = Path(t.fname).stem.split('_')
                if len(pdb_chain) != 2:
                    raise ValueError(f"{t.fname} should be pdbid_chainid")
                pdb_id = pdb_chain[0]
                chain_id = pdb_chain[1]
                mask = (df.pdb_id == pdb_id) & (df.chain_id == chain_id)
                result = df.loc[mask]
                if len(result) >= 1:
                    row = result.iloc[0]
                    label = row["residue_label"]
                    split = row["split"]
                else:
                    label = None
                    split = None
                self.probe.append(label)
                self.probe_split.append(split)
            self.seqs.append(seq)        
        # len(sorted(self.seqs, key=len)[int(0.95*len(self.seqs))])
        self.max_len = max_len

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
        }
        if self.probe is not None:
            breakpoint()
            probe_seq = self.probe[idx][: self.max_len]
            pad_probe = probe_seq + [0] * pad_len
            item["probe_labels"] = torch.tensor(pad_probe, dtype=torch.long)
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
        
        out = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        
        # 3) optionally carry through probe_labels
        if "probe_labels" in batch[0]:
            probe_labels = torch.stack([item["probe_labels"][:batch_max] for item in batch], dim=0)
            out["probe_labels"] = probe_labels
        
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
    def __init__(self, d_model, num_labels):
        super().__init__()
        self.linear = nn.Linear(d_model, num_labels)

    def forward(self, h):
        return self.linear(h)

# -----------------------------
# 4) EVAL HELPERS
# -----------------------------
def evaluate_next_token(model, dataloader, device, criterion):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            nt = mask.sum().item()
            total_loss += loss.item() * nt
            total_tokens += nt
    return total_loss / total_tokens

def evaluate_probe(model, train_loader, val_loader, device, d_model, num_labels, probe_epochs, probe_lr):
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            probe_labels = batch["probe_labels"].to(device)
            _, hidden = model(input_ids, mask)
            B, L, D = hidden.size()
            feats.append(hidden.reshape(B*L, D))
            labs.append(probe_labels.reshape(B*L))
    feat_train = torch.cat(feats, dim=0)
    lab_train  = torch.cat(labs, dim=0)

    probe = ProbeClassifier(d_model, num_labels).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=probe_lr)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    probe.train()
    for _ in range(probe_epochs):
        opt.zero_grad()
        logits_p = probe(feat_train)
        loss_p = crit(logits_p, lab_train)
        loss_p.backward()
        opt.step()

    correct, total = 0, 0
    probe.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            probe_labels = batch["probe_labels"].to(device)
            _, hidden = model(input_ids, mask)
            B, L, D = hidden.size()
            feats_v = hidden.reshape(B*L, D)
            labs_v  = probe_labels.reshape(B*L)
            preds = probe(feats_v).argmax(dim=-1)
            correct += (preds == labs_v).sum().item()
            total += labs_v.numel()
    return correct / total

# -----------------------------
# 5) MAIN TRAIN/EVAL LOOP
# -----------------------------
def main(args):
    # 1) W&B init (disable if debug)
    wandb.init(
        entity=args.wandb_team,
        project=args.wandb_project,
        config=vars(args),
        mode="disabled" if args.debug else None
    )

    # 2) Load & split
    full_ds = ProteinDataset(
        args.checkpoint_path,
        args.labels_path,
        max_len=args.max_len
    )

    # 3) Debug mode: take only first 10
    # next-token splits: train / val / test_ntp
    train_ds, val_ds, test_ntp_ds = split_dataset(full_ds, args.seed)
    # probing splits: probe_train / probe_val / probe_test
    probe_train_ds, probe_val_ds, probe_test_ds = split_probe_dataset(full_ds)        
    if args.debug:
        idx10 = list(range(10))
        train_ds       = Subset(train_ds, idx10)
        val_ds         = Subset(val_ds,   idx10)
        test_ntp_ds    = Subset(test_ntp_ds, idx10)
        probe_train_ds = Subset(probe_train_ds, idx10)
        probe_val_ds   = Subset(probe_val_ds,   idx10)
        probe_test_ds  = Subset(probe_test_ds,  idx10)    

    # 4) DataLoaders
    collate_fn         = make_collate_fn(args.max_len)
    train_loader       = DataLoader(train_ds,       batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader         = DataLoader(val_ds,         batch_size=args.batch_size, collate_fn=collate_fn)
    test_ntp_loader    = DataLoader(test_ntp_ds,    batch_size=args.batch_size, collate_fn=collate_fn)
    probe_train_loader = DataLoader(probe_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    probe_val_loader   = DataLoader(probe_val_ds,   batch_size=args.batch_size, collate_fn=collate_fn)
    probe_test_loader  = DataLoader(probe_test_ds,  batch_size=args.batch_size, collate_fn=collate_fn)

    # 5) Model / optimizer / loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinLM(
        args.vocab_size, args.d_model, args.num_layers,
        args.num_heads, args.d_ff, args.max_len
    ).to(device)

    # Inference mode?  Load weights and skip training
    if args.inference:
        assert args.model_path, "--model_path required in inference mode"
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        crit = nn.CrossEntropyLoss(ignore_index=0)

        # a) Next-token on **test** split
        ntp_loss = evaluate_next_token(model, test_ntp_loader, device, crit)
        ntp_ppl  = torch.exp(torch.tensor(ntp_loss))
        wandb.log({"test/ntp_loss": ntp_loss, "test/ntp_ppl": ntp_ppl})

        # b) Zero-shot probe on **probe_test** (trained on probe_train)
        probe_acc = evaluate_probe(
            model,
            probe_train_loader,
            probe_test_loader,
            device,
            args.d_model,
            num_labels = full_ds.probe['label'].nunique(),
            probe_epochs=args.probe_epochs,
            probe_lr=args.probe_lr
        )
        wandb.log({"test/probe_acc": probe_acc})
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
            loss = crit(logits.view(-1, args.vocab_size), lbls.view(-1))
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
                probe_acc = evaluate_probe(
                    model,
                    probe_train_loader,
                    probe_val_loader,
                    device,
                    args.d_model,
                    num_labels = full_ds.probe['label'].nunique(),
                    probe_epochs=args.probe_epochs,
                    probe_lr=args.probe_lr
                )
                wandb.log({"val/probe_acc": probe_acc, "step": step})
                model.train()

        # end‐of‐epoch checkpoint
        ckpt = f"ckpt_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        wandb.save(ckpt)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()    
    main(args)
