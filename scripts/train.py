import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

# -----------------------------
# 1) DATASET
# -----------------------------
class ProteinDataset(Dataset):
    def __init__(self, token_sequences, max_len):
        """
        token_sequences: List[List[int]]  # already tokenized sequences
        max_len: int                      # pad/truncate length
        """
        self.seqs = token_sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][: self.max_len]
        pad_len = self.max_len - len(seq)
        input_ids = seq + [0] * pad_len
        attention_mask = [1] * len(seq) + [0] * pad_len
        labels = input_ids[1:] + [0]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# -----------------------------
# 2) MODEL
# -----------------------------
class ProteinLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
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
        # tie weights
        self.head.weight = self.token_emb.weight

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        # causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, memory=None, tgt_mask=mask, tgt_key_padding_mask=~attention_mask.bool())
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, x  # return hidden states x for probing

# -----------------------------
# 3) PROBE
# -----------------------------
class ProbeClassifier(nn.Module):
    def __init__(self, d_model, num_labels):
        super().__init__()
        self.linear = nn.Linear(d_model, num_labels)

    def forward(self, h):
        # h: (batch*seq_len, d_model)
        return self.linear(h)

# -----------------------------
# 4) EVALUATION HELPERS
# -----------------------------
def evaluate_next_token(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / total_tokens

def evaluate_probe(model, probe_train_loader, probe_val_loader, device, layer_idx, d_model, num_labels, probe_epochs, probe_lr):
    # extract features
    model.eval()
    features = []
    labels   = []
    with torch.no_grad():
        for batch in probe_train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            probe_labels = batch["probe_labels"].to(device)  # shape: (batch, seq_len)
            _, hidden = model(input_ids, mask)
            h = hidden[:, :, :]  # shape: (B, L, d_model)
            B, L, _ = h.shape
            features.append(h.reshape(B*L, d_model))
            labels.append(probe_labels.reshape(B*L))
    feat_train = torch.cat(features, dim=0)
    lab_train  = torch.cat(labels, dim=0)

    # train linear probe
    probe = ProbeClassifier(d_model, num_labels).to(device)
    opt_probe = torch.optim.Adam(probe.parameters(), lr=probe_lr)
    crit_probe = nn.CrossEntropyLoss()
    probe.train()
    for _ in range(probe_epochs):
        opt_probe.zero_grad()
        logits = probe(feat_train)
        loss = crit_probe(logits, lab_train)
        loss.backward()
        opt_probe.step()

    # eval on val split
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in probe_val_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            probe_labels = batch["probe_labels"].to(device)
            _, hidden = model(input_ids, mask)
            h = hidden[:, :, :]
            B, L, _ = h.shape
            feats = h.reshape(B*L, d_model)
            labs  = probe_labels.reshape(B*L)
            preds = probe(feats).argmax(dim=-1)
            correct += (preds == labs).sum().item()
            total += labs.numel()
    return correct / total

# -----------------------------
# 5) MAIN TRAIN/EVAL LOOP
# -----------------------------
def main():
    # --- hyperparams & init ---
    config = {
      "vocab_size": 10_000,
      "d_model": 256,
      "num_layers": 8,
      "num_heads": 8,
      "d_ff": 1024,
      "max_len": 512,
      "batch_size": 32,
      "lr": 1e-4,
      "epochs": 10,
      "eval_interval": 500,
      "probe_layer": 4,
      "probe_epochs": 5,
      "probe_lr": 1e-3,
    }
    wandb.init(project="protein-motif-lm", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data loaders (you supply tokenized lists) ---
    train_ds = ProteinDataset(train_seqs, config["max_len"])
    test_ds  = ProteinDataset(test_seqs,  config["max_len"])
    probe_train_ds = ProteinDataset(probe_train_seqs, config["max_len"])
    probe_val_ds   = ProteinDataset(probe_val_seqs,   config["max_len"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"])
    probe_train_loader = DataLoader(probe_train_ds, batch_size=config["batch_size"])
    probe_val_loader   = DataLoader(probe_val_ds,   batch_size=config["batch_size"])

    # --- model, optimizer, scheduler, loss ---
    model = ProteinLM(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*config["epochs"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    step = 0
    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, mask)
            loss = criterion(logits.view(-1, config["vocab_size"]), labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0], "step": step})

            if step % config["eval_interval"] == 0:
                # next-token eval
                ntp_loss = evaluate_next_token(model, test_loader, device, criterion)
                ntp_ppl  = torch.exp(torch.tensor(ntp_loss))
                wandb.log({"eval/ntp_loss": ntp_loss, "eval/ntp_ppl": ntp_ppl, "step": step})

                # zero-shot probe
                probe_acc = evaluate_probe(
                    model,
                    probe_train_loader,
                    probe_val_loader,
                    device,
                    config["probe_layer"],
                    config["d_model"],
                    num_labels=probe_num_labels,
                    probe_epochs=config["probe_epochs"],
                    probe_lr=config["probe_lr"],
                )
                wandb.log({"eval/probe_acc": probe_acc, "step": step})

                model.train()

        # checkpoint end of epoch
        ckpt = f"checkpoint_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        wandb.save(ckpt)

    wandb.finish()

if __name__ == "__main__":
    main()
