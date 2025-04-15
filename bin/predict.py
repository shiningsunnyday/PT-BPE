import pyrosetta
from pyrosetta import pose_from_pdb
import nglview as nv
from ipywidgets import HBox
from tqdm import tqdm
import tempfile
import nglview as nv
import imageio
import os
from foldingdiff.datasets import FullCathCanonicalCoordsDataset, extract_pdb_code_and_chain
from foldingdiff.bpe import *
from foldingdiff.plotting import *
import scipy.io
import numpy as np
import subprocess
import argparse
import pickle
from datetime import datetime
import sys
from tape.datasets import LMDBDataset
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
# os.environ['CUDA_AVAILABLE_DEVICES'] = "" # debug

class MyDataset(Dataset):
    def __init__(self, tokenizers, dataset, label_map):
        mapping = {}
        for i, t in enumerate(tokenizers):
            stem = Path(t.fname).stem
            mapping[stem] = i
        my_data = []
        for sample in dataset:
            prot, chain = extract_pdb_code_and_chain(sample['id'])
            key = f"{prot}_{chain}"
            if key in mapping:
                i = mapping[key]
                if sample['fold_label'] not in label_map:
                    continue
                sample['fold_label'] = label_map[sample['fold_label']]
                my_data.append((prot, chain, tokenizers[i], sample))
        self.data = my_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        id, chain, t, sample = self.data[idx]
        protein = ESMProtein.from_protein_chain(ProteinChain.from_rcsb(id,chain_id=chain))
        item = sample
        # item['protein'] = protein
        # item['coords'] = t.compute_coords()
        # item['tree'] = t.bond_to_token.tree
        protein_tensor = client.encode(protein)
        output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
        )        
        item['embeddings'] = output.embeddings[0].to(torch.float32)
        return item



def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_{timestamp}.log")
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().info("Logger initialized.")

def get_logger():
    """Helper to retrieve the global logger."""
    return LOGGER

def parse_args():
    parser = argparse.ArgumentParser(description="FoldingDiff BPE Script")
    parser.add_argument("--pkl-file", type=str, required=True, 
                        help="Load the BPE results.")    
    parser.add_argument("--auto", action='store_true', help='auto set folders')
    parser.add_argument("--save-dir", type=str, default="plots/models", 
                        help="Directory to save ckpts.")
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory where log files will be saved.")
    parser.add_argument("--cuda", default="cpu")
    # hparams
    parser.add_argument("--p_min_size", default=float("inf"), help="when to start using rmsd binning")
    return parser.parse_args()



# Define the probing layer as a two-layer MLP.
# Now, it expects as input a fixed-size vector (already pooled).
class ProbingLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(ProbingLayer, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x is expected to have shape (embed_dim,)
        # We do NOT pool inside the layer now because we pre-pooled externally.
        hidden = self.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return logits
    

def train_probe(args, train_dataset, valid_dataset):
    # Set hyperparameters (adjust as needed)
    embed_dim = 960       # Example value; replace with the actual embedding dimension
    hidden_dim = 128      # Example hidden dimension
    num_classes = 45      # For fold_label classification

    # Create the probing layer and move it to device.
    device = torch.device(args.cuda)
    model = ProbingLayer(embed_dim, hidden_dim, num_classes).to(device)

    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoaders. Here we use batch size of 64 (adjust collate_fn to handle variable-length inputs if needed).
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    num_epochs = 10
    best_val_macro_f1 = -1.0  # initialize best validation macro F1

    for epoch in range(num_epochs):
        # ----------------------
        # Training Loop
        # ----------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        num_train_samples = 0

        for batch in tqdm(train_loader, desc="epoch"):
            optimizer.zero_grad()
            embeddings = batch['embeddings']  # shape: (B, seq_length, embed_dim)
            # Select the tokens (ignoring the first and last tokens). Here, we assume that all samples can be sliced identically.
            # If needed, adjust this for variable-length sequences.
            # For example, pooling could be applied per sample.            
            repr_tensor = embeddings[:, 1:-1, :]  # shape: (B, n_residues, embed_dim)

            # Here, we apply mean pooling along the residue dimension for each sample.
            pooled_repr = repr_tensor.mean(dim=1)  # shape: (B, embed_dim)
            
            pooled_repr = pooled_repr.to(device)
            labels = batch['fold_label'].to(device)  # shape: (B,)
            
            # Forward pass: we pass each fixed-size pooled representation into the model.
            logits = model(pooled_repr[0])  # dummy forward pass to check shape
            # Instead, apply the model per example in the batch.
            # One simple way is to iterate over the batch:
            batch_logits = []
            for i in range(pooled_repr.size(0)):
                logit = model(pooled_repr[i])  # shape: (num_classes,)
                batch_logits.append(logit)
            batch_logits = torch.stack(batch_logits, dim=0)  # shape: (B, num_classes)
            
            loss = criterion(batch_logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * pooled_repr.size(0)
            preds = batch_logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            num_train_samples += pooled_repr.size(0)
        
        avg_train_loss = train_loss / num_train_samples
        train_accuracy = train_correct / num_train_samples

        # ----------------------
        # Validation Loop
        # ----------------------
        model.eval()
        valid_loss = 0.0
        valid_preds = []
        valid_labels = []
        num_valid_samples = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="validation"):
                # For validation, the sample key is assumed to be "output" containing embeddings.
                embeddings = batch['output'].embeddings  # shape: (B, seq_length, embed_dim)
                repr_tensor = embeddings[:, 1:-1, :]  # shape: (B, n_residues, embed_dim)
                pooled_repr = repr_tensor.mean(dim=1)  # shape: (B, embed_dim)
                pooled_repr = pooled_repr.to(device)
                labels = batch['fold_label'].to(device)  # shape: (B,)
                
                # Forward pass for the batch: process each sample.
                batch_logits = []
                for i in range(pooled_repr.size(0)):
                    logit = model(pooled_repr[i])
                    batch_logits.append(logit)
                batch_logits = torch.stack(batch_logits, dim=0)  # shape: (B, num_classes)
                
                loss = criterion(batch_logits, labels)
                valid_loss += loss.item() * pooled_repr.size(0)
                preds = batch_logits.argmax(dim=1)
                valid_preds.extend(preds.cpu().tolist())
                valid_labels.extend(labels.cpu().tolist())
                num_valid_samples += pooled_repr.size(0)
        
        avg_valid_loss = valid_loss / num_valid_samples
        valid_accuracy = sum([1 for p, t in zip(valid_preds, valid_labels) if p == t]) / num_valid_samples
        
        # Compute macro F1 score on validation set.
        val_macro_f1 = f1_score(valid_labels, valid_preds, average='macro')

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy*100:.2f}%, Valid Loss: {avg_valid_loss:.4f}, "
              f"Valid Acc: {valid_accuracy*100:.2f}%, Valid Macro F1: {val_macro_f1:.4f}")

        # ----------------------
        # Checkpoint saving
        # ----------------------
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            os.makedirs(args.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.save_dir, f"best_checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_val_macro_f1,
            }, checkpoint_path)
            print(f"New best macro F1 achieved: {best_val_macro_f1:.4f}. Saved checkpoint to {checkpoint_path}.")

def main(args):    
    if args.auto:
        cur_time = time.time()
        setattr(args, 'plot_dir', f'./plots/learn/{cur_time}')
        setattr(args, 'save_dir', f'./ckpts/{cur_time}')
        os.makedirs(args.plot_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)    
    setup_logger(args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info("Script started.")      
    bpe = pickle.load(open(args.pkl_file, 'rb'))       
    train = LMDBDataset('data/remote_homology_raw/remote_homology_train.lmdb')
    counts = Counter([x['fold_label'] for x in train])
    class_labels = [c for c in counts if counts[c] > 50]
    label_map = dict(zip(class_labels, range(len(class_labels))))
    valid = LMDBDataset('data/remote_homology_raw/remote_homology_valid.lmdb')
    test_family_holdout = LMDBDataset('data/remote_homology_raw/remote_homology_test_family_holdout.lmdb')
    test_fold_holdout = LMDBDataset('data/remote_homology_raw/remote_homology_test_fold_holdout.lmdb')
    test_superfamily_holdout = LMDBDataset('data/remote_homology_raw/remote_homology_test_superfamily_holdout.lmdb')
    train_dataset = MyDataset(bpe.tokenizers, train, label_map)
    valid_dataset = MyDataset(bpe.tokenizers, valid, label_map)
    test_family_dataset = MyDataset(bpe.tokenizers, test_family_holdout, label_map)
    test_fold_dataset = MyDataset(bpe.tokenizers, test_fold_holdout, label_map)
    test_superfamily_dataset = MyDataset(bpe.tokenizers, test_superfamily_holdout, label_map)
    # train using train_dataset, validate with valid_dataset, ignore test_* datasets for now
    # below obtains input representation for first sample
    # sample = train_dataset[0]
    # output = sample['output']
    # repr = output.embeddings[0,1:-1]
    # y = sample['fold_label']
    # probing layer takes as input repr: (n_residues, embed_dim)
    # outputs logits over 45 classes possible for y
    train_probe(args, train_dataset, valid_dataset)
    

if __name__ == "__main__":    
    args = parse_args()
    client = ESMC.from_pretrained("esmc_300m").to(args.cuda) # or "cpu"    
    main(args)
