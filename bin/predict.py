from tqdm import tqdm
import tempfile
import imageio
import os
from foldingdiff.datasets import FullCathCanonicalCoordsDataset, extract_pdb_code_and_chain
from foldingdiff.bpe import *
from foldingdiff.bpe_dataset import *
from foldingdiff.plotting import *
from foldingdiff.utils import str2bool
import scipy.io
import numpy as np
import subprocess
import argparse
import pickle
import json
from datetime import datetime
import sys
import lmdb
from glob import glob
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain
from torch.utils.data import Dataset, default_collate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torchmetrics.regression import R2Score, PearsonCorrCoef, SpearmanCorrCoef
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_AVAILABLE_DEVICES'] = "" # debug

# Create a global lock for GPU operations.
gpu_lock = threading.Lock()

class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pickle.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pickle.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
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
    # task
    parser.add_argument("--debug", action='store_true', help='debug or not')
    parser.add_argument("--test", action='store_true', help='test (instead of train)')
    parser.add_argument("--task", choices=["remote-homology-detection", # per-protein
                                            "structural-flexibility-prediction", # per-residue regression
                                            "epitope-prediction",
                                            "BindInt",
                                            "BindBio",
                                            "CatInt",
                                            "CatBio",
                                            "conserved-site-prediction",
                                            "repeat-motif-prediction",
                                            "epitope-prediction", # per-residue classification
    ], default="remote-homology-detection")
    # data
    parser.add_argument("--pkl-file", type=str, required=True, 
                        help="Load the BPE results.")    
    parser.add_argument("--dedup", action='store_true', help="whether to use the deduplicated or processed data")
    parser.add_argument("--pkl-data-file", type=str, 
                        help="Path to cache the train/valid dataset.")                            
    parser.add_argument("--save-dir", type=str, default="plots/models", 
                        help="Directory to save ckpts.")
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory where log files will be saved.")
    parser.add_argument("--auto", action='store_true', help='auto set folders')
    # training
    parser.add_argument("--cuda", default="cpu")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    # hparams
    parser.add_argument("--p_min_size", default=float("inf"), help="when to start using rmsd binning")
    parser.add_argument("--level", default="protein", help="prediction at protein or residue level", choices=["protein", "residue"])
    parser.add_argument("--regression", type=str2bool, default=False)
    args = parser.parse_args()
    if args.dedup and args.pkl_data_file and "dedup" not in args.pkl_data_file:
        parser.error("specify pkl-data-file to path of dedup data")
    return args


def calculate_regression_metric(logits, targets):
    # logits: (M, )
    # targets: (M, )
    device = logits.device
    r2score_func = R2Score().to(device)
    r2 = r2score_func(logits, targets)
    pearson_func = PearsonCorrCoef().to(device)
    pearsonr = pearson_func(logits, targets)
    spearman_func = SpearmanCorrCoef().to(device)
    spearmanr = spearman_func(logits, targets)
    return {
        "r2": r2,
        "pearsonr": pearsonr,
        "spearmanr": spearmanr
    }

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



class TreeLSTMCell(nn.Module):
    """Binary Tree‑LSTM cell (following Tai et al., 2015)."""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(2 * dim, 5 * dim)

    def forward(self, hl, cl, hr, cr):
        hl = hl.clone()
        cl = cl.clone()
        hr = hr.clone()
        cr = cr.clone()        
        # hl, cl, hr, cr: shape (1, dim)
        combined = torch.cat([hl, hr], dim=1)  # (1, 2*dim)
        out = self.W(combined)
        # Compute 5 parts and split them:
        i, fl, fr, o, g = [x.clone() for x in out.chunk(5, dim=1)]
        i = torch.sigmoid(i)
        fl = torch.sigmoid(fl)
        fr = torch.sigmoid(fr)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = fl * cl + fr * cr + i * g
        h = o * torch.tanh(c)
        return h, c



class UpDownTreeEncoder(nn.Module):
    """
    Up–Down Tree-LSTM encoder that handles a forest via a virtual super-root.
    
    Args:
      dim (int): Dimension of the input (leaf) residue embeddings.
      concat_updown (bool): If True, the final representation at each node is a
                            concatenation of its upward and downward hidden states.
    """
    def __init__(self, dim, concat_updown=True):
        super().__init__()
        self.up_cell = TreeLSTMCell(dim)
        self.down_cell = TreeLSTMCell(dim)
        self.concat = concat_updown

    def forward(self, leaf_emb, forest_edges, n_res, forest_roots):
        """
        Args:
          leaf_emb (Tensor): shape (n_res, dim). Pretrained leaf (residue) embeddings.
          forest_edges (list of tuples): All edges in the forest (in topological order),
                   where each tuple is (parent, left, right). Here, the children indices
                   refer to nodes already computed, and internal node indices are >= n_res.
          n_res (int): Number of leaves (residues).
          forest_roots (list of int): The indices of the roots in the forest.
        
        Returns:
          super_root_vec (Tensor): The protein-level representation from the super-root.
          leaves (Tensor): shape (n_res, feat_dim) containing enriched leaf representations.
                        If concat_updown is True, feat_dim = 2*dim; otherwise, feat_dim = dim.
        """
        device = leaf_emb.device
        dim = leaf_emb.size(1)
        # Determine maximum node index in the forest.
        max_node = max([p for p, _, _ in forest_edges] + list(range(n_res)))
        # Allocate upward tensors.
        H_up = torch.zeros(max_node + 1, dim, device=device)
        C_up = torch.zeros(max_node + 1, dim, device=device)
        # Initialize leaves: indices 0 .. n_res-1 get their pretrained embeddings.
        H_up[:n_res] = leaf_emb  # assume initial cell state is zero.
        # ----- Upward Pass -----
        for (p, l, r) in forest_edges:
            # Get children’s upward states.
            h_l, c_l = H_up[l].unsqueeze(0), C_up[l].unsqueeze(0)
            h_r, c_r = H_up[r].unsqueeze(0), C_up[r].unsqueeze(0)
            h, c = self.up_cell(h_l, c_l, h_r, c_r)
            H_up[p] = h.squeeze(0).clone()
            C_up[p] = c.squeeze(0).clone()
        
        # ----- Build Super-Root -----
        # Aggregate all forest roots by averaging their upward embeddings.
        forest_root_embs = torch.stack([H_up[r] for r in forest_roots], dim=0)
        super_root_up = forest_root_embs.mean(dim=0)  # (dim,)
        # Assign super-root an index: one greater than max_node.
        super_root_idx = max_node + 1
        H_up = torch.cat([H_up, super_root_up.unsqueeze(0)], dim=0)
        C_up = torch.cat([C_up, torch.zeros(1, dim, device=device)], dim=0)
        
        # ----- Downward Pass (Recursive, Functional) -----
        # Build mapping from parent to children.
        children_of = {p: (l, r) for (p, l, r) in forest_edges}
        
        def compute_downward(p, children_of, H_up, C_up, h_down_parent, c_down_parent):
            """
            Recursively compute downward states for node p.
            Returns a dictionary mapping node indices to (h_down, c_down) tuples.
            """
            d = {p: (h_down_parent, c_down_parent)}
            if p in children_of:
                l, r = children_of[p]
                # For the left child, use parent's downward state plus the right
                # child’s upward state as sibling.
                h_sib = H_up[r].unsqueeze(0)
                c_sib = C_up[r].unsqueeze(0)
                h_left, c_left = self.down_cell(h_down_parent, c_down_parent, h_sib, c_sib)
                # For the right child, use parent's downward state plus left sibling.
                h_sib = H_up[l].unsqueeze(0)
                c_sib = C_up[l].unsqueeze(0)
                h_right, c_right = self.down_cell(h_down_parent, c_down_parent, h_sib, c_sib)
                # Recurse on children:
                d.update(compute_downward(l, children_of, H_up, C_up,
                                            h_left.squeeze(0), c_left.squeeze(0)))
                d.update(compute_downward(r, children_of, H_up, C_up,
                                            h_right.squeeze(0), c_right.squeeze(0)))
            return d
        
        # Initialize downward state for super-root as zeros.
        h0 = torch.zeros(1, dim, device=device)
        c0 = torch.zeros(1, dim, device=device)
        down_dict = compute_downward(super_root_idx, children_of, H_up, C_up, h0, c0)
        
        # ----- Build Final Downward Tensor for Leaves -----
        # We construct H_down for leaves (indices 0 to n_res-1) from the dictionary.
        H_down = torch.zeros(n_res, dim, device=device)
        for i in range(n_res):
            if i in down_dict:
                H_down[i] = down_dict[i][0]
            else:
                H_down[i] = torch.zeros(dim, device=device)
        
        # ----- Combine Upward and Downward States -----
        if self.concat:
            leaves = torch.cat([H_up[:n_res], H_down], dim=-1)  # (n_res, 2*dim)
            super_root_vec = torch.cat([H_up[super_root_idx].clone(), down_dict[super_root_idx][0].flatten().clone()], dim=-1)
        else:
            leaves = H_up[:n_res] + H_down
            super_root_vec = H_up[super_root_idx] + down_dict[super_root_idx][0].flatten()
        return super_root_vec, leaves


        
def collate_item(batch):    
    label_key = 'fold_label' if 'fold_label' in batch[0] else 'residue_label'
    n, m = 0, 0
    for sample in batch:
        edges = sample['edges']
        embed = sample['embeddings']        
        n = max(n, embed.shape[0])        
        m = max(m, len(edges))
    all_embeds, all_edges, all_n, all_m, labels = [], [], [], [], []
    for sample in batch:
        edges = sample['edges']
        embed = sample['embeddings']                
        all_n.append(embed.shape[0])     
        all_m.append(len(edges))        
        embed = torch.cat((embed, torch.zeros((n-embed.shape[0], embed.shape[1])).to(embed.device)))
        all_embeds.append(embed)
        edges = edges + [[0, 0, 0] for _ in range(m-len(edges))]
        all_edges.append(edges)   
        labels.append(sample[label_key])
    return {'edges': torch.tensor(all_edges), 'embeddings': torch.stack(all_embeds, axis=0), 'n': torch.tensor(all_n), 'm': torch.tensor(all_m), label_key: torch.tensor(labels)}



def train_probe(args, train_dataset, valid_dataset, num_classes):    
    # Hyperparameters (adjust as needed)
    embed_dim = 960       # pretrained embedding dimension (for residues)
    hidden_dim = 128      # hidden dimension for probe
    # number of classes for fold classification
    device = torch.device(args.cuda)
    if args.level == "protein":
        probe = ProbingLayer(embed_dim * 2, hidden_dim, num_classes).to(device)
    else:
        probe = nn.Linear(embed_dim * 2, num_classes).to(device)

    # Instantiate the Tree Encoder (for up-down tree-LSTM with super-root)
    tree_encoder = UpDownTreeEncoder(embed_dim, concat_updown=True).to(device)

    if num_classes == 1:
        if args.regression:
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(probe.parameters()) + list(tree_encoder.parameters()), lr=0.001)

    if args.debug:
        dataset_kwargs = {
            'batch_size': 1,
            'shuffle': False,
            'collate_fn': collate_item
        }
    else:
        dataset_kwargs = {
            'batch_size': 1,
            'shuffle': True,
            'collate_fn': collate_item,
            # 'num_workers': 8,
            # 'persistent_workers': True
        }
    
    json.dump(args.__dict__, open(os.path.join(args.save_dir, 'args.json'), 'w+'))

    train_loader = DataLoader(train_dataset, **dataset_kwargs)
    valid_loader = DataLoader(valid_dataset, **dataset_kwargs)
    num_epochs = args.epochs

    best_val = -1.0
    # --- Early stopping config ---
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        probe.train()
        tree_encoder.train()
        train_loss = 0.0
        train_correct = 0
        num_train_samples = 0

        for batch in tqdm(train_loader, desc="looping over batches"):
            if args.debug and num_train_samples:
                break
            optimizer.zero_grad()
            # Expected in batch:
            #   'embeddings': (B, n_res_max, embed_dim)
            #   'edges': list of length B; each element is a list of (parent, left, right)
            #   'n': number of residues per protein (B,)
            #   'fold_label': protein-level labels (B,)
            #   Optionally: 'res_labels': residue-level labels if needed
            repr_tensor = batch['embeddings'].to(device)  # (B, n_res_max, embed_dim)
            edges_batch = batch['edges']                  # list (B) of edge-lists
            n_batch = batch['n']                          # (B,)
            B = repr_tensor.size(0)
            batch_logits = []
            all_labels = []
            batch_loss = [] # for residue level task, loss is calculated for each residue for each protein then averaged over the proteins in the batch

            for i in range(B):
                n = n_batch[i].item()
                # Get the residue embeddings for this protein.
                leaf_emb = repr_tensor[i, :n, :]          # (n, embed_dim)
                # Get the edge list (forest structure) for this protein.
                edges = edges_batch[i][:batch['m'][i].item()]  # list of (parent, left, right)
                # Assume also that edges are organized in topological order.
                # Also assume that forest_roots is provided or can be extracted.
                # For example, if each edge is (parent,left,right), then the set of roots
                # is computed as: all nodes that never appear as children.
                child_set = set()
                for (p, l, r) in edges:
                    child_set.add(l)
                    child_set.add(r)
                if len(edges):
                    forest_roots = [r for r in range(edges.max() + 1) if r not in child_set]
                else:
                    forest_roots = [r for r in range(n)]
                # If the protein is a single tree, forest_roots will be a singleton.
                # Use the tree encoder with super-root to get protein- and residue-level encodings.
                protein_vec, leaves = tree_encoder(leaf_emb, edges, n, forest_roots)
                if args.level == "protein":
                    logit = probe(protein_vec)           # (num_classes,)
                    batch_logits.append(logit)
                    all_labels.append(batch['fold_label'][i])
                else:
                    logits = probe(leaves)               # (num_residues, num_classes)
                    if num_classes == 1:
                        logits = logits.squeeze(1)  # (num_residues,)
                        item_loss = criterion(logits, batch['residue_label'][i].float().to(device))
                    else:
                        item_loss = criterion(logits, batch['residue_label'][i].to(device))
                    batch_loss.append(item_loss)
                    batch_logits.append(logits)
                    all_labels.append(batch['residue_label'][i].to(device))

            if args.level == "protein":
                batch_logits = torch.stack(batch_logits, dim=0)  # (B, num_classes)
                labels = torch.stack(all_labels).to(device)       # (B,)
                loss = criterion(batch_logits, labels)
                num_train_samples += B
            else:
                batch_logits = torch.concat(batch_logits, dim=0)  # (sum(n_residues), num_classes)
                labels = torch.concat(all_labels, dim=0)         # (sum(n_residues),)
                loss = torch.stack(batch_loss).mean()  # average over proteins in batch
                num_train_samples += batch_logits.size(0)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B
            if not args.regression:
                if num_classes == 1:        
                    preds = torch.sigmoid(batch_logits)
                    preds = (preds > 0.5).long()
                else:
                    preds = batch_logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / num_train_samples
        if not args.regression:
            train_accuracy = train_correct / num_train_samples

        # ----------------------
        # Validation Loop
        # ----------------------
        probe.eval()
        tree_encoder.eval()  # ensure the tree encoder is in eval mode as well
        valid_loss = 0.0
        valid_preds = []
        valid_labels = []
        valid_scores = []
        num_valid_samples = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="validation"):
                if args.debug and num_valid_samples:
                    break
                # Expected in batch:
                #   'embeddings': (B, n_res_max, embed_dim)
                #   'edges': list of length B; each element is a list of (parent, left, right)
                #   'n': number of residues per protein (B,)
                #   'fold_label': protein-level labels (B,)
                #   Optionally: 'res_labels': residue-level labels if needed
                repr_tensor = batch['embeddings'].to(device)  # (B, n_res_max, embed_dim)
                edges_batch = batch['edges']                  # list (B) of edge-lists
                n_batch = batch['n']                          # (B,)
                B = repr_tensor.size(0)
                if args.level == "protein":
                    labels = batch['fold_label'].to(device)
                else:
                    labels = batch['residue_label'].to(device)   # shape: (B,)                
                batch_logits = []
                batch_loss = [] # for residue level loss
                # Process each example in the batch.
                for i in range(B):
                    n = batch['n'][i].item()
                    # Get residue embeddings for this protein.
                    leaf_emb = repr_tensor[i, :n, :]        # (n, embed_dim)
                    # Get the forest edges for this protein.
                    edges = batch['edges'][i][: batch['m'][i].item()]  # list of (parent, left, right)                 
                    # Compute forest roots:
                    child_set = set()
                    for (p, l, r) in edges:
                        child_set.add(l)
                        child_set.add(r)
                    if len(edges):
                        forest_roots = [r for r in range(edges.max() + 1) if r not in child_set]
                    else:
                        forest_roots = [r for r in range(n)]                                            
                    # Use the tree encoder with super-root to get protein- and residue-level embeddings.
                    protein_vec, leaves = tree_encoder(leaf_emb, edges, n, forest_roots)                    
                    if args.level == "protein":
                        logit = probe(protein_vec)         # (num_classes,)
                        batch_logits.append(logit)
                    else:
                        logits = probe(leaves)               # (num_residues, num_classes)          
                        if num_classes == 1:
                            if args.regression:
                                ignore_index = torch.logical_or(labels.isnan(), (torch.abs(labels - (-100)) < 1e-6))
                                logits = logits.squeeze()[~ignore_index.squeeze()]
                                item_loss = criterion(logits, labels.squeeze())                                
                            else:
                                logits = logits.squeeze(1)  # (num_residues,)
                                item_loss = criterion(logits, batch['residue_label'][i].float().to(device))
                        else:
                            item_loss = criterion(logits, batch['residue_label'][i].to(device))
                        valid_loss += item_loss.item()
                        batch_logits.append(logits)
                
                if args.level == "protein":
                    batch_logits = torch.stack(batch_logits, dim=0)  # (B, num_classes)
                    loss = criterion(batch_logits, labels)
                    valid_loss += loss.item() * B
                    num_valid_samples += B
                else:
                    batch_logits = torch.concat(batch_logits, dim=0) # (sum(n_residues), num_classes)
                    num_valid_samples += batch_logits.size(0)
                
                if num_classes == 1:
                    if args.regression:
                        preds = batch_logits
                        labels = labels.squeeze(0)
                    else:
                        scores = torch.sigmoid(batch_logits)
                        preds = (scores > 0.5).long()
                        labels = labels.squeeze(0)
                        valid_scores.extend(scores.cpu().tolist())
                else:
                    preds = batch_logits.argmax(dim=1)

                valid_preds.extend(preds.cpu().tolist())                
                valid_labels.extend(labels.cpu().tolist())

        avg_valid_loss = valid_loss / num_valid_samples
        valid_accuracy = sum([1 for p, t in zip(valid_preds, valid_labels) if p == t]) / num_valid_samples
        
        # Compute macro F1 score on validation set.
        if num_classes == 1:
            if args.regression:
                val_metrics = calculate_regression_metric(torch.as_tensor(valid_preds), torch.as_tensor(valid_labels))
                val_pearson_r = val_metrics['pearsonr'].item()
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                    f"Valid Loss: {avg_valid_loss:.4f}, "
                    f"Valid Pearson R: {val_pearson_r:.4f}")                   
            else:
                val_auroc = roc_auc_score(valid_labels, valid_scores)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_accuracy*100:.2f}%, Valid Loss: {avg_valid_loss:.4f}, "
                    f"Valid Acc: {valid_accuracy*100:.2f}%, Valid AUROC: {val_auroc:.4f}")            
        else:
            val_macro_f1 = f1_score(valid_labels, valid_preds, average='macro')
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy*100:.2f}%, Valid Loss: {avg_valid_loss:.4f}, "
                f"Valid Acc: {valid_accuracy*100:.2f}%, Valid Macro F1: {val_macro_f1:.4f}")


        # ----------------------
        # Checkpoint saving
        # ----------------------
        if num_classes == 1:
            if args.regression and val_pearson_r > best_val:
                best_val = val_pearson_r
                os.makedirs(args.save_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.save_dir, f"best_checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': tree_encoder.state_dict(),
                    'probe_state_dict': probe.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_pearson_r': best_val,
                }, checkpoint_path)
                print(f"New best pearson r achieved: {best_val:.4f}. Saved checkpoint to {checkpoint_path}.")            
                epochs_no_improve = 0
            elif not args.regression and val_auroc > best_val:
                best_val = val_auroc
                os.makedirs(args.save_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.save_dir, f"best_checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': tree_encoder.state_dict(),
                    'probe_state_dict': probe.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auroc': best_val,
                }, checkpoint_path)
                print(f"New best auroc achieved: {best_val:.4f}. Saved checkpoint to {checkpoint_path}.")            
                epochs_no_improve = 0
        elif num_classes > 1 and val_macro_f1 > best_val:
            best_val = val_macro_f1
            os.makedirs(args.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.save_dir, f"best_checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': tree_encoder.state_dict(),
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_val,
            }, checkpoint_path)
            print(f"New best macro F1 achieved: {best_val:.4f}. Saved checkpoint to {checkpoint_path}.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if num_classes == 1:
                    if args.regression:
                        print(
                            f"Early stopping at epoch {epoch+1}: "
                            f"no val pearson r improvement for {patience} epochs "
                            f"(best val pearson r: {best_val:.4f})."
                        )                        
                    else:
                        print(
                            f"Early stopping at epoch {epoch+1}: "
                            f"no val auroc improvement for {patience} epochs "
                            f"(best val auroc: {best_val:.4f})."
                        )
                else:
                    print(
                        f"Early stopping at epoch {epoch+1}: "
                        f"no val macro F1 improvement for {patience} epochs "
                        f"(best val macro f1: {best_val:.4f})."
                    )                    
                break            

def test_probe(args, test_datasets, num_classes):
    """
    Load the best checkpoint from args.save_dir, then for each test dataset
    in test_datasets (a dict name->dataset or list of (name, dataset)),
    run evaluation and write metrics to a text file in args.save_dir.
    """
    device = torch.device(args.cuda)

    # 1) Find the best checkpoint file
    ckpt_paths = glob(os.path.join(args.save_dir, "best_checkpoint_epoch_*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint found matching best_checkpoint_epoch_*.pt in {args.save_dir}")
    # pick the most recently modified
    best_ckpt = max(ckpt_paths, key=os.path.getmtime)

    # 2) Initialize model components
    embed_dim = 960
    hidden_dim = 128

    # probe head
    if args.level == "protein":
        probe = ProbingLayer(embed_dim * 2, hidden_dim, num_classes).to(device)
    else:
        probe = nn.Linear(embed_dim * 2, num_classes).to(device)

    # tree encoder
    tree_encoder = UpDownTreeEncoder(embed_dim, concat_updown=True).to(device)

    # 3) Load checkpoint
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    tree_encoder.load_state_dict(ckpt["encoder_state_dict"])
    probe.load_state_dict(ckpt["probe_state_dict"])
    tree_encoder.eval()
    probe.eval()

    # 4) Prepare test datasets iteration
    if isinstance(test_datasets, dict):
        items = test_datasets.items()
    else:
        # assume list of (name, dataset)
        items = test_datasets

    results = []

    for name, dataset in items:
        # DataLoader for test
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_item
        )

        all_preds = []
        all_labels = []
        all_scores = []
        all_losses = []
        if num_classes == 1:
            if args.regression:
                criterion = nn.MSELoss()
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Test [{name}]"):
                # unpack batch
                repr_tensor = batch['embeddings'].to(device)  # (1, n_res_max, embed_dim)
                edges_batch  = batch['edges']
                n_batch      = batch['n']
                m            = batch['m'][0].item()
                if args.level == "protein":
                    label = batch['fold_label'].to(device)  # (1,)
                else:
                    label = batch['residue_label'].to(device)

                # single example
                n = n_batch[0].item()
                leaf_emb = repr_tensor[0, :n, :]  # (n, embed_dim)
                edges = edges_batch[0][:m]         # list of (parent,left,right)
                # compute forest roots
                child_set = set()
                for (p, l, r) in edges:
                    child_set.add(l)
                    child_set.add(r)
                if len(edges):
                    forest_roots = [r for r in range(edges.max() + 1) if r not in child_set]
                else:
                    forest_roots = [r for r in range(n)]

                # encode
                protein_vec, leaves = tree_encoder(leaf_emb, edges, n, forest_roots)

                # predict
                if args.level == "protein":
                    logits = probe(protein_vec.unsqueeze(0))  # (1, C)
                    pred = logits.argmax(dim=1)
                    loss = criterion(logits, label)
                    all_preds.append(pred.item())
                    all_labels.append(label.item())
                    all_losses.append(loss.item())
                else:
                    logits = probe(leaves)       # (num_residues, C)
                    if num_classes == 1:
                        if args.regression:
                            label = label.squeeze(0)
                            pred = logits.squeeze(1)
                            loss = criterion(logits.squeeze(), label)
                        else:
                            logits = logits.squeeze(1)
                            scores = torch.sigmoid(logits) # (num_residues,)
                            pred = (scores > 0.5).long()
                            label = label.squeeze(0)
                            loss = criterion(logits, label.float())
                            all_scores.extend(scores.cpu().tolist())
                    else:
                        pred = logits.argmax(dim=1) # (num_residues,)
                        loss = criterion(logits, label)
                    all_losses.append(loss.item())
                    all_preds.extend(pred.cpu().tolist())
                    all_labels.extend(label.cpu().tolist())

        # compute metrics
        if not args.regression:
            acc = accuracy_score(all_labels, all_preds)
        if num_classes == 1:
            if args.regression:
                metrics = calculate_regression_metric(torch.as_tensor(all_preds), torch.as_tensor(all_labels))
                mean_loss = sum(all_losses) / len(all_losses)
                results.append((name, mean_loss, 0.0, metrics))
            else:
                metric = roc_auc_score(all_labels, all_scores)
                mean_loss = sum(all_losses) / len(all_losses)
                results.append((name, mean_loss, acc, metric))
        else:
            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            mean_loss = sum(all_losses) / len(all_losses)
            results.append((name, mean_loss, acc, macro_f1))

    # 5) Write results to file
    out_path = os.path.join(args.save_dir, "test_results.txt")
    with open(out_path, "w") as f:
        for name, loss, acc, metric in results:
            f.write(f"Dataset: {name}\n")
            f.write(f"  Test Loss: {loss:.4f}\n")
            f.write(f"  Accuracy:  {acc*100:.2f}%\n")
            if num_classes == 1:
                if args.regression:                    
                    for k, v in metric.items():
                        f.write(f"  {k}:   {v.item():.4f}\n\n")
                else:
                    f.write(f"  AUROC:   {metric:.4f}\n\n")
            else:
                f.write(f"  Macro F1:   {metric:.4f}\n\n")                

    print(f"Test results written to {out_path}")

def load_datasets(args):
    bpe = pickle.load(open(args.pkl_file, 'rb'))
    if args.pkl_data_file:
        if os.path.exists(args.pkl_data_file):
            train_dataset, validation_dataset, test_datasets = pickle.load(open(args.pkl_data_file, 'rb'))            
            if isinstance(train_dataset, list):
                for train, val, tests in zip(train_dataset, validation_dataset, test_datasets):
                    print(f"Train: {len(train)}, Valid: {len(val)}, Test: {[len(x[1]) for x in tests]}")
            else:
                print(f"Train: {len(train_dataset)}, Valid: {len(validation_dataset)}, Test: {[len(x[1]) for x in test_datasets]}")
            return train_dataset, validation_dataset, test_datasets
        # train = LMDBDataset('data/remote_homology_raw/remote_homology_train.lmdb')
        # counts = Counter([x['fold_label'] for x in train])
        # class_labels = [c for c in counts if counts[c] > 50]
        # label_map = dict(zip(class_labels, range(len(class_labels))))
        # valid = LMDBDataset('data/remote_homology_raw/remote_homology_valid.lmdb')
        # test_family_holdout = LMDBDataset('data/remote_homology_raw/remote_homology_test_family_holdout.lmdb')
        # test_fold_holdout = LMDBDataset('data/remote_homology_raw/remote_homology_test_fold_holdout.lmdb')
        # test_superfamily_holdout = LMDBDataset('data/remote_homology_raw/remote_homology_test_superfamily_holdout.lmdb')
        # train_dataset = MyDataset(bpe.tokenizers, train, label_map, args.debug)
        # validation_dataset = MyDataset(bpe.tokenizers, valid, label_map, args.debug)
        # test_family_dataset = MyDataset(bpe.tokenizers, test_family_holdout, label_map)
        # test_fold_dataset = MyDataset(bpe.tokenizers, test_fold_holdout, label_map)
        # test_superfamily_dataset = MyDataset(bpe.tokenizers, test_superfamily_holdout, label_map)
        # test_datasets = [('family', test_family_dataset), ('fold', test_fold_dataset), ('superfamily', test_superfamily_dataset)]
    if args.task == "BindInt":
        prefix = 'data/struct_token_bench/InterProFunctionDataset_binding_label'
        test_splits = ['fold_test', 'superfamily_test']
    elif args.task == "BindBio":
        prefix = 'data/struct_token_bench/BioLIP2FunctionDataset_catalytic_label'
        test_splits = ['fold_test', 'superfamily_test']
    elif args.task == "BindShake":
        prefix = 'data/struct_token_bench/ProteinShakeBindingSiteDataset_binding_site'
        test_splits = ['test']
    elif args.task == "repeat-motif-prediction":            
        prefix = 'data/struct_token_bench/InterProFunctionDataset_repeat_label'
        test_splits = ['fold_test', 'superfamily_test']
    elif args.task == "CatInt":
        prefix = 'data/struct_token_bench/InterProFunctionDataset_activesite_label'
        test_splits = ['fold_test', 'superfamily_test']
    elif args.task == "CatBio":
        prefix = 'data/struct_token_bench/BioLIP2FunctionDataset_catalytic_label'
        test_splits = ['fold_test', 'superfamily_test']
    elif args.task == "conserved-site-prediction":
        prefix = 'data/struct_token_bench/InterProFunctionDataset_conservedsite_label'
        test_splits = ['fold_test', 'superfamily_test']
    elif args.task == "epitope-prediction":
        prefix = 'data/struct_token_bench/ProteinGLUEEpitopeRegionDataset_epitope_label'
        test_splits = ['fold_test', 'superfamily_test']
    elif args.task == "structural-flexibility-prediction":
        prefix = [f'data/struct_token_bench/AtlasDataset_{metric}_score' for metric in ['rmsf', 'neq', 'bfactor']]
        test_splits = [['fold_test', 'superfamily_test'] for metric in ['rmsf', 'neq', 'bfactor']]
    elif args.task == "remote-homology-detection":
        prefix = 'data/struct_token_bench/TapeRemoteHomologyDataset_fold_label'
        test_splits = ['test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout']            
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")
    # get processed structtokenbench files
    if isinstance(prefix, list):
        train_datasets, validation_datasets, tests_datasets = [], [], []
        for i, pre in enumerate(prefix):
            datasets = {}
            for split in ["train", "validation"] + test_splits[i]:
                with open(f'{pre}_{split}.jsonl', 'r') as f:
                    datasets[f"{split}_dataset"] = [json.loads(line) for line in f]        
                datasets[f"{split}_dataset"] = (MyDataset if args.task == "remote-homology-detection" else ResidueDataset)(bpe.tokenizers, datasets[f"{split}_dataset"], args.debug)
            test_datasets = [(split, datasets[f"{split}_dataset"]) for split in test_splits[i]]
            train_dataset, validation_dataset = datasets["train_dataset"], datasets["validation_dataset"]            
            train_datasets.append(train_dataset)
            validation_datasets.append(validation_dataset)
            tests_datasets.append(test_datasets)
            print(f"Train: {len(train_dataset)}, Valid: {len(validation_dataset)}, Test: {[len(x[1]) for x in test_datasets]}")
        if args.pkl_data_file:
            pickle.dump((train_datasets, validation_datasets, tests_datasets), open(args.pkl_data_file, 'wb+'))
        return train_datasets, validation_datasets, tests_datasets            
    else:
        datasets = {}
        for split in ["train", "validation"] + test_splits:
            with open(f'{prefix}_{split}.jsonl', 'r') as f:
                datasets[f"{split}_dataset"] = [json.loads(line) for line in f]        
            datasets[f"{split}_dataset"] = (MyDataset if args.task == "remote-homology-detection" else ResidueDataset)(bpe.tokenizers, datasets[f"{split}_dataset"], args.debug)
        test_datasets = [(split, datasets[f"{split}_dataset"]) for split in test_splits]   
        train_dataset, validation_dataset = datasets["train_dataset"], datasets["validation_dataset"]
        if args.pkl_data_file:
            pickle.dump((train_dataset, validation_dataset, test_datasets), open(args.pkl_data_file, 'wb+'))        
        print(f"Train: {len(train_dataset)}, Valid: {len(validation_dataset)}, Test: {[len(x[1]) for x in test_datasets]}")
        return train_dataset, validation_dataset, test_datasets



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

    # train using train_dataset, validate with valid_dataset, ignore test_* datasets for now
    train_dataset, valid_dataset, test_datasets = load_datasets(args)
    if args.task == "structural-flexibility-prediction":
        for i, (train, valid, tests) in enumerate(zip(train_dataset, valid_dataset, test_datasets)):
            if not args.test:
                train_probe(args, train, valid, num_classes=1)
            test_probe(args, tests, num_classes=1)
            prefix = ['rmsf', 'neq', 'bfactor'][i]
            for file in glob(os.path.join(args.save_dir, "*.pt"))+glob(os.path.join(args.save_dir, "*.txt")):
                os.rename(file, Path(file).parent / (prefix+'_'+Path(file).name))
            # distinguish the artifacts
    elif args.task == "remote-homology-detection":
        if not args.test:
            train_probe(args, train_dataset, valid_dataset, num_classes=45)
        test_probe(args, test_datasets, num_classes=45)
    else:
        if not args.test:
            train_probe(args, train_dataset, valid_dataset, num_classes=train_dataset.num_classes)
        test_probe(args, test_datasets, num_classes=train_dataset.num_classes)
    

if __name__ == "__main__":    
    args = parse_args()
    client = ESMC.from_pretrained("esmc_300m").to(args.cuda) # or "cpu"        
    main(args)
