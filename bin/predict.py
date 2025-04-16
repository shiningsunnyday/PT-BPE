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
from torch.utils.data import Dataset, default_collate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_AVAILABLE_DEVICES'] = "" # debug

# Create a global lock for GPU operations.
gpu_lock = threading.Lock()


def traverse(tree):
    def _postorder(n, nodes, edges):
        assert (n.left is None) == (n.right is None), f"{n} {n.left} {n.right}"        
        if n.left:           
            level = _postorder(n.left, nodes, edges)
            level = max(level, _postorder(n.right, nodes, edges))            
            if level >= len(edges):
                edges.append([])
            edges[level].append((n, n.left, n.right))
            level += 1
        else: # reached leaf
            assert n.value[-1] == 1
            level = 0
        if level >= len(nodes):
            nodes.append([])
        nodes[level].append(n)
        return level
    all_edges = []
    all_nodes = []
    nmap = {}
    roots = [tree.nodes[k] for k in sorted(tree.nodes)]    
    for root in roots:
        nodes, edges = [], []
        _postorder(root, nodes, edges) # appends [[nodes of level i] for i]
        all_edges.append(edges)
        all_nodes.append(nodes[::-1])
    while True:
        stop = True
        for nodes in all_nodes:
            if len(nodes) == 0:
                continue
            stop = False
            for n in nodes.pop(-1):
                if n.value[-1] == 1:
                    assert n.value[0] == len(nmap)
                nmap[n.value] = len(nmap)
        if stop:
            break
    all_edges = [(nmap[p.value], nmap[l.value], nmap[r.value]) for edges in all_edges for edges_level in edges for p, l, r in edges_level]
    return nmap, all_edges


def compute_embedding(item):
    _id, chain, _, _ = item
    protein = ESMProtein.from_protein_chain(ProteinChain.from_rcsb(_id, chain_id=chain))    
    # Ensure only one thread performs GPU operations at a time.
    with gpu_lock:
        protein_tensor = client.encode(protein)
        output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
    embed = output.embeddings[0, 1:-1].to(torch.float32).to('cpu')
    return embed



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
        self.precompute()
    

    def precompute(self):
        # debug, comment out
        self.esm_outputs = [torch.rand((sample['protein_length'], 960)).to(torch.float32).to('cpu') for _,_,_,sample in self.data]
        return
        # end debug
        self.esm_outputs = []
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(compute_embedding, self.data),
                total=len(self.data),
                desc="precomputing esm embeddings"
            ))
        self.esm_outputs.extend(results)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _, chain, t, sample = self.data[idx]
        item = sample
        # item['protein'] = protein
        # item['coords'] = t.compute_coords()
        tree = t.bond_to_token.tree
        nmap, edges = traverse(tree)
        item['edges'] = edges        
        item['embeddings'] = self.esm_outputs[idx]
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
    parser.add_argument("--task", choices=["remote-homology-detection", # per-protein
                                            "structural-flexibility-prediction", # per-residue regression
                                            "binding-site-prediction", "catalytic-site-prediction", "conserved-site-prediction", "repeat-motif-prediction", "epitope-region-prediction" # per-residue classification
    ])
    # data
    parser.add_argument("--pkl-file", type=str, required=True, 
                        help="Load the BPE results.")    
    parser.add_argument("--save-dir", type=str, default="plots/models", 
                        help="Directory to save ckpts.")
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory where log files will be saved.")
    parser.add_argument("--auto", action='store_true', help='auto set folders')
    # training
    parser.add_argument("--cuda", default="cpu")
    # hparams
    parser.add_argument("--p_min_size", default=float("inf"), help="when to start using rmsd binning")
    parser.add_argument("--level", default="protein", help="prediction at protein or residue level", choices=["protein", "residue"])
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
        labels.append(sample['fold_label'])
    return {'edges': torch.tensor(all_edges), 'embeddings': torch.stack(all_embeds, axis=0), 'n': torch.tensor(all_n), 'm': torch.tensor(all_m), 'fold_label': torch.tensor(labels)}



def train_probe(args, train_dataset, valid_dataset):
    # Hyperparameters (adjust as needed)
    embed_dim = 960       # pretrained embedding dimension (for residues)
    hidden_dim = 128      # hidden dimension for probe
    num_classes = 45      # number of classes for fold classification

    device = torch.device(args.cuda)
    if args.level == "protein":
        probe = ProbingLayer(embed_dim * 2, hidden_dim, num_classes).to(device)
    else:
        probe = nn.Linear(embed_dim * 2, num_classes).to(device)

    # Instantiate the Tree Encoder (for up-down tree-LSTM with super-root)
    tree_encoder = UpDownTreeEncoder(embed_dim, concat_updown=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(probe.parameters()) + list(tree_encoder.parameters()), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_item, num_workers=8, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_item, num_workers=8, persistent_workers=True)
    num_epochs = 10

    best_val_macro_f1 = -1.0

    for epoch in range(num_epochs):
        probe.train()
        tree_encoder.train()
        train_loss = 0.0
        train_correct = 0
        num_train_samples = 0

        for batch in tqdm(train_loader, desc="looping over batches"):
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
                forest_roots = [r for r in range(n, max([p for p,_,_ in edges]) + 1) if r not in child_set]
                # If the protein is a single tree, forest_roots will be a singleton.
                # Use the tree encoder with super-root to get protein- and residue-level encodings.
                protein_vec, leaves = tree_encoder(leaf_emb, edges, n, forest_roots)
                if args.level == "protein":
                    logit = probe(protein_vec)           # (num_classes,)
                    batch_logits.append(logit)
                    all_labels.append(batch['fold_label'][i])
                else:
                    logits = probe(leaves)               # (n, num_classes)
                    logit = logits.mean(dim=0)           # aggregate per residue
                    batch_logits.append(logit)
                    all_labels.append(batch['fold_label'][i])
            
            batch_logits = torch.stack(batch_logits, dim=0)  # (B, num_classes)
            labels = torch.stack(all_labels).to(device)       # (B,)
            loss = criterion(batch_logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B
            preds = batch_logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            num_train_samples += B

        avg_train_loss = train_loss / num_train_samples
        train_accuracy = train_correct / num_train_samples

        # ----------------------
        # Validation Loop
        # ----------------------
        probe.eval()
        tree_encoder.eval()  # ensure the tree encoder is in eval mode as well
        valid_loss = 0.0
        valid_preds = []
        valid_labels = []
        num_valid_samples = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="validation"):
                # For validation, assume:
                #  - batch['output'].embeddings: tensor (B, seq_length, embed_dim)
                #  - batch['edges']: list (B) of edge lists (each is a list of (parent, left, right))
                #  - batch['n']: number of residues per protein (B,)
                #  - batch['fold_label']: protein-level labels (B,)
                #  - batch['m']: number of valid edges per protein (B,)
                embeddings = batch['output'].embeddings  # shape: (B, seq_length, embed_dim)
                # In our training loop we assumed the first and last tokens are special, so we remove them.
                repr_tensor = embeddings[:, 1:-1, :]      # shape: (B, n_residues, embed_dim)
                repr_tensor = repr_tensor.to(device)
                labels = batch['fold_label'].to(device)   # shape: (B,)
                
                batch_logits = []
                # Process each example in the batch.
                B = repr_tensor.size(0)
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
                    # All nodes from n to max_edge_parent are potential roots.
                    potential_roots = list(range(n, max([p for p, _, _ in edges]) + 1))
                    forest_roots = [r for r in potential_roots if r not in child_set]
                    
                    # Use the tree encoder with super-root to get protein- and residue-level embeddings.
                    protein_vec, leaves = tree_encoder(leaf_emb, edges, n, forest_roots)
                    
                    if args.level == "protein":
                        logit = probe(protein_vec)         # (num_classes,)
                    else:
                        # Residue-level: predict per residue then aggregate, e.g. via mean.
                        logits = probe(leaves)              # (n, num_classes)
                        logit = logits.mean(dim=0)          # aggregate to one per protein
                    batch_logits.append(logit)
                
                batch_logits = torch.stack(batch_logits, dim=0)  # (B, num_classes)
                loss = criterion(batch_logits, labels)
                valid_loss += loss.item() * B
                
                preds = batch_logits.argmax(dim=1)
                valid_preds.extend(preds.cpu().tolist())
                valid_labels.extend(labels.cpu().tolist())
                num_valid_samples += B
        
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



def load_datasets(args):
    if args.task == "remote-homology-detection":
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
    elif args.task == "binding-site-prediction":
        breakpoint()



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

    # train using train_dataset, validate with valid_dataset, ignore test_* datasets for now
    train_dataset, valid_dataset, _ = load_datasets(args)
    train_probe(args, train_dataset, valid_dataset)
    

if __name__ == "__main__":    
    args = parse_args()
    client = ESMC.from_pretrained("esmc_300m").to(args.cuda) # or "cpu"        
    breakpoint()
    main(args)
