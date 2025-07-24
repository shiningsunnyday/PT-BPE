import os
import tempfile
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import math
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from pathlib import Path
from pyzernike import ZernikeDescriptor
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from biotite.structure.io.pdb import PDBFile
from Bio.PDB import PDBParser, DSSP
import pickle
from foldingdiff.foldseek import *

HYDROPATHY = {
    # Kyte–Doolittle index
    "A": 1.8, "C": 2.5, "D":-3.5, "E":-3.5, "F": 2.8, "G":-0.4,
    "H":-3.2, "I": 4.5, "K":-3.9, "L": 3.8, "M": 1.9, "N":-3.5,
    "P":-1.6, "Q":-3.5, "R":-4.5, "S":-0.8, "T":-0.7, "V": 4.2,
    "W":-0.9, "Y":-1.3, "X": 0.0, "-": 0.0,
}

def voxelize(coords, grid_size=64, padding=2.0):
    """
    Convert 3D points into a binary occupancy grid.
    - coords: (n_points,3) numpy array
    - grid_size: number of voxels per axis
    - padding: extra Ångstroms around the min/max before voxelizing
    """
    # Compute bounding box
    mins = coords.min(axis=0) - padding
    maxs = coords.max(axis=0) + padding
    # Voxel spacing
    spacing = (maxs - mins) / (grid_size - 1)
    # Initialize grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    # Map each atom into voxel coords
    ijk = ((coords - mins) / spacing).astype(int)
    # Clip to valid range
    ijk = np.clip(ijk, 0, grid_size - 1)
    # Set occupancy
    for (i, j, k) in ijk:
        grid[i, j, k] = 1.0
    return grid


def compute_3d_zernike(grid, order=8):
    """
    Compute the 3D Zernike descriptor up to a given order.
    """
    # Fit descriptor
    zd = ZernikeDescriptor.fit(data=grid, order=order)
    # Extract the invariant coefficients
    coeffs = zd.get_coefficients()
    return coeffs

    

class BackboneResidueFeaturizer(nn.Module):
    """
    Produce a tensor of shape (seq_len, feat_dim) for a single protein chain.
    """
    def __init__(self):
        super().__init__()
        self.aa_to_idx = {aa:i for i,aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.out_dim = 20 + 1 + 1 + 3 + 1 + 10         # AA one‑hot + hydropathy + disorder + 3‑state SS + pLDDT + embedding
        self.labels = [f"aa_one_hot_{i}" for i in range(20)] + \
                      ["hydropathy", "disorder"] + \
                      [f"ss_{i}" for i in range(3)] + \
                      ["plddt"] + \
                      [f"embedding_{i}" for i in range(10)]
        # self.one_hot_projector = nn.Linear(20, 20)
        # self.hydropathy_projector = nn.Linear(1, 10)
        self.embedding_projector = nn.Linear(2560, 10)

    def forward(self, aa_seq:str,
                      ss_pred:str=None,          # secondary structure 'H/E/C' string from e.g. PSIPRED
                      disorder:np.ndarray=None,  # per‑res residue disorder score [0,1]
                      plddt:np.ndarray=None,
                      embedding:np.ndarray=None):    # AlphaFold confidence

        L = len(aa_seq)
        device = next(self.parameters()).device
        one_hot = torch.zeros(L, 20).to(device)
        hydro   = torch.zeros(L, 1).to(device)
        for i, aa in enumerate(aa_seq):
            if aa in self.aa_to_idx:
                one_hot[i, self.aa_to_idx[aa]] = 1.0
            hydro[i,0] = HYDROPATHY.get(aa, 0.0)

        # Disorder (fallback 0.0)
        if disorder is None: disorder = np.zeros(L)
        disorder = torch.from_numpy(disorder).float().unsqueeze(-1)
        disorder = disorder.to(device)

        # Secondary structure to one‑hot
        if ss_pred is None: ss_pred = "C"*L
        ss_map = {"H":0, "E":1, "C":2}
        ss_onehot = torch.zeros(L, 3).to(device)
        for i, s in enumerate(ss_pred):
            ss_onehot[i, ss_map.get(s,"C")] = 1.0
        
        if plddt is None: plddt = np.zeros(L)
        plddt = torch.from_numpy(plddt).float().unsqueeze(-1) / 100.0  # scale 0…1
        plddt = plddt.to(device)
        if embedding is None: embedding = np.zeros(L, 2560)
        embedding = torch.from_numpy(embedding).float().to(device)
        embedding = self.embedding_projector(embedding)
        return torch.cat([one_hot, hydro, disorder, ss_onehot, plddt, embedding], dim=-1)


class SegmentFeatureAggregator(nn.Module):
    """
    Turn a span X[i:j] (tensor (len, feat_dim)) into a fixed vector.
    We concatenate:
      – mean of each feature over the span
      – variance
      – length (normalised)
      – log‑length
    """
    def __init__(self, per_res_dim:int, per_res_labels: List[str]):
        super().__init__()
        self.per_res_dim = per_res_dim
        self.per_res_labels = [f"{feat}_{suffix}" for feat in per_res_labels \
            for suffix in ["mean", "std"]] + ["length", "log_length"]
        self.out_dim = 2*per_res_dim + 2  # mean + var + len + log(len)

    def forward(self, span:torch.Tensor):
        # span: (L, per_res_dim)
        mean = span.mean(dim=0)
        var  = span.var(dim=0, unbiased=False)
        L = span.size(0)
        length = torch.tensor([L], dtype=span.dtype, device=span.device)
        log_length = torch.log(length.float()+1e-6)
        out = torch.cat([mean, var, length/500.0, log_length/6.0], dim=0)
        assert out.shape[-1] == self.out_dim
        return out


class SegmentPairFeatureAggregator(nn.Module):
    """
    Turn a span X[i:j] (tensor (len, feat_dim)) into a fixed vector.
    We concatenate:
      – mean of each feature over the span
      – variance
      – length (normalised)
      – log‑length
    """
    def __init__(self, per_res_dim:int, per_res_labels: List[str]):
        super().__init__()
        self.per_res_dim = per_res_dim
        self.per_res_labels = [f"{feat}_{suffix}" for feat in per_res_labels \
            for suffix in ["mean", "std"]] + ["length", "log_length"]
        self.out_dim = 2*per_res_dim + 2  # mean + var + len + log(len)

    def forward(self, span:torch.Tensor):
        # span: (L, per_res_dim)
        mean = span.mean(dim=0)
        var  = span.var(dim=0, unbiased=False)
        L = span.size(0)
        length = torch.tensor([L], dtype=span.dtype, device=span.device)
        log_length = torch.log(length.float()+1e-6)
        out = torch.cat([mean, var, length/500.0, log_length/6.0], dim=0)
        assert out.shape[-1] == self.out_dim
        return out


class SegmentPotentialMLP(nn.Module):
    """
    Attention-based potential:
      - learns a per-feature attention over the agg_vec
      - computes a context vector and maps it to a scalar potential
      - returns both the potential and the attention weights
    """
    def __init__(self, agg_dim:int, attn_dim:int=32, hidden:int=64):
        super().__init__()
        # project each feature scalar → hidden
        self.feature_proj = nn.Linear(1, attn_dim, bias=False)
        # a learned query vector for scoring
        self.query = nn.Parameter(torch.randn(attn_dim))
        # final MLP on the weighted summary
        self.mlp = nn.Sequential(
            nn.Linear(attn_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, agg_vec: torch.Tensor):
        """
        Args:
          agg_vec: (batch, D)  the D semantic features for each segment
        Returns:
          phi    : (batch,)    the scalar log-potential
          weights: (batch, D)  the attention weight on each feature
        """
        B, D = agg_vec.shape
        # 1) Make it (batch, D, 1) so we can project each feature separately
        x = agg_vec.unsqueeze(-1)              # (B, D, 1)
        
        # 2) Project each feature into attn_dim
        K = self.feature_proj(x)               # (B, D, attn_dim)
        # 3) Score each feature against the query vector
        #    Dot-product: (B, D, attn_dim) · (attn_dim,) → (B, D)
        scores = K @ self.query                # (B, D)
        
        # 4) Normalize into attention weights
        weights = torch.softmax(scores, dim=-1) # (B, D)
        
        # 5) Compute context vector: weighted sum of the projected features
        #    (B, D, attn_dim) * (B, D, 1) → sum dim=1 → (B, attn_dim)
        ctx = (K * weights.unsqueeze(-1)).sum(dim=1)  # (B, attn_dim)
        
        # 6) Map to scalar potential
        phi = self.mlp(ctx).squeeze(-1)        # (B,)
        
        return phi, weights



class SegmentPairPotentialMLP(nn.Module):
    """
    Attention-based potential:
      - learns a per-feature attention over the agg_vec
      - computes a context vector and maps it to a scalar potential
      - returns both the potential and the attention weights
    """
    def __init__(self, agg_dim:int, attn_dim:int=32, hidden:int=64):
        super().__init__()
        # project each feature scalar → hidden
        self.feature_proj = nn.Linear(1, attn_dim, bias=False)
        # a learned query vector for scoring
        self.query = nn.Parameter(torch.randn(attn_dim))
        # final MLP on the weighted summary
        self.mlp = nn.Sequential(
            nn.Linear(attn_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, agg_vec: torch.Tensor):
        """
        Args:
          agg_vec: (batch, D)  the D semantic features for each segment
        Returns:
          phi    : (batch,)    the scalar log-potential
          weights: (batch, D)  the attention weight on each feature
        """
        B, D = agg_vec.shape
        # 1) Make it (batch, D, 1) so we can project each feature separately
        x = agg_vec.unsqueeze(-1)              # (B, D, 1)
        
        # 2) Project each feature into attn_dim
        K = self.feature_proj(x)               # (B, D, attn_dim)
        # 3) Score each feature against the query vector
        #    Dot-product: (B, D, attn_dim) · (attn_dim,) → (B, D)
        scores = K @ self.query                # (B, D)
        
        # 4) Normalize into attention weights
        weights = torch.softmax(scores, dim=-1) # (B, D)
        
        # 5) Compute context vector: weighted sum of the projected features
        #    (B, D, attn_dim) * (B, D, 1) → sum dim=1 → (B, attn_dim)
        ctx = (K * weights.unsqueeze(-1)).sum(dim=1)  # (B, attn_dim)
        
        # 6) Map to scalar potential
        phi = self.mlp(ctx).squeeze(-1)        # (B,)
        
        return phi, weights



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional encoding module.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on
        # position and i (dimension).
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Div term: exponential decay factors
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LongSequenceGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=1, bidirectional=False):
        """
        A lightweight GRU-based model to encode a sequence of dihedral angles,
        now returning a per-timestep output after applying a fully connected layer.
        """
        super(LongSequenceGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional)
        
        self.embedding_size = hidden_size * (2 if bidirectional else 1)
        
        # Dense layer that will be applied to each timestep's hidden state.
        self.fc = nn.Linear(self.embedding_size, 1)

    def forward(self, x, lengths):
        """
        Forward pass.
        
        Parameters:
          x (torch.Tensor): Padded tensor of shape (batch, max_seq_len, input_size).
          lengths (list or torch.Tensor): The true lengths of each sequence in the batch.
        
        Returns:
          out_all_steps (torch.Tensor): shape (batch, max_seq_len, 1), 
                                        containing an output scalar for each time step.
          hidden (torch.Tensor): The final hidden state(s) of shape 
                                 (num_layers * num_directions, batch, hidden_size).
        """
        # Pack the padded sequence for efficient processing.
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Run the GRU
        packed_out, hidden = self.gru(packed_x)  
        # packed_out is a PackedSequence of shape (sum(all seq lengths), hidden_size * num_directions)
        
        # Convert back to a padded sequence so we have (batch, max_seq_len, hidden_dim)
        # hidden_dim = hidden_size * num_directions
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )
        # padded_out: (batch, max_seq_len, embedding_size)
        
        # Apply fully connected layer to each timestep. This yields (batch, max_seq_len, 1).
        out_all_steps = self.fc(padded_out)
        
        # out_all_steps is (batch, max_seq_len, 1)
        return out_all_steps, hidden


# ---------------------------------------------------------------------
# 2.  Semi‑CRF wrapper
# ---------------------------------------------------------------------

class SemiCRFModel(nn.Module):
    """
    Combines:
      – token‑level encoder  (AngleTransformer or BertForDiffusionBase)
      – residue‑wise biochemical features  -> pooled segment vector
      – segment MLP potential
      – optional length bias γ
    The `forward()` returns the *segment score table* out[i][l].
    """
    def __init__(
        self,        
        res_featurizer: BackboneResidueFeaturizer,
        seg_aggregator: SegmentFeatureAggregator,
        seg_potential: SegmentPotentialMLP,
        encoder: Optional[nn.Module] = None,
        length_bias: float = 0.0,
        max_seg_len: int = 100,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.encoder         = encoder
        self.featurizer      = res_featurizer
        self.aggregator      = seg_aggregator
        self.seg_mlp         = seg_potential
        self.gamma           = length_bias
        self.max_seg_len     = max_seg_len
        self.device          = device or torch.device("cpu")
        if self.encoder:
            # linear projection from encoder hidden -> scalar
            self.enc2score = nn.Linear(
                encoder.config.hidden_size if hasattr(encoder, "config") else encoder.d_model,
                1,
            )
        else:
            self.zernike_projector = nn.Linear(9, 1)
            self.feats = defaultdict(dict)
    
    @staticmethod
    # Top‐level helper must be importable (no self):
    def _compute_protein_fps(args):
        """
        Compute all (i,j) fingerprints for a single protein t.
        Returns (prot_id, { (i,j): torch.Tensor(fp) }).
        """
        t, grid_size, padding, order = args
        prot_id = Path(t.fname).stem
        aa_seq  = t.aa
        coords  = t.compute_coords()
        # ensure numpy for voxelize / 3DZD
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords

        fps_dict = {}
        for i in range(len(aa_seq)):
            for j in range(i+1, len(aa_seq)+1):
                subcoords = coords_np[3*i:3*j]
                grid      = voxelize(subcoords, grid_size=grid_size, padding=padding)
                fp        = compute_3d_zernike(grid, order=order)
                fps_dict[(i, j)] = fp

        return prot_id, fps_dict

    @staticmethod
    # Top‐level helper must be importable (no self):
    def _compute_protein_plddt(args, batch_size):
        """
        Compute all pLDDT confidences for a batch of proteins.
        Returns (prot_ids, [torch.Tensor(confidences)])
        """
        # 1) Gather IDs (here we assume .aa is the sequence string)
        prot_ids = [Path(t.fname).stem for t in args]
        sequences = [t.aa for t in args]  # or [t.sequence for t in args], adjust as needed

        # 2) Create a temp file for input and dump the sequences
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as in_f:
            for seq in sequences:
                in_f.write(seq+'\n')
            in_path = in_f.name
        print(in_path)
        # 3) Reserve an output path
        out_fd, out_path = tempfile.mkstemp(suffix='.pt')
        os.close(out_fd)
        # locate your script relative to this file
        python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "get_plddt.py")
        try:
            # this will invoke the python inside the esmfold env
            subprocess.run([
                "conda", "run", "-n", "esmfold",
                "python", python_path,
                "--in-file",  in_path,
                "--out-file", out_path,
                "--batch-size", str(batch_size)
            ], check=True)
            plddts = torch.load(out_path)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("=== STDOUT ===")
            print(e.stdout)
            print("=== STDERR ===")
            print(e.stderr)
        finally:
            # 6) Clean up
            os.remove(in_path)
            os.remove(out_path)
        print("PLDDT prediction completed.")
        return prot_ids, plddts
    
    @staticmethod
    def _compute_protein_disorder(args):
        """
        Compute all disorder scores for a batch of proteins.
        Returns (prot_ids, [list of disorder scores])
        """       
        # 1) Gather protein IDs and sequences
        prot_ids = [Path(t.fname).stem for t in args]
        sequences = [t.aa for t in args]  # or [t.sequence for t in args], adjust as needed

        # 2) Create a temp file for input and dump the sequences in FASTA format
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.seq', delete=False) as in_f:
            for seq, prot_id in zip(sequences, prot_ids):
                in_f.write(f">{prot_id}\n{seq}\n")
            in_path = in_f.name

        # 3) Path to the iupred2a.py script (modify this according to your setup)
        python_path = os.path.join(Path(__file__).parents[2], "iupred2a/iupred2a.py")
        
        try:
            # 4) Run iupred2a.py using subprocess, capture stdout directly
            result = subprocess.run(
                ["python", python_path, in_path, "long"],  # Assumes "long" mode for disorder prediction
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True  # Capture the output as text
            )            
            # 5) Parse the stdout to extract the per-residue disorder scores
            disorder_scores = []
            lines = result.stdout.split('\n')            
            last = 0
            for seq in sequences:
                scores = []
                for a, l in zip(seq, lines[last: last+len(seq)]):
                    parts = l.split('\t')
                    aa = parts[1]
                    assert a == aa
                    residue_score = float(parts[2])  # The score is in the third column
                    scores.append(residue_score)
                disorder_scores.append(scores)
                last = last + len(seq)
            assert last + 1 == len(lines)

        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("=== STDOUT ===")
            print(e.stdout)
            print("=== STDERR ===")
            print(e.stderr)
        finally:
            # 6) Clean up the temporary file
            os.remove(in_path)
        
        print("Disorder prediction completed.")
        return prot_ids, disorder_scores      


    @staticmethod
    def _compute_protein_embedding(args, batch_size):
        """
        Compute all embeddings for a batch of proteins.
        Returns (prot_ids, [list of embeddings scores])
        """       
        # 1) Gather IDs (here we assume .aa is the sequence string)
        prot_ids = [Path(t.fname).stem for t in args]
        sequences = [t.aa for t in args]  # or [t.sequence for t in args], adjust as needed

        # 2) Create a temp file for input and dump the sequences
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as in_f:
            for seq in sequences:
                in_f.write(seq+'\n')
            in_path = in_f.name

        # 3) Reserve an output path
        out_fd, out_path = tempfile.mkstemp(suffix='.pt')
        os.close(out_fd)
        # locate your script relative to this file
        python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "get_embedding.py")
        try:
            # this will invoke the python inside the esmfold env
            subprocess.run([
                "conda", "run", "-n", "esmfold",
                "python", python_path,
                "--in-file",  in_path,
                "--out-file", out_path,
                "--batch-size", str(batch_size)
            ], check=True)
            embeddings = torch.load(out_path)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("=== STDOUT ===")
            print(e.stdout)
            print("=== STDERR ===")
            print(e.stderr)
        finally:
            # 6) Clean up
            os.remove(in_path)
            os.remove(out_path)
        print("Embedding done.")
        return prot_ids, embeddings  


    @staticmethod
    def _compute_protein_sec(args):
        """
        Compute all sec features for a batch of proteins.
        Returns (prot_ids, [list of sec types])
        """       
        # 1) Gather protein IDs and sequences
        prot_ids = [Path(t.fname).stem for t in args]
        ss_preds = []
        for t in args:
            fname = t.fname
            parser = PDBParser()
            structure = parser.get_structure(Path(fname).stem, fname)
            import biotite.structure as struc
            source_struct = PDBFile.read(open(t.fname)).get_structure(model=1)
            backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
            res_ids = []
            for a in backbone_atoms:
                if len(res_ids) and res_ids[-1] == a.res_id:
                    continue
                res_ids.append(a.res_id)
            assert len(res_ids) == len(t.aa)
            model = structure[0]  # assuming you want the first model            
            try:
                dssp = DSSP(model, fname)
                print("dssp good")
            except:
                ss_preds.append("".join(["C" for _ in res_ids]))
                continue
            ss_map = {"H":"H", "G":"H", "I":"H",  # helix types → H
                    "E":"E", "B":"E",           # sheet types → E
                    "T":"C", "S":"C", " ": "C", "-": "C"} # turns/others → C
            ss_list = [None for _ in t.aa]
            ss_dict = {}
            for key in dssp.keys():
                _, (_, res_id, _) = key                
                ss_dict[res_id] = ss_map[dssp[key][2]]
            for i, res_id in enumerate(res_ids):
                if res_id in ss_dict:
                    ss_list[i] = ss_dict[res_id]
                else:
                    ss_list[i] = "C"
            ss_pred = "".join(ss_list)  # length L
            assert len(ss_pred) == len(t.aa)
            ss_preds.append(ss_pred)
        print("Secondary structure prediction done.")
        return prot_ids, ss_preds      
    

    def compute_feats(self, dataset, config):
        tasks = {
            "disorder":   lambda: self.compute_disorder(dataset, **config["disorder"]),
            "embeddings": lambda: self.compute_embeddings(dataset, **config["embeddings"]),            
            "sec":        lambda: self.compute_sec(dataset, **config["sec"]),
            "plddt":      lambda: self.compute_plddt(dataset, **config["plddt"]),
            "fps":        lambda: self.compute_fps(dataset, **config["fps"]),
        }
        for task in tasks:
            tasks[task]()
        
        # spin up one “worker” per compute_*, all running concurrently
        # with ThreadPoolExecutor(max_workers=len(tasks)) as exec:
        #     future_to_name = {
        #         exec.submit(fn): name
        #         for name, fn in tasks.items()
        #     }
        #     for future in as_completed(future_to_name):
        #         name = future_to_name[future]
        #         try:
        #             future.result()   # will re-raise if that task errored
        #             print(f"{name} done")
        #         except Exception as e:
        #             print(f"⚠️ {name} failed: {e}")        

    @staticmethod
    def dump_feat_batch(i, batch_size, save_dir):
        batch_path = os.path.join(save_dir, f"feats_{batch_size}_{i}.pkl")
        print(f"[batch {i}] loading {batch_path!r}")
        with open(batch_path, "rb") as f:
            feats = pickle.load(f)
        for prot_id, feat in feats.items():
            out_path = os.path.join(save_dir, f"{prot_id}.pkl")
            # you can choose "xb" if you want to avoid overwriting existing files
            with open(out_path, "wb") as g:
                pickle.dump(feat, g)
        print(f"[batch {i}] done")
        return i        


    def compute_batch_feats(self, dataset, config, save_dir, batch_size):
        # compute feats in batches    
        n_batches = (len(dataset)+batch_size-1)//batch_size
        for i in range(n_batches):                            
            batch_path = os.path.join(save_dir, f"feats_{batch_size}_{i}.pkl")
            all_dumped = all([os.path.exists(os.path.join(save_dir, f"{Path(t.fname).stem}.pkl")) \
                for t in dataset[batch_size*i:batch_size*(i+1)]])
            if all_dumped:
                print(f"batch {i} already dumped")
                continue            
            elif os.path.exists(batch_path):
                print(f"batch {i} done, continuing")
                continue
            else:
                print(f"begin feat computation batch {i}")
                # check if already all dumped                
                self.compute_feats(dataset[batch_size*i:batch_size*(i+1)], config)
                print(f"begin saving feat batch {i}")
                pickle.dump(self.feats, open(batch_path, "wb+"))
                print(f"done saving feat batch {i}")
                self.feats = defaultdict(dict)
        print("all feat batches ready")
        # cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
        # cpus = min(cpus, 10)
        # ctx = mp.get_context('spawn')
        # with ProcessPoolExecutor(max_workers=cpus) as exe:
        #     # This will print each batch’s completion as they finish:
        #     for i in exe.map(SemiCRFModel.dump_feat_batch, range(n_batches), [batch_size]*n_batches, [save_dir]*n_batches):
        #         pass
        for i in range(n_batches):
            all_dumped = all([os.path.exists(os.path.join(save_dir, f"{Path(t.fname).stem}.pkl")) \
                for t in dataset[batch_size*i:batch_size*(i+1)]])
            if all_dumped:
                continue
            else:
                SemiCRFModel.dump_feat_batch(i, batch_size, save_dir)
        print("All feats dumped.")

        
    def compute_fps(self,
                    dataset,
                    grid_size=64,
                    padding=2.0,
                    order=8,
                    max_workers=20):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        args_list = [
            (t, grid_size, padding, order)
            for t in dataset
        ]
        if max_workers:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(SemiCRFModel._compute_protein_fps, args): args[0]
                    for args in args_list
                }

                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing fps"):
                    prot_id, fps_dict = fut.result()
                    self.feats[prot_id]['fp'] = fps_dict
        else:
            res = [SemiCRFModel._compute_protein_fps(args) for args in args_list]
            for prot_id, fps_dict in res:
                self.feats[prot_id]['fp'] = fps_dict                


    def compute_plddt(self,
                    dataset,
                    max_workers=0, batch_size=1000, model_batch_size=10):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        func = partial(SemiCRFModel._compute_protein_plddt, batch_size=model_batch_size)
        if max_workers:
            args_list = [dataset[i*batch_size:(i+1)*(batch_size)] for i in range((len(dataset)+batch_size-1)//batch_size)]
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(func, args): args[0]
                    for args in args_list
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing plddts"):
                    prot_ids, plddts = fut.result()
                    for i, prot_id in enumerate(prot_ids):
                        plddt = plddts[i]
                        plddt = plddt[plddt==plddt]
                        self.feats[prot_id]['plddt'] = plddt.cpu().numpy()
        else:
            prot_ids, plddts = func(dataset)
            for i, prot_id in enumerate(prot_ids):
                plddt = plddts[i]
                plddt[plddt!=plddt] = 0.
                self.feats[prot_id]['plddt'] = plddt.cpu().numpy()               


    def compute_disorder(self,
                    dataset,
                    max_workers=20, batch_size=5):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        func = partial(SemiCRFModel._compute_protein_disorder)
        if max_workers:
            args_list = [dataset[i*batch_size:(i+1)*(batch_size)] for i in range((len(dataset)+batch_size-1)//batch_size)]
            ctx = mp.get_context('spawn')            
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(func, args): args[0]
                    for args in args_list
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing disorder"):
                    prot_ids, disorders = fut.result()
                    for i, prot_id in enumerate(prot_ids):
                        self.feats[prot_id]['disorder'] = np.array(disorders[i])
        else:
            prot_ids, disorders = func(dataset)
            for i, prot_id in enumerate(prot_ids):
                self.feats[prot_id]['disorder'] = np.array(disorders[i])                


    def compute_embeddings(self,
                    dataset,
                    max_workers=0, batch_size=1000, model_batch_size=10):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        func = partial(SemiCRFModel._compute_protein_embedding, batch_size=model_batch_size)
        if max_workers:
            args_list = [dataset[i*batch_size:(i+1)*(batch_size)] for i in range((len(dataset)+batch_size-1)//batch_size)]
            ctx = mp.get_context('spawn')
            # embeddings = SemiCRFModel._compute_protein_embedding(dataset)        
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(func, args): args[0]
                    for args in args_list
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing embedding"):
                    prot_ids, embeddings = fut.result()
                    for i, prot_id in enumerate(prot_ids):
                        self.feats[prot_id]['embedding'] = embeddings[i].cpu().numpy()
        else:
            prot_ids, embeddings = func(dataset)
            for i, prot_id in enumerate(prot_ids):
                self.feats[prot_id]['embedding'] = embeddings[i].cpu().numpy()                



    def compute_sec(self,
                    dataset,
                    max_workers=20, batch_size=20):        
        if max_workers:
            args_list = [dataset[i*batch_size:(i+1)*(batch_size)] for i in range((len(dataset)+batch_size-1)//batch_size)]
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(SemiCRFModel._compute_protein_sec, args): args[0]
                    for args in args_list
                }
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing sec"):
                    prot_ids, secs = fut.result()
                    for i, prot_id in enumerate(prot_ids):
                        self.feats[prot_id]['sec'] = secs[i]
        else:
            prot_ids, secs = SemiCRFModel._compute_protein_sec(dataset)            
            for i, prot_id in enumerate(prot_ids):
                self.feats[prot_id]['sec'] = secs[i]                

    # -----------------------------------------------------------------
    # main API ---------------------------------------------------------
    # -----------------------------------------------------------------
    def forward(
        self,
        feats: Dict,
        aa_seq: str,
        angles_tensor: torch.Tensor,            # (1, L, num_features)
        coords_tensor: torch.Tensor,            # (3*L, 3)
        timestep: torch.Tensor,                 # (1,) or (1,1)
        attention_mask: torch.Tensor,           # (1, L)
        batch_size: int = 64
    ) -> List[List[torch.Tensor]]:
        """
        Returns:
            out[i][l]   scalar (log‑)score for span [i, i+l-1]
                        i in 0…L-1, l in 1…L-i
        """

        # 1) token‑level encoder
        #    -> per‑token hidden (batch, L, hidden)
        if self.encoder:
            L = angles_tensor.size(1)
            assert L == len(aa_seq), "angle tensor and sequence length differ"            
            enc_hidden = self.encoder(
                inputs        = angles_tensor.to(self.device),
                timestep      = timestep.to(self.device),
                attention_mask= attention_mask.to(self.device),
            )
            if isinstance(enc_hidden, tuple) or isinstance(enc_hidden, list):
                enc_hidden = enc_hidden[0]          # (1, L, hidden)
            token_repr = enc_hidden.squeeze(0)       # (L, hidden)
        else:
            L = len(aa_seq)
        
        # 2) pre‑compute residue‑level biochemical feature tensor (L, feat_dim)        
        plddt = feats['plddt']
        disorder = feats['disorder']
        ss_pred = feats['sec']
        embedding = feats['embedding']        
        res_feats = self.featurizer(
            aa_seq, ss_pred=ss_pred,
            disorder=disorder,
            plddt=plddt,
            embedding=embedding
        ).to(self.device)                        # (L, feat_dim)


        # 3) build out[i][l]
        out: List[List[torch.Tensor]] = [[None]*(L+1) for _ in range(L)]
        attn_out: List[List[torch.Tensor]] = [[None]*(L+1) for _ in range(L)]
        
        # cumulative sums for O(1) span pooling of token repr & res feats
        if self.encoder:
            cumsum_token = torch.cat([torch.zeros(1, token_repr.size(-1), device=self.device),
                                    token_repr.cumsum(dim=0)], dim=0)

        # a) compute bio_scores, in batches
        agg_vecs: List[torch.Tensor] = []
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                # biochemical segment potential
                span_feat = res_feats[i:j]                       # (l, feat_dim)
                agg_vec   = self.aggregator(span_feat)           # (agg_dim,)                
                agg_vecs.append(agg_vec)
                
        # bio_scores = []
        # for i in range((len(agg_vecs)+batch_size-1)//batch_size):
        #     batch = torch.stack(agg_vecs[batch_size*i:batch_size*(i+1)], axis=0)
        #     scores, _ = self.seg_mlp(batch)
        #     bio_scores
        bio_scores, attn_scores = self.seg_mlp(torch.stack(agg_vecs, axis=0))
        index = 0
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                # biochemical segment potential scores
                out[i][l] = bio_scores[index]
                attn_out[i][l] = attn_scores[index]
                index += 1
       
        # b) compute descr_scores
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l

                fp = feats['fp'][(i, j)]
                fp = torch.as_tensor(fp).to(self.device)
                descr_score = self.zernike_projector(fp)

                # total score + length bias
                out[i][l] += descr_score.squeeze()

        # c) add length bias and (if encoder) compute encoder scores
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                if self.encoder:
                    # mean token embedding for span
                    span_token_mean = (cumsum_token[j] - cumsum_token[i]) / l
                    enc_score = self.enc2score(span_token_mean).squeeze(-1)
                else:
                    enc_score = 0.0

                # total score + length bias
                out[i][l] += enc_score + self.gamma * l

        return out, attn_out

    # convenience helper identical to your old precompute signature
    def precompute(self, feats, aa_seq, angles_tensor=None, coords_tensor=None, timestep=None, attention_mask=None):
        return self.forward(feats, aa_seq, angles_tensor, coords_tensor, timestep, attention_mask)        


class SemiCRF2DModel(SemiCRFModel):
    def __init__(
        self,        
        res_featurizer: BackboneResidueFeaturizer,
        seg_aggregator: SegmentFeatureAggregator,
        seg_pair_aggregator: SegmentPairFeatureAggregator,
        seg_potential: SegmentPotentialMLP,
        seg_pair_potential: SegmentPairPotentialMLP,
        length_bias: float = 0.0,
        max_seg_len: int = 100,
        device: Optional[torch.device] = None
    ):
        super().__init__(
            res_featurizer=res_featurizer, 
            seg_aggregator=seg_aggregator,
            seg_potential=seg_potential,
            length_bias=length_bias,
            max_seg_len=max_seg_len,
            device=device
        )
        self.foldseek_projector = nn.Linear(22, 1)
        self.seg_pair_mlp    = seg_pair_potential        
        self.seg_pair_aggregator = seg_pair_aggregator


    def compute_feats(self, dataset, config):    
        tasks = {
            "disorder":   lambda: self.compute_disorder(dataset, **config["disorder"]),
            "embeddings": lambda: self.compute_embeddings(dataset, **config["embeddings"]),            
            "sec":        lambda: self.compute_sec(dataset, **config["sec"]),
            "plddt":      lambda: self.compute_plddt(dataset, **config["plddt"]),
            "fps":        lambda: self.compute_fps(dataset, **config["fps"]),
            "foldseek":   lambda: self.compute_foldseek(dataset, self.max_seg_len, **config["foldseek"])
        }
        
        for task in tasks:
            tasks[task]()
        # spin up one “worker” per compute_*, all running concurrently
        # with ThreadPoolExecutor(max_workers=len(tasks)) as exec:
        #     future_to_name = {
        #         exec.submit(fn): name
        #         for name, fn in tasks.items()
        #     }
        #     for future in as_completed(future_to_name):
        #         name = future_to_name[future]
        #         try:
        #             future.result()   # will re-raise if that task errored
        #             print(f"{name} done")
        #         except Exception as e:
        #             print(f"⚠️ {name} failed: {e}")



    @staticmethod
    # Top‐level helper must be importable (no self):
    def _compute_protein_foldseeks(args):
        """
        Compute all (i, l1, l2) foldseeks for a single protein t.
        Returns (prot_id, { (i,l1,l2): torch.Tensor(foldseek) }).
        """
        fname, aa_seq, coords, beta_c, max_seg_len = args
        prot_id = Path(fname).stem
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        L = len(aa_seq)
        foldseeks_dict = {}
        for i in range(L):
            max_l2 = min(max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(max_seg_len, i)
                for l1 in range(1, max_l1 + 1):                    
                    c = coords_np[3*(i-l1): 3*(i+l2)]                    
                    cb = beta_c[i-l1:i+l2]
                    n, ca, c = c[0::3], c[1::3], c[2::3]
                    feats, mask, _ = structure2features(ca, n, c, cb)
                    foldseeks_dict[(i, l1, l2)] = feats

        return prot_id, foldseeks_dict        


    def compute_foldseek(self,
                         dataset,
                         max_L,
                         max_workers=20
    ):
        """
        Parallelize at the protein level.  Uses one process per protein
        and shows a tqdm bar as each finishes.
        """
        args_list = [
            (t.fname, t.aa, t.compute_coords(), t.beta_coords, max_L)
            for t in dataset
        ]
        if max_workers:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(SemiCRF2DModel._compute_protein_foldseeks, args): args[0]
                    for args in args_list
                }

                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Computing foldseeks"):
                    prot_id, foldseeks_dict = fut.result()
                    self.feats[prot_id]['foldseek'] = foldseeks_dict
        else:
            res = [SemiCRF2DModel._compute_protein_foldseeks(args) for args in args_list]
            for prot_id, foldseeks_dict in res:
                self.feats[prot_id]['foldseek'] = foldseeks_dict     
    
    # -----------------------------------------------------------------
    # main API ---------------------------------------------------------
    # -----------------------------------------------------------------
    def forward(
        self,
        feats: Dict,
        aa_seq: str
    ) -> Tuple[
        torch.Tensor,                              # unary_out, shape (L, max_seg_len+1)
        List[List[torch.Tensor]],                  # unary_attn_out, shape (L, max_seg_len+1)
        torch.Tensor,                              # edge_out, shape (L, max_seg_len+1, max_seg_len+1)
        List[List[List[torch.Tensor]]]             # edge_attn_out, shape (L, max_seg_len+1, max_seg_len+1)
    ]:
        """
        Compute unary and pairwise segment scores for a protein sequence.
        Args:
            feats (Dict): Dictionary of per-protein features, including 'plddt', 'disorder', 'sec', 'embedding', 'fp', 'foldseek'.
            aa_seq (str): Amino acid sequence string for the protein.

        Returns:
            unary_out (torch.Tensor): Segment unary scores, shape (L, max_seg_len+1), where unary_out[i][l] is the score for span [i, i+l-1].
            unary_attn_out (List[List[torch.Tensor]]): Unary attention weights per span [[attn_vec for l in 0..max_seg_len] for i in 0..L-1].
            edge_out (torch.Tensor): Pairwise segment scores, shape (L, max_seg_len+1, max_seg_len+1), where edge_out[i][l1][l2] scores split at i.
            edge_attn_out (List[List[List[torch.Tensor]]]): Pairwise attention weights per span/split [[[attn_vec for l2] for l1] for i].
        """
        # 1) token‑level encoder
        #    -> per‑token hidden (batch, L, hidden)
        L = len(aa_seq)
        
        # 2) pre‑compute residue‑level biochemical feature tensor (L, feat_dim)        
        plddt = feats['plddt']
        disorder = feats['disorder']
        ss_pred = feats['sec']
        embedding = feats['embedding']        
        res_feats = self.featurizer(
            aa_seq, ss_pred=ss_pred,
            disorder=disorder,
            plddt=plddt,
            embedding=embedding
        ).to(self.device)                        # (L, feat_dim)

        # 3) build unary_out[i][l] and edge_out[i][l1][l2]
        unary_out: torch.Tensor = torch.tensor([[0.0]*(self.max_seg_len+1) for _ in range(L)], device=self.device)
        unary_attn_out: List[List[torch.Tensor]] = [[0.0]*(self.max_seg_len+1) for _ in range(L)]
        edge_out: torch.Tensor = torch.tensor([[[0.0]*(self.max_seg_len+1)]*(self.max_seg_len+1) for _ in range(L)], device=self.device)
        edge_attn_out: List[List[List[torch.Tensor]]] = [[[0.0]*(self.max_seg_len+1)]*(self.max_seg_len+1) for _ in range(L)]
        
        # a) compute bio_scores, in batches
        agg_vecs: List[torch.Tensor] = []
        agg_vec_pairs: List[List[torch.Tensor]] = []
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                # biochemical segment potential
                span_feat = res_feats[i:j]                       # (l, feat_dim)
                agg_vec   = self.aggregator(span_feat)           # (agg_dim,)                
                agg_vecs.append(agg_vec)
        
        for i in range(L):
            max_l2 = min(self.max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(self.max_seg_len, i)
                for l1 in range(1, max_l1 + 1):
                    # biochemical segment potential
                    span_feat1 = self.aggregator(res_feats[i-l1: i])
                    span_feat2 = self.aggregator(res_feats[i: i+l2])
                    pair_span_feat = torch.cat((span_feat1, span_feat2))
                    agg_vec_pairs.append(pair_span_feat)

        bio_scores, attn_scores = self.seg_mlp(torch.stack(agg_vecs, axis=0))
        bio_pair_scores, attn_pair_scores = self.seg_pair_mlp(torch.stack(agg_vec_pairs, axis=0))

        index = 0
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                # biochemical segment potential scores
                unary_out[i][l] = bio_scores[index]
                unary_attn_out[i][l] = attn_scores[index]
                index += 1

        index = 0
        for i in range(L):
            max_l2 = min(self.max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(self.max_seg_len, i)
                for l1 in range(1, max_l1 + 1):
                    # biochemical segment potential scores
                    edge_out[i][l1][l2] = bio_pair_scores[index]
                    edge_attn_out[i][l1][l2] = attn_pair_scores[index]
                    index += 1
       
        # b) compute descr_scores
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                j = i + l
                fp = feats['fp'][(i, j)]
                fp = torch.as_tensor(fp, dtype=torch.float32).to(self.device)
                descr_score = self.zernike_projector(fp)
                # total score + length bias
                unary_out[i][l] += descr_score.squeeze()

        for i in range(L):
            max_l2 = min(self.max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(self.max_seg_len, i)
                for l1 in range(1, max_l1 + 1):
                    foldseek = feats['foldseek'][(i, l1, l2)]                    
                    foldseek = torch.as_tensor(foldseek, dtype=torch.float32).to(self.device)
                    foldseek = self.seg_pair_aggregator(foldseek)
                    descr_score = self.foldseek_projector(foldseek)
                    edge_out[i][l1][l2] += descr_score.squeeze()

        # c) add length bias
        for i in range(L):
            max_l = min(self.max_seg_len, L - i)
            for l in range(1, max_l + 1):
                unary_out[i][l] += self.gamma * l

        for i in range(L):
            max_l2 = min(self.max_seg_len, L - i)
            for l2 in range(1, max_l2 + 1):
                max_l1 = min(self.max_seg_len, i)
                for l1 in range(1, max_l1 + 1):
                    edge_out[i][l1][l2] += self.gamma * (l1+l2)

        return unary_out, unary_attn_out, edge_out, edge_attn_out


    # convenience helper identical to your old precompute signature
    def precompute(self, feats, aa_seq):
        return self.forward(feats, aa_seq)   
    

def zero_initialize(self):
    # Loop over all named parameters and zero them
    for name, param in self.named_parameters():
        nn.init.constant_(param, 0.0)

def l1_penalty(self):
    """
    Compute the sum of absolute values of all parameters in this module.
    """
    l1_sum = 0.0
    for param in self.parameters():
        l1_sum += param.abs().sum()
    return l1_sum        




    
class AngleTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=8, num_layers=2, dropout=0.1, max_len=5000):
        """
        A lightweight Transformer-based model to encode a sequence of dihedral angles
        into a single scalar.
        
        Parameters:
            input_size (int): Dimensionality of each input token (1 if each angle is scalar).
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
        """
        super(AngleTransformer, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)  # project scalar to d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final dense layer mapping the pooled representation to a scalar.
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x, lengths):
        """
        Parameters:
            x (torch.Tensor): Padded tensor of shape (batch, max_seq_len, input_size).
            lengths (list or torch.Tensor): Actual lengths for each sequence in the batch.
            
        Returns:
            output (torch.Tensor): Tensor of shape (batch, 1) containing a scalar per sequence.
        """
        batch, seq_len, _ = x.size()
        # Create key padding mask: True for padded positions.
        # Assume that padding is at the end of each sequence.
        mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=x.device)
        for i, L in enumerate(lengths):
            if L < seq_len:
                mask[i, L:] = True
        
        # Project input and add positional encoding.
        x = self.input_linear(x)            # shape: (batch, seq_len, d_model)
        x = self.pos_encoder(x)             # shape: (batch, seq_len, d_model)
        
        # Transformer expects shape (seq_len, batch, d_model)
        x = x.transpose(0, 1)               # shape: (seq_len, batch, d_model)
        # Process through Transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)  # shape: (seq_len, batch, d_model)
        encoded = encoded.transpose(0, 1)   # shape: (batch, seq_len, d_model)
        return encoded
        # # Pooling: Mean over non-padded timesteps for each sequence.
        # pooled = []
        # for i, L in enumerate(lengths):
        #     # Take the mean over the first L timesteps
        #     pooled.append(encoded[i, :L].mean(dim=0))
        # pooled = torch.stack(pooled, dim=0)  # shape: (batch, d_model)
        
        # # Final dense layer to produce a single scalar per sequence.
        # output = self.fc(pooled)            # shape: (batch, 1)
        # return output
