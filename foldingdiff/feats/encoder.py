import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
from .utils import *

class BackboneResidueFeaturizer(nn.Module):
    """
    Produce a tensor of shape (seq_len, feat_dim) for a single protein chain.
    """
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.aa_to_idx = {aa:i for i,aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        out_dim = 20 + 1 # AA one‑hot + hydropathy
        labels = [f"aa_one_hot_{i}" for i in range(20)] + ["hydropathy"]
        if config["disorder"]["enabled"]:
            out_dim += 1 # disorder
            labels += ["disorder"]
        if config["sec"]["enabled"]:
            out_dim += 3 # 3‑state SS
            labels += [f"ss_{i}" for i in range(3)]
        if config["disorder"]["enabled"]:
            out_dim += 1 # pLDDT
            labels += ["plddt"]
        if config["embeddings"]["enabled"]:
            out_dim += 10 # embedding
            labels += [f"embedding_{i}" for i in range(10)]
            self.embedding_projector = nn.Linear(2560, 10)
        self.out_dim = out_dim
        self.labels = labels
        self.device = device
        # self.one_hot_projector = nn.Linear(20, 20)
        # self.hydropathy_projector = nn.Linear(1, 10)        

    def forward(self, aa_seq:str,
                      ss_pred:str=None,          # secondary structure 'H/E/C' string from e.g. PSIPRED
                      disorder:np.ndarray=None,  # per‑res residue disorder score [0,1]
                      plddt:np.ndarray=None,
                      embedding:np.ndarray=None):    # AlphaFold confidence

        L = len(aa_seq)
        device = self.device
        one_hot = torch.zeros(L, 20).to(device)
        hydro   = torch.zeros(L, 1).to(device)
        for i, aa in enumerate(aa_seq):
            if aa in self.aa_to_idx:
                one_hot[i, self.aa_to_idx[aa]] = 1.0
            hydro[i,0] = HYDROPATHY.get(aa, 0.0)
        opt_feats = []
        # Disorder
        if disorder is not None:
            disorder = torch.from_numpy(disorder).float().unsqueeze(-1)
            disorder = disorder.to(device)
            opt_feats.append(disorder)
        # Secondary structure to one‑hot
        if ss_pred is not None:
            ss_map = {"H":0, "E":1, "C":2}
            ss_onehot = torch.zeros(L, 3).to(device)
            for i, s in enumerate(ss_pred):
                ss_onehot[i, ss_map.get(s,"C")] = 1.0
            opt_feats.append(ss_onehot)
        # PLDDT
        if plddt is not None:
            plddt = torch.from_numpy(plddt).float().unsqueeze(-1) / 100.0  # scale 0…1
            plddt = plddt.to(device)
            opt_feats.append(plddt)
        if embedding is not None:
            embedding = torch.from_numpy(embedding).float().to(device)
            embedding = self.embedding_projector(embedding)
            opt_feats.append(embedding)
        return torch.cat([one_hot, hydro] + opt_feats, dim=-1)


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

