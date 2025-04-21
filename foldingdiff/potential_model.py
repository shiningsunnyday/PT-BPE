import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
import numpy as np

from Bio.SeqUtils.ProtParam import ProteinAnalysis

HYDROPATHY = {
    # Kyte–Doolittle index
    "A": 1.8, "C": 2.5, "D":-3.5, "E":-3.5, "F": 2.8, "G":-0.4,
    "H":-3.2, "I": 4.5, "K":-3.9, "L": 3.8, "M": 1.9, "N":-3.5,
    "P":-1.6, "Q":-3.5, "R":-4.5, "S":-0.8, "T":-0.7, "V": 4.2,
    "W":-0.9, "Y":-1.3, "X": 0.0, "-": 0.0,
}

class BackboneResidueFeaturizer(nn.Module):
    """
    Produce a tensor of shape (seq_len, feat_dim) for a single protein chain.
    """
    def __init__(self):
        super().__init__()
        self.aa_to_idx = {aa:i for i,aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.out_dim = 20 + 1 + 1 + 3 + 1          # AA one‑hot + hydropathy + disorder + 3‑state SS + pLDDT

    def forward(self, aa_seq:str,
                      ss_pred:str=None,          # secondary structure 'H/E/C' string from e.g. PSIPRED
                      disorder:np.ndarray=None,  # per‑res residue disorder score [0,1]
                      plddt:np.ndarray=None):    # AlphaFold confidence

        L = len(aa_seq)
        one_hot = torch.zeros(L, 20)
        hydro   = torch.zeros(L, 1)
        for i, aa in enumerate(aa_seq):
            if aa in self.aa_to_idx:
                one_hot[i, self.aa_to_idx[aa]] = 1.0
            hydro[i,0] = HYDROPATHY.get(aa, 0.0)

        # Disorder (fallback 0.0)
        if disorder is None: disorder = np.zeros(L)
        disorder = torch.from_numpy(disorder).float().unsqueeze(-1)

        # Secondary structure to one‑hot
        if ss_pred is None: ss_pred = "C"*L
        ss_map = {"H":0, "E":1, "C":2}
        ss_onehot = torch.zeros(L, 3)
        for i, s in enumerate(ss_pred):
            ss_onehot[i, ss_map.get(s,"C")] = 1.0

        if plddt is None: plddt = np.zeros(L)
        plddt = torch.from_numpy(plddt).float().unsqueeze(-1) / 100.0  # scale 0…1

        return torch.cat([one_hot, hydro, disorder, ss_onehot, plddt], dim=-1)


class SegmentFeatureAggregator(nn.Module):
    """
    Turn a span X[i:j] (tensor (len, feat_dim)) into a fixed vector.
    We concatenate:
      – mean of each feature over the span
      – variance
      – length (normalised)
      – log‑length
    """
    def __init__(self, per_res_dim:int):
        super().__init__()
        self.per_res_dim = per_res_dim
        self.out_dim = 2*per_res_dim + 2  # mean + var + len + log(len)

    def forward(self, span:torch.Tensor):
        # span: (L, per_res_dim)
        mean = span.mean(dim=0)
        var  = span.var(dim=0, unbiased=False)
        L = span.size(0)
        length = torch.tensor([L], dtype=span.dtype, device=span.device)
        log_length = torch.log(length.float()+1e-6)
        return torch.cat([mean, var, length/500.0, log_length/6.0], dim=0)


class SegmentPotentialMLP(nn.Module):
    """
    Small 2‑layer MLP: agg_vec -> hidden -> scalar
    """
    def __init__(self, agg_dim:int, hidden:int=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(agg_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, agg_vec:torch.Tensor):
        return self.net(agg_vec).squeeze(-1)      # scalar


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
        device: Optional[torch.device] = None,
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

    # -----------------------------------------------------------------
    # main API ---------------------------------------------------------
    # -----------------------------------------------------------------
    def forward(
        self,
        aa_seq: str,
        angles_tensor: torch.Tensor,            # (1, L, num_features)
        timestep: torch.Tensor,                 # (1,) or (1,1)
        attention_mask: torch.Tensor,           # (1, L)
        ss_pred: Optional[str] = None,
        disorder: Optional[torch.Tensor] = None,
        plddt: Optional[torch.Tensor] = None,
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
        res_feats = self.featurizer(
            aa_seq, ss_pred=ss_pred,
            disorder=None if disorder is None else disorder.cpu().numpy(),
            plddt=None if plddt is None else plddt.cpu().numpy(),
        ).to(self.device)                        # (L, feat_dim)

        # 3) build out[i][l]
        out: List[List[torch.Tensor]] = [[None]*(L+1) for _ in range(L)]

        # cumulative sums for O(1) span pooling of token repr & res feats
        if self.encoder:
            cumsum_token = torch.cat([torch.zeros(1, token_repr.size(-1), device=self.device),
                                    token_repr.cumsum(dim=0)], dim=0)
        cumsum_feat  = torch.cat([torch.zeros(1, res_feats.size(-1), device=self.device),
                                  res_feats.cumsum(dim=0)], dim=0)

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

                # biochemical segment potential
                span_feat = res_feats[i:j]                       # (l, feat_dim)
                agg_vec   = self.aggregator(span_feat)           # (agg_dim,)
                bio_score = self.seg_mlp(agg_vec)

                # total score + length bias
                out[i][l] = enc_score + bio_score + self.gamma * l

        return out

    # convenience helper identical to your old precompute signature
    def precompute(self, aa_seq, angles_tensor=None, timestep=None, attention_mask=None, ss_pred=None, disorder=None, plddt=None):
        out = self.forward(aa_seq, angles_tensor, timestep, attention_mask,
                           ss_pred=ss_pred, disorder=disorder, plddt=plddt)
        return out


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
