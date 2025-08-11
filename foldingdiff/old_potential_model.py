
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
