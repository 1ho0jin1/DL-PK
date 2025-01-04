import torch
import torch.nn as nn

class TransformerPK(nn.Module):
    """
    A simple Transformer Encoder-based model for PK data (time-series),
    with a learnable positional encoding.
    Args:
            input_dim (int): Number of features per time step 
                             (e.g., 3 if [TIME, AMT, DV]).
            d_model (int): Dimensionality of the Transformer embeddings.
            nhead (int): Number of attention heads in each Transformer layer.
            num_layers (int): Number of TransformerEncoder layers.
            dim_feedforward (int): Dim of the feedforward network within each layer.
            output_dim (int): Final output dimension (e.g., 1 for regression).
            max_len (int): Max sequence length for the positional embeddings.
    """

    def __init__(
        self,
        input_dim=3,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        output_dim=1,
        max_len=5000
    ):
        super().__init__()

        # 1) Embed input_dim -> d_model
        self.embedding = nn.Linear(input_dim, d_model)

        # 2) Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len=max_len)

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='relu',
            batch_first=True  # Important if input is (batch, seq, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) Final projection to output_dim
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x, aux=None):
        """
        Args:
            x (torch.Tensor): shape (batch_size, seq_len, input_dim)
            aux (torch.Tensor): Shape (batch_size, aux_dim) or None.
        Returns:
            torch.Tensor: shape (batch_size, output_dim)
        """
        # (1) Project to d_model
        x_emb = self.embedding(x)  # shape: (batch_size, seq_len, d_model)

        # (2) Add positional encoding
        x_emb = self.pos_encoding(x_emb)  # (batch_size, seq_len, d_model)

        # (3) Pass through Transformer Encoder
        #     out shape: (batch_size, seq_len, d_model)
        out = self.transformer_encoder(x_emb)

        # (4) Take the last time-step or use pooling
        last_out = out[:, -1, :]  # shape (batch_size, d_model)

        # (5) Final linear layer
        pred = self.fc(last_out)  # shape (batch_size, output_dim)

        return pred



class LearnablePositionalEncoding(nn.Module):
    """
    Adds a learnable position embedding to the input embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): The embedding dimension (must match the Transformer d_model).
            max_len (int): Maximum sequence length you expect.
        """
        super().__init__()
        # We will embed positions [0..max_len-1] in a trainable matrix of shape (max_len, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        # Optionally, you could initialize it in some manner; by default it's random uniform.

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: shape (batch_size, seq_len, d_model) with positional encodings added
        """
        batch_size, seq_len, _ = x.size()

        # Create a tensor of positions [0, 1, 2, ..., seq_len-1], on the same device as x
        positions = torch.arange(seq_len, device=x.device).long()

        # Get position embeddings: shape (seq_len, d_model)
        pos_emb = self.pos_embedding(positions)

        # Expand to (1, seq_len, d_model) so it can be broadcast to all batch elements
        pos_emb = pos_emb.unsqueeze(0)  # shape: (1, seq_len, d_model)

        # Add to the input embeddings
        return x + pos_emb