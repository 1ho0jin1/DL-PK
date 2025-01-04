import torch
import torch.nn as nn

class GRUPK(nn.Module):
    """
    A simple GRU-based model for PK data (time-series).

    Args:
        input_dim (int): Number of input features per time step (e.g., 3 if [TIME, AMT, DV]).
        hidden_dim (int): Number of hidden units in the GRU.
        num_layers (int): Number of stacked GRU layers.
        output_dim (int): Number of outputs (e.g., 1 for a single continuous prediction).
        bidirectional (bool): If True, use a bidirectional GRU.
    """
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=1, output_dim=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, output_dim)

    def forward(self, x):
        """
        Forward pass through the GRU model.
        
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, input_dim).
        
        Returns:
            torch.Tensor: Model output of shape (batch_size, output_dim).
        """
        # GRU returns (output, h_n)
        # output shape: (batch_size, seq_len, hidden_dim * num_directions)
        out, h = self.gru(x)

        # Taking the last time-step from the GRU output
        last_out = out[:, -1, :]

        # Alternatively, for single-direction single-layer, we could use h[-1, :, :].
        # For multi-layer, or bidirectional, you'd extract the relevant hidden states.

        pred = self.fc(last_out)  # shape: (batch_size, output_dim)
        return pred
