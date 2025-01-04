import torch
import torch.nn as nn



class LSTMPK(nn.Module):
    """
    A simple LSTM-based model for PK data (time-series).

    Args:
        input_dim (int): Number of input features per time step (e.g., 3 if [TIME, AMT, DV]).
        hidden_dim (int): Number of hidden units in the LSTM.
        num_layers (int): Number of stacked LSTM layers.
        output_dim (int): Number of outputs (e.g., 1 for a single continuous prediction).
        bidirectional (bool): If True, use a bidirectional LSTM.
    """
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=1, output_dim=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # If bidirectional, the final LSTM output size is hidden_dim * 2
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, output_dim)

    def forward(self, x, aux=None):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, input_dim).
            aux (torch.Tensor): Shape (batch_size, aux_dim) or None.
        Returns:
            torch.Tensor: Model output of shape (batch_size, output_dim).
        """
        # LSTM returns (output, (h_n, c_n))
        # output shape: (batch_size, seq_len, hidden_dim * num_directions)
        out, (h, c) = self.lstm(x)

        # Option 1: Take the LAST time-step from LSTM output
        #           out[:, -1, :] has shape (batch_size, hidden_dim * num_directions)
        last_out = out[:, -1, :]

        # Option 2 (sometimes used): Use the final hidden state h_n
        # But for multi-layer, we typically take h_n at the last layer (and handle directions).
        # For a single-direction, single-layer: h[-1, :, :] has shape (batch_size, hidden_dim).
        # For bidirectional: you might need to concat h[-2,:,:] and h[-1,:,:].

        # We'll stick with last_out from the sequence.
        pred = self.fc(last_out)  # shape: (batch_size, output_dim)
        return pred