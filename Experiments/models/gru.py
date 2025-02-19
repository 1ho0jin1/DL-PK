import torch
import torch.nn as nn

class GRUPK(nn.Module):
    """
    A simple GRU-based model for PK data (time-series).

    Args:
        input_dim (int): Number of input features per time step (e.g., 4 if [TIME, TAD, AMT, DV]).
        hidden_dim (int): Number of hidden units in the GRU.
        num_layers (int): Number of stacked GRU layers.
        output_dim (int): Number of outputs (e.g., 1 for a single continuous prediction).
        meta_dim (int): Number of meta data dimension.
        bidirectional (bool): If True, use a bidirectional GRU.
    """
    def __init__(self, input_dim=4, meta_dim=4, hidden_dim=32, num_layers=1, output_dim=1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_dim+meta_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, output_dim)

    def forward(self, x, meta=None):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, input_dim).
            meta (torch.Tensor): Shape (batch_size, meta_dim) or None.
        Returns:
            torch.Tensor: Model output of shape (batch_size, output_dim).
        """
        # GRU returns (output, h_n)
        # output shape: (batch_size, seq_len, hidden_dim * num_directions)
        
        # append meta data to input
        if meta is not None:
            meta = torch.tile(meta.unsqueeze(1), (1,x.shape[1],1))
            x = torch.cat((x, meta), dim=-1)
        
        out, h = self.gru(x)

        # Taking the last time-step from the GRU output
        last_out = out[:, -1, :]

        # Alternatively, for single-direction single-layer, we could use h[-1, :, :].
        # For multi-layer, or bidirectional, you'd extract the relevant hidden states.

        pred = self.fc(last_out)  # shape: (batch_size, output_dim)
        return pred




if __name__ == "__main__":
    import sys
    sys.path.append('/home/hj/DL-PK/Experiments')
    from dataloader import *
    # load data and create dataloaders
    train_trfm = transforms.Compose([
        ConsecutiveSampling(24+24),
        Normalize(),
    ])
    train_data = PKDataset('/home/hj/DL-PK/Experiments/dataset/train', transform=train_trfm)
    train_loader = DataLoader(train_data, batch_size=7, shuffle=True)
    batch = next(iter(train_loader))
    data = batch['data']
    meta = batch['meta']
    print(data.shape, meta.shape)
    
    
    # create model
    model = GRUPK(input_dim=4, meta_dim=4, hidden_dim=32, num_layers=1, output_dim=1, bidirectional=False)

    with torch.no_grad():
        output = model(data, meta)
    print(output.shape)
    
    print()