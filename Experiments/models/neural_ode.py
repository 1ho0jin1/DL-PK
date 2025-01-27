import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    Defines the ODE function for the dynamics.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in the ODE function.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0.5)

    def forward(self, t, x):
        """
        Args:
            t (torch.Tensor): Time steps.
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Derivative of x with respect to t.
        """
        return self.net(x)


class Encoder(nn.Module):
    """
    Encodes the input sequence into two parts (mean, std) for a latent representation.
    Args:
        input_dim (int): Number of input features per time step.
        meta_dim (int): Number of meta features to concatenate.
        hidden_dim (int): Number of hidden units in the GRU.
        hidden_dim (int): Dimension of the latent space.
        device (torch.device): Device for computation (default: CPU).
    """
    def __init__(self, input_dim, meta_dim, hidden_dim, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim+meta_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim))  # 2*hidden_dim -> [mean, std]

    def forward(self, x, meta=None):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, input_dim).
            meta (torch.Tensor): Shape (batch_size, meta_dim) or None.
        Returns:
            torch.Tensor: Model output of shape (batch_size, output_dim).
        """
        # append meta data to input
        if meta is not None:
            meta = torch.tile(meta.unsqueeze(1), (1,x.shape[1],1))
            x = torch.cat((x, meta), dim=-1)

        _, h = self.gru(x)  # h: (1, batch_size, hidden_dim)
        h = h.squeeze(0)  # Remove the first dimension
        out = self.fc(h)  # shape: (batch_size, 2 * hidden_dim)
        mean, std = out[:, :self.hidden_dim], out[:, self.hidden_dim:]
        return mean, std


class NeuralODE(nn.Module):
    """
    Full Neural-ODE model combining encoder, ODE function, and a linear layer.

    Args:
        input_dim (int): Number of input features per time step.
        meta_dim (int): Number of meta features to concatenate.
        hidden_dim (int): Number of hidden units in the encoder and ODE function.
        output_dim (int): Number of output features.
    """
    def __init__(self, input_dim, meta_dim, hidden_dim, output_dim=1, tol=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize the encoder, ODE function, and FC layer
        self.encoder = Encoder(input_dim, meta_dim, hidden_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + meta_dim, output_dim)

        # tolerance for the ODE solver
        self.tol = tol
    
    def sample_standard_gaussian(self, mean, std, device):
        d = torch.distributions.normal.Normal(
                torch.Tensor([0.]).to(device),
                torch.Tensor([1.]).to(device))
        r = d.sample(mean.size()).squeeze(-1)
        return r * std.float() + mean.float()

    def forward(self, x, meta=None):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, input_dim), input_dim: [TIME, TAD, AMT, DV]
            meta (torch.Tensor): Shape (batch_size, meta_dim) or None, meta_dim: [SEX, AGE, WT, Cr]
        Returns:
            torch.Tensor: Model output of shape (batch_size, output_dim).
        """
        B, N = x.shape[:2]  # B: batch size, N: sequence length
        device = x.device
        times = x[:, :, 0]
        doses = x[:, :, 2]
        
        # 1) Encode input into mean/log_var for the initial latent distribution
        qz0_mean, qz0_var = self.encoder(x, meta)

        # 2) Sample initial latent z0
        z0 = self.sample_standard_gaussian(qz0_mean, qz0_var, device)

        # 3) Solve ODE for each time step
        # NOTE: odeint cannot handle batched inputs, so we loop over the batch dimension
        solves = torch.zeros((B,N,self.hidden_dim), device=x.device)
        for b in range(B):
            z0_ = z0[b]
            time_ = times[b]
            dose_ = doses[b]
            solves_ = z0_.unsqueeze(0).clone()  # trajectory of the ODE solution with initial value z0
            for idx, (time0, time1) in enumerate(zip(time_[:-1], time_[1:])):
                z0_ = z0_ + dose_[idx]
                time_interval = torch.Tensor([time0 - time0, time1 - time0]).to(device)
                sol = odeint(self.ode_func, z0_, time_interval, rtol=self.tol, atol=self.tol)
                z0_ = sol[-1].clone()
                solves_ = torch.cat([solves_, sol[-1:, :]], 0)
            solves[b] = solves_

        # simply use the solution of the last timestep as input
        latent = torch.cat((solves[:,-1,:], meta), dim=-1)
        pred = self.fc(latent)
        return pred



if __name__ == "__main__":
    import sys
    sys.path.append('/home/hj/DL-PK/Experiments')
    from dataloader import *
    # load data and create dataloaders
    train_trfm = transforms.Compose([
        ConsecutiveSampling(24+24),
        PKPreprocess(),
    ])
    train_data = PKDataset('/home/hj/DL-PK/Experiments/dataset/train', transform=train_trfm)
    train_loader = DataLoader(train_data, batch_size=7, shuffle=True)
    batch = next(iter(train_loader))
    data = batch['data']
    meta = batch['meta']
    print(data.shape, meta.shape)
    
    
    # create model
    model = NeuralODE(input_dim=4, meta_dim=4, hidden_dim=32, output_dim=1)

    with torch.no_grad():
        output = model(data, meta)
    print(output.shape)
    
    print()