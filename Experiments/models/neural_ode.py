import torch
import torch.nn as nn
import torchode as to



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
            t (torch.Tensor): Time for current step
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Derivative of x with respect to t.
        """
        dxdt = self.net(x)
        return dxdt


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
    def __init__(self, input_dim, meta_dim, hidden_dim, output_dim=1, atol=1e-6, rtol=1e-3):
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
        self.atol = atol
        self.rtol = rtol
        
        # Initialize jit solver for torchode
        self.term = to.ODETerm(self.ode_func)
        self.step_method = to.Dopri5(term=self.term)
        self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.term)
        self.solver = to.AutoDiffAdjoint(self.step_method, self.step_size_controller)
        self.jit_solver = torch.compile(self.solver)
        # Register gradient logging hooks for all parameters
        self.register_gradient_hooks()

    def register_gradient_hooks(self):
        # Helper to create a hook that logs the gradient norm.
        def get_hook(name):
            def hook(grad):
                # You can replace print with logging to a file if needed.
                print(f"Gradient norm for {name}: {grad.norm().item():.6f}")
                return grad
            return hook

        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(get_hook(name))
    
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
        y0 = self.sample_standard_gaussian(qz0_mean, qz0_var, device)

        # 3) Solve ODE for each time step
        sol = self.jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=times))
        # sol = self.jit_solver.solve(to.InitialValueProblem(y0=y0, t_start=times[:,0], t_end=times[:,-1]))

        # simply use the solution of the last timestep as input
        latent = torch.cat((sol.ys[:,-1,:], meta), dim=-1)
        pred = self.fc(latent)
        return pred



if __name__ == "__main__":
    import sys
    sys.path.append('/home/hj/DL-PK/Experiments')
    from utils.general import set_random_seed
    from dataloader import *
    set_random_seed(2025)
    # load data and create dataloaders
    train_trfm = transforms.Compose([
        ConsecutiveSampling(16),
        PKPreprocess(),
    ])
    train_data = PKDataset('/home/hj/DL-PK/Experiments/dataset/train', transform=train_trfm)
    train_loader = DataLoader(train_data, batch_size=7, shuffle=False)
    batch = next(iter(train_loader))
    data = batch['data']
    meta = batch['meta']
    print(data.shape, meta.shape)
    
    
    # create model
    model = NeuralODE(input_dim=4, meta_dim=4, hidden_dim=32, output_dim=1)

    with torch.no_grad():
        output = model(data, meta)
    print(output.shape)
    print(output)
    
    print()