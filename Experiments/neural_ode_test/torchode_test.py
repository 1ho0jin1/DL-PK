import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchode as to
from torchdiffeq import odeint

# def f(t, y):
#     return -0.5 * y

# Define the vector field using a neural network
class VectorField(nn.Module):
    def __init__(self):
        super(VectorField, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, t, y):
        return self.net(y)
    
with torch.no_grad():
    f = VectorField()
    y0 = torch.tensor([[1.2], [5.0]])
    n_steps = 10
    t_eval = torch.stack((torch.linspace(0, 5, n_steps), torch.linspace(3, 4, n_steps)))

    term = to.ODETerm(f)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    jit_solver = torch.compile(solver)

    sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
print(sol.stats)
# => {'n_f_evals': tensor([26, 26]), 'n_steps': tensor([4, 2]),
# =>  'n_accepted': tensor([4, 2]), 'n_initialized': tensor([10, 10])}

plt.plot(sol.ts[0], sol.ys[0])
plt.plot(sol.ts[1], sol.ys[1])
plt.savefig('torchode-test.jpg', bbox_inches='tight')


solves = torch.FloatTensor([])
with torch.no_grad():
    for i in range(2):
        y0_b = y0[i]
        t_eval_b = t_eval[i]
        solves_b = odeint(f, y0_b, t_eval_b, atol=1e-6, rtol=1e-3).squeeze().unsqueeze(0)
        solves = torch.cat((solves, solves_b), dim=0)
plt.plot(t_eval[0], solves[0])
plt.plot(t_eval[1], solves[1])
plt.savefig('torchdiffeq-test.jpg', bbox_inches='tight')

diff = (solves - sol.ys.squeeze()).abs()
print("Absolute difference:", diff)