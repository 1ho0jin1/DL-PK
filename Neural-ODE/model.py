import os
import torch
import torch.nn as nn
from utils.general import init_network_weights, reverse



class ODEFunc(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0.5)

    def forward(self, t, x):
        # print(x)
        return self.net(x)


class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, device=torch.device("cpu")):
        super(Encoder, self).__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.hiddens_to_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        init_network_weights(self.hiddens_to_output, std=0.001)

        # self.rnn = nn.RNN(self.input_dim, self.hidden_dim, nonlinearity="relu").to(device)
        self.rnn = nn.GRU(self.input_dim, self.hidden_dim).to(device)

    def forward(self, data):
        data = data.permute(1, 0, 2)
        data = reverse(data)
        output_rnn, _ = self.rnn(data)
        #print(output_rnn)
        outputs = self.hiddens_to_output(output_rnn[-1])
        #print(outputs)
        
        return outputs


class Classifier(nn.Module):

    def __init__(self, latent_dim, output_dim):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 20, 32),
            nn.SELU(),
            nn.Linear(32, output_dim),
        )
        
        init_network_weights(self.net, std=0.001)

    def forward(self, z, cmax_time):
        cmax_time = cmax_time.repeat(z.size(0), 1, 1)
        z = torch.cat([z, cmax_time], 2)
        return self.net(z)



def load_model(ckpt_path, encoder=None, ode_func=None, classifier=None, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")

    checkpt = torch.load(ckpt_path)
    if encoder is not None:
        encoder_state = checkpt["encoder"]
        encoder.load_state_dict(encoder_state)
        encoder.to(device)

    if ode_func is not None:
        ode_state = checkpt["ode"]
        ode_func.load_state_dict(ode_state)
        ode_func.to(device)

    if classifier is not None:
        classifier_state = checkpt["classifier"]
        classifier.load_state_dict(classifier_state)
        classifier.to(device)


