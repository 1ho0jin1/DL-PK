import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # use Agg backend for efficiency
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils.general import set_random_seed
from dataloader import *


#set project base directory
base = Path(__file__).parent


def main(args):
    # set random seed
    set_random_seed(args.seed)

    # create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    if args.plotall:
        os.makedirs(args.save_dir / 'plots', exist_ok=True)
    
    # set device
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # load ckpt and set configs
    ckpt = torch.load(args.ckpt_path)
    for key, value in ckpt['configs'].items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # load data and create dataloaders
    dataset = PKDataset(args.source_dir, transform=Normalize())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # create model
    if args.model == 'lstm':
        model = LSTMPK(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    elif args.model == 'gru':
        model = GRUPK(input_dim=args.input_dim, meta_dim=args.meta_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    elif args.model == 'node':
        model = NeuralODE(input_dim=args.input_dim, meta_dim=args.meta_dim, hidden_dim=args.hidden_dim).to(device)
    # elif args.model == 'transformer':
    #     model = TransformerPK().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}. Must be one of 'lstm', 'gru', 'transformer'")
    
    # load model weights
    model.load_state_dict(ckpt['model'])

    # set criterion: L1 / L2 loss
    MAE = nn.L1Loss()
    MSE = nn.MSELoss()

    # inference loop
    model.eval()
    with torch.no_grad():
        loss_mae, loss_mse = 0,0
        for iter, batch in tqdm(enumerate(dataloader)):
            data = batch['data'].to(device)
            meta = batch['meta'].to(device)

            B, N = data.shape[:2]
            input = data.clone()  # make a copy as we will modify the input
            output_logs = data[:, :args.seq_len, 3]
            for i in range(0, N-args.seq_len):
                input_i = input[:, i:i+args.seq_len]
                target = data[:, i+args.seq_len, 3].view(-1, 1)
                if i > 0:  # substitute DV of the last time step with the predicted value
                    input_i[:, -1, 3] = output.squeeze()
                output = model(input_i, meta)
                output_logs = torch.cat([output_logs, output], dim=1)
                loss_mae += MAE(output, target).item()
                loss_mse += MSE(output, target).item()

            if args.plot and iter == 0:
                # plot the first few samples
                fig, ax = plt.subplots(4,4, figsize=(20,16))
                for i in range(16):
                    loss_i = MSE(data[i, :, 3], output_logs[i]).item()
                    ax[i//4, i%4].plot(data[i, :, 3].cpu().numpy(), label='Label')
                    ax[i//4, i%4].plot(output_logs[i].cpu().numpy(), linestyle='--',label='Prediction')
                    ax[i//4, i%4].set_title(f'ID:{batch["ptid"][i]}, MSE:{loss_i:.4f}')
                    ax[i//4, i%4].legend(['Label', 'Prediction'])
                fig.savefig(args.save_dir / 'inference.png', bbox_inches='tight', dpi=300)
                plt.close()
            if args.plotall:
                # plot all samples
                for i in range(B):
                    fig, ax = plt.subplots(1,1, figsize=(5,4))
                    loss_i = MSE(data[i, :, 3], output_logs[i]).item()
                    ax.plot(data[i, :, 3].cpu().numpy(), label='Label')
                    ax.plot(output_logs[i].cpu().numpy(), linestyle='--',label='Prediction')
                    ax.set_title(f'ID:{batch["ptid"][i]}, MSE:{loss_i:.4f}')
                    ax.legend(['Label', 'Prediction'])
                    fig.savefig(args.save_dir / f'plots/{batch["ptid"][i]}.png', bbox_inches='tight')
                    plt.close()

        
    loss_mae /= len(dataloader) * (N - args.seq_len)
    loss_mse /= len(dataloader) * (N - args.seq_len)
    with open(args.save_dir / 'loss.txt', 'w') as fp:
        fp.write(f"MAE: {loss_mae:.6f}\n")
        fp.write(f"MSE: {loss_mse:.6f}\n")




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # directory arguments
    args.add_argument('--source_dir', type=str, default='/home/hj/DL-PK/Experiments/dataset/test')
    args.add_argument('--ckpt_path', type=str, default='', help='path to .ckpt')
    args.add_argument('--run_name', type=str, default='gru_test', help='name of this run')
    args.add_argument('--device', type=int, default=0, help='cuda index. ignored if cuda device is unavailable')
    args.add_argument('--num_workers', type=int, default=16, help='number of workers for dataloader')

    # logging arguments
    args.add_argument('--plot', action='store_true', help='plot first few samples')
    args.add_argument('--plotall', action='store_true', help='plot all samples')


    # miscellaneous arguments: no need to change!
    args.add_argument('--seed', type=int, default=2025)
    args = args.parse_args()

    # set save directory
    args.save_dir = base / 'runs/inference' / args.run_name

    main(args)