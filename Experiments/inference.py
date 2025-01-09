import os
import argparse
import numpy as np
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
from dataloader import PKDataset, consecutive_sampling


#set project base directory
base = Path(__file__).parent


def main(args):
    # set random seed
    set_random_seed(args.seed)

    # create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # set device
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # load data and create dataloaders
    dataset = PKDataset(args.source_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # create model
    if args.model == 'lstm':
        model = LSTMPK().to(device)
    elif args.model == 'gru':
        model = GRUPK().to(device)
    elif args.model == 'transformer':
        model = TransformerPK().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}. Must be one of 'lstm', 'gru', 'transformer'")
    
    # load model weights
    ckpt = torch.load(args.weight_path)
    model.load_state_dict(ckpt['model'])

    # set criterion: L2 loss
    criterion = nn.MSELoss()

    # inference loop
    model.eval()
    with torch.no_grad():
        for iter, batch in tqdm(enumerate(dataloader)):
            data = batch['data'].to(device)
            meta = batch['meta'].to(device)
            
            B, N = data.shape[:2]
            input = data.clone()  # make a copy as we will modify the input
            output_logs = data[:, :args.seq_len, 2]
            for i in range(0, N-args.seq_len):
                input_i = input[:, i:i+args.seq_len-1]
                target = data[:, i+args.seq_len-1, 2].view(-1, 1)
                if i > 0:  # substitute DV of the last time step with the predicted value
                    input_i[:, -1, 2] = output.squeeze()
            
                output = model(input_i, meta)
                loss = criterion(output, target)
                output_logs = torch.cat([output_logs, output], dim=1)
                
            if iter == 0 and args.plot:
                # plot the first batch
                for i in range(B):
                    plt.plot(data[i, :, 2].cpu().numpy(), label='Label')
                    plt.plot(output_logs[i].cpu().numpy(), linestyle='--',label='Prediction')
                    plt.title(f'ID: {batch["ptid"][i]}')
                    plt.legend(['Label', 'Prediction'])
                    plt.savefig(args.save_dir / f'ptid_{batch["ptid"][i]}.png')
                    plt.close()



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # directory arguments
    args.add_argument('--source_dir', type=str, default=r'C:\Users\qkrgh\Jupyter\DL-PK\Experiments\dataset\train')
    args.add_argument('--weight_path', type=str, default=r'C:\Users\qkrgh\Jupyter\DL-PK\Experiments\runs\train\gru\best.pt', help='model weight path')
    args.add_argument('--run_name', type=str, default='gru_train', help='name of the training run')

    # training arguments
    args.add_argument('--device', type=str, default='0')
    args.add_argument('--model', type=str, default='gru', help='lstm, gru, transformer')
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--seq_len', type=int, default=240)

    # logging arguments
    args.add_argument('--plot', action='store_true', help='plot the first batch')


    # miscellaneous arguments: no need to change!
    args.add_argument('--seed', type=int, default=2025)
    args = args.parse_args()

    # set save directory
    args.save_dir = base / 'runs/inference' / args.run_name

    main(args)