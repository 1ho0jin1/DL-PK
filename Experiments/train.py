import os
import argparse
import numpy as np
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
from dataloader import PKDataset


#set project base directory
base = Path(__file__).parent


def main(args):
    # set random seed
    set_random_seed(args.seed)

    # create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # set device
    device = torch.device(args.device)

    # load data and create dataloaders
    train_data = PKDataset(os.path.join(args.data_dir, 'train'))
    valid_data = PKDataset(os.path.join(args.data_dir, 'valid'))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    # create model
    if args.model == 'lstm':
        model = LSTMPK().to(device)
    elif args.model == 'gru':
        model = GRUPK().to(device)
    elif args.model == 'transformer':
        model = TransformerPK().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}. Must be one of 'lstm', 'gru', 'transformer'")

    # define loss function, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # create tensorboard writer
    writer = SummaryWriter(args.save_dir)

    # training loop
    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data = batch['data'].to(device)
            meta = batch['meta'].to(device)      # TODO: use metadata somehow
            target = data[:, -1, 2].view(-1, 1)  # target: predict DV of the last time step

            output = model(data, meta)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        writer.add_scalar('loss/train', train_loss, epoch)

        # validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for i, batch in enumerate(valid_loader):
                data = batch['data'].to(device)
                meta = batch['meta'].to(device)
                target = data[:, -1, 2].view(-1, 1)  # target: predict DV of the last time step
                output = model(data, meta)
                loss = criterion(output, target)
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            writer.add_scalar('loss/valid', valid_loss, epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")
        scheduler.step()

        # save model if validation loss is minimum
        if epoch == 0 or valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.model}.pt'))




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # directory arguments
    args.add_argument('--data_dir', type=str, default=r'C:\Users\qkrgh\Jupyter\DL-PK\Experiments\dataset')
    args.add_argument('--run_name', type=str, default='', help='name of the training run')

    # training arguments
    args.add_argument('--device', type=str, default='cpu')
    args.add_argument('--model', type=str, default='', help='lstm, gru, transformer')
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--epochs', type=int, default=10)

    # miscellaneous arguments: no need to change!
    args.add_argument('--seed', type=int, default=2025)
    args = args.parse_args()

    # set save directory
    args.save_dir = base / 'runs/train' / args.run_name

    main(args)