import os
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from models import *
from dataloader import *
from utils.general import set_random_seed


#set project base directory
base = Path(__file__).parent


def main(args):
    # set random seed
    set_random_seed(args.seed)
    
    # set device
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # load data and create dataloaders
    train_trfm = transforms.Compose([
        ConsecutiveSampling(args.seq_len),
        PKPreprocess(),
    ])
    train_data = PKDataset(os.path.join(args.data_dir, 'train'), transform=train_trfm)
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
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
            input = data[:, :-1]                 # input: all time steps except the last one
            target = data[:, -1, 2].view(-1, 1)  # target: predict DV of the last time step

            output = model(input, meta)
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
                
                # split sequences into args.seq_len and predict each splits
                N = data.shape[1]
                for i in range(0, N, args.seq_len):
                    s, e = i, min(i+args.seq_len-1, N-1)
                    input = data[:, s:e]
                    target = data[:, e, 2].view(-1, 1)
                    output = model(input, meta)
                    loss = criterion(output, target)
                    valid_loss += loss.item()

            valid_loss /= len(valid_loader) * (N // args.seq_len)
            writer.add_scalar('loss/valid', valid_loss, epoch)

        tqdm.write(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")
        
        # log learning rate
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # step scheduler
        scheduler.step()

        # save model if validation loss is minimum
        if epoch == 0 or valid_loss < min_loss:
            min_loss = valid_loss
            torch.save({
                "configs": vars(args),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }, os.path.join(args.save_dir, f'best.pt'))
            



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # directory arguments
    args.add_argument('--data_dir', type=str, default=r'C:\Users\qkrgh\Jupyter\DL-PK\Experiments\dataset')
    args.add_argument('--yaml_path', type=str, default=r'C:\Users\qkrgh\Jupyter\DL-PK\Experiments\config\gru_250109.yaml', help='path of config.yaml')
    args.add_argument('--run_name', type=str, default='gru_250109', help='name of this run')
    args.add_argument('--device', type=int, default=0, help='cuda index. ignored if cuda device is unavailable')
    
    # miscellaneous arguments: no need to change!
    args.add_argument('--seed', type=int, default=2025)
    args = args.parse_args()
    # parse configs
    with open(args.yaml_path, 'r') as fp:
        config = yaml.safe_load(fp)
    for key, value in config.items():
        setattr(args, key, value)

    # set & create save directory
    args.save_dir = base / 'runs/train' / args.run_name
    os.makedirs(args.save_dir, exist_ok=True)
    print("Save directory:", args.save_dir)

    main(args)