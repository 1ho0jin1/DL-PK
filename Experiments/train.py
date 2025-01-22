import os
import yaml
import argparse
import matplotlib
matplotlib.use("Agg")  # use Agg backend for efficiency
import matplotlib.pyplot as plt
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
        ConsecutiveSampling(args.seq_len+args.pred_steps),
        PKPreprocess(),
    ])
    train_data = PKDataset(os.path.join(args.data_dir, 'train'), transform=train_trfm)
    valid_data = PKDataset(os.path.join(args.data_dir, 'valid'), transform=PKPreprocess())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # create model
    if args.model == 'lstm':
        model = LSTMPK(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    elif args.model == 'gru':
        model = GRUPK(input_dim=args.input_dim, meta_dim=args.meta_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    # elif args.model == 'transformer':
    #     model = TransformerPK().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}. Must be one of 'lstm', 'gru', 'transformer'")

    # define loss function, optimizer, scheduler
    if hasattr(args, "l1loss") and args.l1loss:
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # create tensorboard writer
    writer = SummaryWriter(args.save_dir)

    # training loop
    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0.0
        
        # curriculum learning: gradually decrease supervision with epochs
        supervision_ratio = (1 - epoch / args.epochs) * float(args.curriculum)

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data = batch['data'].to(device)
            meta = batch['meta'].to(device)      # TODO: use metadata somehow
            input = data[:, :-1]                 # input: all time steps except the last one
            target = data[:, -1, 3].view(-1, 1)  # target: predict DV of the last time step

            B, N = data.shape[:2]
            input = data.clone()  # make a copy as we will modify the input
            loss = 0.0
            for i in range(args.pred_steps):
                input_i = input[:, i:i+args.seq_len]
                target = data[:, i+args.seq_len, 3].view(-1, 1)
                if i > 0 and np.random.random() < supervision_ratio:  # substitute DV of the last time step with the predicted value
                    input_i[:, -1, 3] = output.squeeze()
                else:
                    input_i[:, -1, 3] = data[:, i+args.seq_len, 3]    # simply use the ground truth value for supervision
                output = model(input_i, meta)
                loss += criterion(output, target)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader) * args.pred_steps
        writer.add_scalar('loss/train', train_loss, epoch)

        # validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for iter, batch in enumerate(valid_loader):
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
                    loss = criterion(output, target)
                    valid_loss += loss.item()
                    output_logs = torch.cat([output_logs, output], dim=1)
                
                # visualize
                if args.plot_every > 0 and iter == 0:
                    if epoch == 0 or (epoch+1) % args.plot_every == 0:
                        # plot the first 16 patients
                        fig, ax = plt.subplots(4,4, figsize=(20,16))
                        for i in range(16):
                            loss_i = criterion(data[i, :, 3], output_logs[i]).item()
                            ax[i//4, i%4].plot(data[i, :, 3].cpu().numpy(), label='Label')
                            ax[i//4, i%4].plot(output_logs[i].cpu().numpy(), linestyle='--',label='Prediction')
                            ax[i//4, i%4].set_title(f'ID:{batch["ptid"][i]}, MSE:{loss_i:.3f}')
                            ax[i//4, i%4].legend(['Label', 'Prediction'])
                        fig.savefig(args.save_dir / f'val_epoch{epoch}.png', bbox_inches='tight', dpi=300)

            valid_loss /= len(valid_loader) * (N - args.seq_len)
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
                }, os.path.join(args.save_dir, 'best.pt'))
        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            torch.save({
                "configs": vars(args),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }, os.path.join(args.save_dir, f'epoch{epoch+1}.pt'))



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # directory arguments
    args.add_argument('--data_dir', type=str, default='/home/hj/DL-PK/Experiments/dataset', help='dataset directory where ./train ./valid exists')
    args.add_argument('--yaml_path', type=str, default='/home/hj/DL-PK/Experiments/configs/gru_250122.yaml', help='path of config.yaml')
    args.add_argument('--run_name', type=str, default='testrun', help='name of this run')
    args.add_argument('--device', type=int, default=0, help='cuda index. ignored if cuda device is unavailable')
    args.add_argument('--num_workers', type=int, default=16, help='number of workers for dataloader')
    
    # logging arguments
    args.add_argument('--plot_every', type=int, default=100, help='Plot every N epochs; Do not save when -1')
    args.add_argument('--ckpt_every', type=int, default=-1, help='Save checkpoints every N epochs; Do not save when -1')
    
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