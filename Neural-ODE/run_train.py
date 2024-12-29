import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

from model import Encoder, ODEFunc, Classifier, load_model
from utils.dataset import parse_tdm1
from utils.general import get_logger, sample_standard_gaussian
from utils.loss import compute_loss_on_train, compute_loss_on_test




def main(args):
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # make save dir
    os.makedirs(args.save_dir, exist_ok=True)

    ########################################################################
    ## Main runnings
    torch.manual_seed(args.random_seed + args.model + args.fold)
    np.random.seed(args.random_seed + args.model + args.fold)

    ckpt_path = os.path.join(args.save_dir, f"fold_{args.fold}_model_{args.model}.ckpt")

    ########################################################################
    tdm1_obj = parse_tdm1(phase="train", device='cpu')
    input_dim = tdm1_obj["input_dim"]
    hidden_dim = 128
    latent_dim = 6

    encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
    ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=16)
    classifier = Classifier(latent_dim=latent_dim, output_dim=1)

    if args.continue_train:
        load_model(ckpt_path, encoder, ode_func, classifier, device)

    ########################################################################
    ## Train
    log_path = f"{args.save_dir}/fold_{args.fold}_model_{args.model}.log"
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

    batches_per_epoch = tdm1_obj["n_train_batches"]
    criterion = nn.MSELoss().to(device=device)
    params = (list(encoder.parameters()) + 
            list(ode_func.parameters()) + 
            list(classifier.parameters()))
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.l2)
    best_rmse = 0x7fffffff
    best_epochs = 0

    for epoch in range(1, args.epochs):

        for _ in tqdm(range(batches_per_epoch), ascii=True):
            optimizer.zero_grad()

            ptnms, times, features, labels, cmax_time = tdm1_obj["train_dataloader"].__next__()
            dosing = torch.zeros([features.size(0), features.size(1), latent_dim])
            dosing[:, :, 0] = features[:, :, -2]
            dosing = dosing.permute(1, 0, 2)

            encoder_out = encoder(features)
            qz0_mean, qz0_var = encoder_out[:, :latent_dim], encoder_out[:, latent_dim:]
            z0 = sample_standard_gaussian(qz0_mean, qz0_var)
            
            solves = z0.unsqueeze(0).clone()
            try:
                for idx, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
                    z0 += dosing[idx]
                    time_interval = torch.Tensor([time0 - time0, time1 - time0])
                    sol = odeint(ode_func, z0, time_interval, rtol=args.tol, atol=args.tol)
                    z0 = sol[-1].clone()
                    solves = torch.cat([solves, sol[-1:, :]], 0)
            except AssertionError:
                print(times)
                print(time0, time1, time_interval, ptnms)
                continue
        
            preds = classifier(solves, cmax_time)

            loss = compute_loss_on_train(criterion, labels, preds)
            try: 
                loss.backward()
            except RuntimeError:
                print(ptnms)
                print(times)
                continue
            optimizer.step()
        
        idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
        preds = preds.permute(1, 0, 2)[idx_not_nan]
        labels = labels[idx_not_nan]
        print(preds)
        print(labels)

        with torch.no_grad():
            
            train_res = compute_loss_on_test(encoder, ode_func, classifier, args,
                tdm1_obj["train_dataloader"], tdm1_obj["n_train_batches"], 
                device, phase="train")

            validation_res = compute_loss_on_test(encoder, ode_func, classifier, args,
                tdm1_obj["val_dataloader"], tdm1_obj["n_val_batches"], 
                device, phase="validate")
            
            train_loss = train_res["loss"] 
            validation_loss = validation_res["loss"]
            if validation_loss < best_rmse:
                torch.save({'encoder': encoder.state_dict(),
                            'ode': ode_func.state_dict(),
                            'classifier': classifier.state_dict(),
                            'args': args}, ckpt_path)
                best_rmse = validation_loss
                best_epochs = epoch

            message = """
            Epoch {:04d} | Training loss {:.6f} | Training R2 {:.6f} | Validation loss {:.6f} | Validation R2 {:.6f}
            Best loss {:.6f} | Best epoch {:04d}
            """.format(epoch, train_loss, train_res["r2"], validation_loss, validation_res["r2"], best_rmse, best_epochs)
            logger.info(message)
            
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser("neural ODE model")
    parser.add_argument("--data", type=str, default='', help="data file for processing")
    parser.add_argument("--fold", type=int, default=1, help="current fold number")
    parser.add_argument("--model", type=int, default=1, help="current model number")
    parser.add_argument("--continue-train", action="store_true", help="continue training")
    parser.add_argument("--random-seed", type=int, default=1000, help="random seed")

    parser.add_argument("--layer", type=int, default=2, help="hidden layer of the ODE Function")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--l2", type=float, default=0.1, help="l2 regularization")
    parser.add_argument("--hidden", type=int, help="hidden dim in ODE Function")
    parser.add_argument("--tol", type=float, default=0.0001, help="control the precision in ODE solver")
    parser.add_argument("--epochs", type=int, default=30, help="epochs for training")

    args = parser.parse_args()

    # save path
    base = Path(__file__).parent
    args.save_dir = f"{base}/runs/train/fold_{args.fold}/model_{args.model}"
    print("Run saved at", args.save_dir)

    main(args)