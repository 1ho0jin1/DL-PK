import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from model import Encoder, ODEFunc, Classifier, load_model
from utils.dataset import parse_tdm1
from utils.plot import plot_preds_on_test_interp



def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###############################################################
    ## Main runnings
    ckpt_path = os.path.join(args.save_dir, f"fold_{args.fold}_model_{args.model}.ckpt")
    eval_path = os.path.join(args.save_dir, f"fold_{args.fold}_model_{args.model}.csv")
    res_path = "rmse.csv"

    ########################################################################
    tdm1_obj = parse_tdm1(args.base, device, phase="test")
    input_dim = tdm1_obj["input_dim"]
    hidden_dim = 128 
    latent_dim = 6

    encoder = Encoder(input_dim=input_dim, output_dim=2 * latent_dim, hidden_dim=hidden_dim)
    ode_func = ODEFunc(input_dim=latent_dim, hidden_dim=16)
    classifier = Classifier(latent_dim=latent_dim, output_dim=1)

    load_model(ckpt_path, encoder, ode_func, classifier, device)

    # ########################################################################
    # ## Predict & Evaluate
    # with torch.no_grad():
    #     test_res = utils.compute_loss_on_test(encoder, ode_func, classifier, args,
    #         tdm1_obj["test_dataloader"], tdm1_obj["n_test_batches"], 
    #         device, phase="test")

    # eval_results = pd.DataFrame(test_res).drop(columns="loss")
    # eval_results.to_csv(eval_path, index=False)
    # ########################################################################

    ########################################################################
    ## Predict & Plot Results
    with torch.no_grad():
        test_res = plot_preds_on_test_interp(encoder, ode_func, classifier, args,
            tdm1_obj["test_dataloader"], tdm1_obj["n_test_batches"], 
            device, phase="test", save_dir=f"{args.save_dir}/plots_model_{args.model}")
    ########################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1, help='fold index')
    parser.add_argument('--model', type=int, default=1, help='model index')
    args = parser.parse_args()

    # save path
    base = Path(__file__).parent
    args.save_dir = f"{base}/runs/train/fold_{args.fold}/model_{args.model}"

    main(args)