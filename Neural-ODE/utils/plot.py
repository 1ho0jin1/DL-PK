import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchdiffeq import odeint
from general import sample_standard_gaussian
from sklearn.metrics import mean_squared_error, r2_score



def plot_preds_on_test(encoder, ode_func, classifier, args, dataloader, n_batches, device, phase, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)
    latent_dim = 6

    for itr in tqdm(range(n_batches)):
        ptnm, times, features, labels, cmax_time = dataloader.__next__()  # features: ['TFDS','TIME','CYCL','AMT',"PK_round1"]
        dosing = torch.zeros([features.size(0), features.size(1), latent_dim])
        dosing[:, :, 0] = features[:, :, -2]  # AMT: Dosing amounts
        dosing = dosing.permute(1, 0, 2)

        encoder_out = encoder(features)  # seems like trained as VAE
        qz0_mean, qz0_var = encoder_out[:, :latent_dim], encoder_out[:, latent_dim:]
        z0 = sample_standard_gaussian(qz0_mean, qz0_var)

        solves = z0.unsqueeze(0).clone()
        try:
            for idx, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
                z0 += dosing[idx]
                time_interval = torch.Tensor([time0 - time0, time1 - time0])
                sol = odeint(ode_func, z0, time_interval, rtol=args.tol, atol=args.tol)  # sol[0] == z[time0] -> sol[1] = z[time1]
                z0 = sol[-1].clone()
                solves = torch.cat([solves, sol[-1:, :]], 0)
        except AssertionError:
            print(times)
            print(time0, time1, time_interval, ptnm)
            continue
    
        preds = classifier(solves, cmax_time).permute(1, 0, 2)


        idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
        # print(idx_not_nan)
        preds = preds[idx_not_nan].cpu()
        labels = labels[idx_not_nan].cpu()
        times = times[idx_not_nan.flatten()].cpu() * 24  # from days to hours
        amt = dosing[..., 0].squeeze().cpu()
        rmse_loss = mean_squared_error(
            preds.cpu().numpy(), labels.cpu().numpy(),
            squared=False
        )
        plot_dv_amt(ptnm[0], rmse_loss, times, preds, labels, amt, save_dir)



        
        
def plot_preds_on_test_interp(encoder, ode_func, classifier, args, dataloader, n_batches, device, phase, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)
    latent_dim = 6

    for itr in tqdm(range(n_batches)):
        ptnm, times, features, labels, cmax_time = dataloader.__next__()  # features: ['TFDS','TIME','CYCL','AMT',"PK_round1"]
        
        # Let's not consider NaN for now
        assert torch.isnan(labels).sum() == 0, "NaN in labels"

        TFDS = features[:, :, 0]       # Time from Dose in hours
        TIME = features[:, :, 1]       # Time in days
        CYCL = features[:, :, 2]       # Cycle number
        AMT = features[:, :, 3]        # Dosing amount
        PK_round1 = features[:, :, 4]  # Concentration <- must be "guessed" through interpolation

        # Interpolation: from time 0 to max time in 1 day interval
        TIME_I = torch.arange(0, TIME.max() + 1)
        mask = AMT > 0
        AMT_I = torch.zeros_like(TIME_I)
        AMT_I[TIME[mask].to(int)] = AMT[mask]
        TFDS_ = torch.cat((TIME[mask], TIME[:, -1] + 1))
        TFDS_I = torch.arange(TFDS_[1])
        CYCL_I = torch.ones(int(TFDS_[1]),)
        for i in range(1, len(TFDS_) - 1):
            t = TFDS_[i+1] - TFDS_[i]
            TFDS_I = torch.cat((TFDS_I, torch.arange(t)))
            CYCL_I = torch.cat((CYCL_I, torch.ones(int(t),) * (i + 1)))
        TFDS_I = TFDS_I * 24  # from days to hours

        # perform simple 1-d interpolation for PK_round1
        PK_round1_I = np.interp(np.arange(int(TIME.max())+1), TIME.squeeze(), PK_round1.squeeze())
        PK_round1_I = torch.Tensor(PK_round1_I)

        # stack all features to create interpolated features
        features_I = torch.stack((TFDS_I, TIME_I, CYCL_I, AMT_I, PK_round1_I), dim=1).unsqueeze(0)

        dosing = torch.zeros([features_I.size(0), features_I.size(1), latent_dim])
        dosing[:, :, 0] = features_I[:, :, -2]  # AMT: Dosing amounts
        dosing = dosing.permute(1, 0, 2)

        encoder_out = encoder(features_I)  # seems like trained as VAE
        qz0_mean, qz0_var = encoder_out[:, :latent_dim], encoder_out[:, latent_dim:]
        z0 = sample_standard_gaussian(qz0_mean, qz0_var)

        solves = z0.unsqueeze(0).clone()
        try:
            for idx, (time0, time1) in enumerate(zip(TIME_I[:-1], TIME_I[1:])):
                z0 += dosing[idx]
                time_interval = torch.Tensor([time0 - time0, time1 - time0])
                sol = odeint(ode_func, z0, time_interval, rtol=args.tol, atol=args.tol)  # sol[0] == z[time0] -> sol[1] = z[time1]
                z0 = sol[-1].clone()
                solves = torch.cat([solves, sol[-1:, :]], 0)
        except AssertionError:
            print(TIME_I)
            print(time0, time1, time_interval, ptnm)
            continue
    
        preds = classifier(solves, cmax_time).permute(1, 0, 2)

        # VISUALIZATION
        preds = preds.squeeze().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        amt = dosing[..., 0].squeeze().cpu().numpy()
        # for index preds for RMSE calculation
        preds_ = preds[times.to(int)]
        times = times.cpu() * 24  # from days to hours
        TIME_I = TIME_I.cpu() * 24  # from days to hours
        
        # compute rmse
        rmse_loss = mean_squared_error(preds_, labels, squared=False)

        # plot interp preds
        plot_dv_amt_interp(ptnm[0], rmse_loss, TIME_I, preds, times, labels, amt, save_dir)



def plot_dv_amt(ptnm, rmse_loss, times, dv_pred, dv_gt, amt, save_dir):
    # patient number
    ptnm = str(ptnm).zfill(4)
    
    # Create a dual-axis plot
    fig, ax1 = plt.subplots()

    # Plot "DV" on the primary y-axis
    ax1.set_xlabel('Time (hr)')
    ax1.set_ylabel('DV (mcg/mL)', color='tab:blue')
    # ground truth
    ax1.scatter(times, dv_gt, color='tab:blue', marker='^', s=50, label='Ground Truth')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # prediction
    ax1.plot(times, dv_pred, color='deepskyblue', linestyle='--')
    ax1.scatter(times, dv_pred, color='deepskyblue', marker='o', s=30, label='Prediction')
    # ax1.set_ylim(0, 100)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add secondary y-axis for "AMT"
    ax2 = ax1.twinx()
    ax2.set_ylabel('AMT (mg)', color='tab:red')
    amt_mask = amt > 0
    ax2.scatter(times[amt_mask], amt[amt_mask], color='tab:red', marker='*', s=100, label='Dose (AMT)')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # ax2.set_ylim(100, 500)

    # Add legends
        # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Add a single unified legend
    fig.legend(handles, labels)

    # Add a title and show the plot
    plt.title(f'PT#{ptnm}, RMSE:{rmse_loss:.3f}')
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f"{save_dir}/ptnm-{ptnm}.jpg", bbox_inches='tight', dpi=200)
    plt.close()



def plot_dv_amt_interp(ptnm, rmse_loss, times_i, dv_pred, times, dv_gt, amt, save_dir):
    save_dir += "_interp"
    
    # patient number
    ptnm = str(ptnm).zfill(4)
    
    # Create a dual-axis plot
    fig, ax1 = plt.subplots()

    # Plot "DV" on the primary y-axis
    ax1.set_xlabel('Time (hr)')
    ax1.set_ylabel('DV (mcg/mL)', color='tab:blue')
    # ground truth
    ax1.scatter(times, dv_gt, color='tab:blue', marker='^', s=50, label='Ground Truth')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # prediction
    ax1.plot(times_i, dv_pred, color='deepskyblue', linestyle='--')
    ax1.scatter(times_i, dv_pred, color='deepskyblue', marker='o', s=30, label='Prediction')
    # ax1.set_ylim(0, 100)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add secondary y-axis for "AMT"
    ax2 = ax1.twinx()
    ax2.set_ylabel('AMT (mg)', color='tab:red')
    amt_mask = amt > 0
    ax2.scatter(times_i[amt_mask], amt[amt_mask], color='tab:red', marker='*', s=100, label='Dose (AMT)')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # ax2.set_ylim(100, 500)

    # Add legends
        # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Add a single unified legend
    fig.legend(handles, labels)

    # Add a title and show the plot
    plt.title(f'PT#{ptnm}, RMSE:{rmse_loss:.3f}')
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f"{save_dir}/ptnm-{ptnm}.jpg", bbox_inches='tight', dpi=200)
    plt.close()
