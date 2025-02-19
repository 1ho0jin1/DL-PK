import torch
from torchdiffeq import odeint
from general import sample_standard_gaussian
from sklearn.metrics import mean_squared_error, r2_score



def compute_loss_on_train(criterion, labels, preds):
    preds = preds.permute(1, 0, 2)

    idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
    # print(idx_not_nan)
    preds = preds[idx_not_nan]
    labels = labels[idx_not_nan]

    return torch.sqrt(criterion(preds, labels))


def compute_loss_on_test(encoder, ode_func, classifier, args, dataloader, n_batches, device, phase):
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)
    latent_dim = 6

    for itr in range(n_batches):
        ptnm, times, features, labels, cmax_time = dataloader.__next__()
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
            print(time0, time1, time_interval, ptnm)
            continue
    
        preds = classifier(solves, cmax_time).permute(1, 0, 2)

        if phase == "test":
            idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
            # print(idx_not_nan)
            preds = preds[idx_not_nan]
            labels = labels[idx_not_nan]

            times = times[idx_not_nan.flatten()]
            ptnms += ptnm * len(times)
            Times = torch.cat((Times, times*24))

            predictions = torch.cat((predictions, preds))
            ground_truth = torch.cat((ground_truth, labels))
        
        else:
            """
            time_idx = (times >= 21)
            preds = preds[:, time_idx, :]
            labels = labels[:, time_idx, :]
            """
            idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
            # print(idx_not_nan)
            preds = preds[idx_not_nan]
            labels = labels[idx_not_nan]

            predictions = torch.cat((predictions, preds))
            ground_truth = torch.cat((ground_truth, labels))

    rmse_loss = mean_squared_error(
        ground_truth.cpu().numpy(), predictions.cpu().numpy(),
        squared=False
    )
    r2 = r2_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())

    if phase == "test":
        return {"PTNM": ptnms,
                "TIME": Times,  
                "labels": ground_truth.cpu().tolist(), 
                "preds": predictions.cpu().tolist(),
                "loss": rmse_loss}
    else:
        return {"labels": ground_truth.cpu().tolist(), 
                "preds": predictions.cpu().tolist(),
                "loss": rmse_loss,
                "r2": r2}


def compute_loss_on_interp(encoder, ode_func, classifier, args, dataloader, dataloader_o, n_batches, device, phase):
    ptnms = []
    Times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)
    latent_dim = 6

    for itr in range(n_batches):
        ptnm, times, features, labels, cmax_time = dataloader.__next__()
        ptnm_o, times_o, features_o, labels_o, cmax_time_o = dataloader_o.__next__()
        assert ptnm == ptnm_o

        dosing = torch.zeros([features.size(0), features.size(1), latent_dim])
        dosing[:, :, 0] = features[:, :, -2]
        dosing = dosing.permute(1, 0, 2)

        encoder_out = encoder(features_o)
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
            print(time0, time1, time_interval, ptnm)
            continue
    
        preds = classifier(solves, cmax_time).permute(1, 0, 2)

        idx_not_nan = ~(torch.isnan(labels) | (labels == -1))
        # print(idx_not_nan)
        preds = preds[idx_not_nan]
        labels = labels[idx_not_nan]

        times = times[idx_not_nan.flatten()]
        ptnms += ptnm * len(times)
        Times = torch.cat((Times, times*24))

        predictions = torch.cat((predictions, preds))
        ground_truth = torch.cat((ground_truth, labels))
        
    rmse_loss = mean_squared_error(
        ground_truth.cpu().numpy(), predictions.cpu().numpy(),
        squared=False
    )
    r2 = r2_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())

    return {"PTNM": ptnms,
            "TIME": Times,  
            "labels": ground_truth.cpu().tolist(), 
            "preds": predictions.cpu().tolist(),
            "loss": rmse_loss}



if __name__ == "__main__":
    print()