import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


base = Path(__file__).parent

# Preprocess data
for fn in ['ground_truth_241231.csv']:
    data = pd.read_csv(base / fn)  # read data
    dest = base / fn[:-4]          # destination folder

    columns_var = ['TIME', 'AMT', 'DV']  # We drop TAD as it can be derived from TIME and AMT
    columns_fixed = ['SEX', 'AGE', 'WT', 'Cr']  # These columns are fixed for each ID
    for id_ in tqdm(data['ID'].unique()):
        dest_id = dest / str(id_).zfill(4)
        os.makedirs(dest_id, exist_ok=True)
        
        data_id = data[data['ID'] == id_]
        meta_id = data_id[columns_fixed].to_numpy()
        data_id = data_id[columns_var].to_numpy()

        # NaN for AMT: this is simply 0.0
        data_id[np.isnan(data_id[:,1]), 1] = 0.0
        # NaN for DV: dose stage, so DV should be identical to previous row if time is same
        idx = np.nonzero(np.isnan(data_id[:,2]))[0]
        assert (data_id[idx, 0] == data_id[idx-1, 0]).all(), "Time should be same for dose stage"
        data_id[idx, 2] = data_id[idx-1, 2]

        assert np.unique(meta_id).shape == (4,), "These columns should be same for each ID"
        SEX, AGE, WT, Cr = meta_id[0]
        with open(dest_id / 'meta.txt', 'w') as fp:
            fp.write(f"{int(SEX)} {int(AGE)} {WT:.4f} {Cr:.4f}\n")
        for TIME, AMT, DV in data_id:
            with open(dest_id / 'data.txt', 'a') as fp:
                fp.write(f"{TIME:.4f} {AMT:.4f} {DV:.4f}\n")