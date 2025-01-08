import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
 
base = Path(__file__).parent
 
def process_patient_ids(patient_ids, data, data_aux, columns_var, columns_fixed, dest, fn):
    for id_ in tqdm(patient_ids):
        print(f"Processing Patient ID {id_}...")
        dest_id = dest / str(id_).zfill(4)
        data_id = data[data['ID'] == id_]
        meta_id = data_id[columns_fixed].to_numpy()
        data_id = data_id[columns_var].to_numpy()
 
        # deal with NaNs for AMT
        data_id[np.isnan(data_id[:,1]), 1] = 0.0  # NaN for AMT means 0
        # deal with NaNs for DV
        dv_nan_idx = np.isnan(data_id[:,2])
        if fn == 'ground_truth_241231.csv':
            data_id[dv_nan_idx, 2] = data_id[np.where(dv_nan_idx)[0]-1, 2]
        elif fn == 'observation_241231.csv':
            timestamp = data_id[dv_nan_idx, 0]
            data_aux_id = data_aux[data_aux['ID'] == id_][columns_var].to_numpy()
            sorter = np.argsort(data_aux_id[:,0])
            aux_idx = np.searchsorted(data_aux_id[:,0], timestamp, sorter=sorter)
            data_id[dv_nan_idx, 2] = data_aux_id[aux_idx, 2]
       
        # check DV correctness
        if (data_id[:, 2] > 1000).sum().any():
            print(f"ID {id_} has max DV:{data_id[:, 2].max()}, skipping as it may be corrupted")
            continue
        assert np.unique(meta_id).shape == (4,), "These columns should be same for each ID"
        
        # make destination folder
        os.makedirs(dest_id, exist_ok=True)
        SEX, AGE, WT, Cr = meta_id[0]
        with open(dest_id / 'meta.txt', 'w') as fp:
            fp.write(f"{int(SEX)} {int(AGE)} {WT:.4f} {Cr:.4f}\n")
        for TIME, AMT, DV in data_id:
            with open(dest_id / 'data.txt', 'a') as fp:
                fp.write(f"{TIME:.4f} {AMT:.4f} {DV:.4f}\n")
 
def main():
    num_workers = min(16, os.cpu_count())
   
    for fn in ['ground_truth_241231.csv']:
        data = pd.read_csv(base / fn)  # read data
        dest = base / fn[:-4]          # destination folder
 
        if fn == 'observation_241231.csv':
            data_aux = pd.read_csv(base / 'ground_truth_241231.csv')
        else:
            data_aux = None
 
        columns_var = ['TIME', 'AMT', 'DV']  # We drop TAD as it can be derived from TIME and AMT
        columns_fixed = ['SEX', 'AGE', 'WT', 'Cr']  # These columns are fixed for each ID
 
        unique_ids = data['ID'].unique()
        chunk_size = len(unique_ids) // num_workers
        chunks = [unique_ids[i:i + chunk_size] for i in range(0, len(unique_ids), chunk_size)]
 
        with Pool(num_workers) as pool:
            pool.starmap(process_patient_ids, [(chunk, data, data_aux, columns_var, columns_fixed, dest, fn) for chunk in chunks])
 
if __name__ == "__main__":
    main()
 