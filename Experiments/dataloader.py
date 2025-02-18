import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms



class PKDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the 'DATA' folder.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., for normalization).
        
        NOTE: Here we assume a directory structure like:
        DATA
            ├── 0001
            │    ├── data.txt
            │    └── meta.txt
            ├── 0002
            │    ├── data.txt
            │    └── meta.txt
            ├── ...
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patient_ids = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        # pre-load all data and meta files
        print("Loading data and meta files...")
        self.patient_data = []
        self.patient_meta = []
        for patient_id in tqdm(self.patient_ids):
            data_file = os.path.join(self.root_dir, patient_id, 'data.txt')
            meta_file = os.path.join(self.root_dir, patient_id, 'meta.txt')
            data = np.loadtxt(data_file, delimiter=None, ndmin=2)
            meta = np.loadtxt(meta_file, delimiter=None)
            self.patient_data.append(data)
            self.patient_meta.append(meta)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data = self.patient_data[idx]
        meta = self.patient_meta[idx]

        data_tensor = torch.from_numpy(data).float()
        meta_tensor = torch.from_numpy(meta).float()

        sample = {
            'ptid': patient_id,   # patient ID
            'data': data_tensor,  # shape: (num_rows, 4)
            'meta': meta_tensor   # shape: (4,)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample



class ConsecutiveSampling(object):
    """
    Samples a consecutive sequence of length `seq_len` from the data.
    """
    def __init__(self, seq_len=24):
        self.seq_len = seq_len
    def __call__(self, sample):
        num_rows = sample['data'].shape[0]
        start_idx = np.random.randint(0, num_rows - self.seq_len + 1)
        sample['data'] = sample['data'][start_idx:start_idx+self.seq_len]
        return sample



class Normalize(object):
    """
    sample['data']: (N,4) array, each row is ["TIME", "TAD", "AMT", "DV"]
    sample['meta']: (4,) array, each row is ["SEX", "AGE", "WT", "Cr"]
    AMT : re-scale by scale_amt set to 1000 (mg)
    DV  : re-scale by scale_dv set to 100 (mg/L)
    AGE : re-scale by scale_age set to 100 (yrs)
    WT  : re-scale by scale_wt set to 100 (kg)
    """
    def __init__(self, scale_amt=1000, scale_dv=100, scale_age=100, scale_wt=100):
        self.scale_amt = scale_amt
        self.scale_dv = scale_dv
        self.scale_age = scale_age
        self.scale_wt = scale_wt
    def __call__(self, sample):
        # rescale AMT, DV, AGE, WT
        sample['data'][:,2] = sample['data'][:,2] / self.scale_amt
        sample['data'][:,3] = sample['data'][:,3] / self.scale_dv
        sample['meta'][1] = sample['meta'][1] / self.scale_age
        sample['meta'][2] = sample['meta'][2] / self.scale_wt
        return sample



class RandomScaling(object):
    """
    Randomly scale both DV and AMT values by a factor between a given range, (0.8, 1.2) for default.
    NOTE: DV-AMT relationship may not be linear, but we assume linearity for small enough perturbations.
    """
    def __init__(self, p=0.2, scale_range=(0.8, 1.2)):
        self.p = p
        self.scale_range = scale_range
    def __call__(self, sample):
        if np.random.rand() < self.p:
            scale_factor = np.random.uniform(*self.scale_range)
            sample['data'][:, 2:4] *= scale_factor  # scale AMT and DV
        return sample



class DVJitter(object):
    """
    Add random noise to DV values to simulate measurement error.
    Noise is assumed to be at most 10% of the DV value (i.e., multiplicative noise).
    """
    def __init__(self, noise_ratio=0.1):
        self.noise_ratio = noise_ratio
    def __call__(self, sample):
        noise = (torch.rand_like(sample['data'][:,3]) - 0.5) * 2 * self.noise_ratio
        sample['data'][:,3] *= (1 + noise)  # add multiplicative noise to DV
        return sample



class RandomNullSampling(object):
    """
    Randomly add a 'null sample' where all AMT and DV values are set to zero.
    This regularizes the model to learn meaningful PK dynamics (i.e., no DV if no AMT),
    instead of just predicting some periodic pattern regardless of the input.
    """
    def __init__(self, p=0.1):
        self.p = p
    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample['data'][:, 2:4] = 0.0  # set AMT and DV to zero
        return sample




if __name__ == "__main__":
    base = Path(__file__).parent
    path = base / 'dataset/valid'
    
    transform = transforms.Compose([
        ConsecutiveSampling(seq_len=24),
        Normalize(),
        RandomScaling(p=0.2, scale_range=(0.8, 1.2)),
        DVJitter(),
        RandomNullSampling(p=1.0),
    ])
    
    dataset = PKDataset(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=17, shuffle=True)
    
    # get one sample
    data = next(iter(dataloader))
    print(data['ptid'])
    print(data['meta'])
    print(data['data'].shape)