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
    def __init__(self, seq_len=240):
        self.seq_len = seq_len
    def __call__(self, sample):
        num_rows = sample['data'].shape[0]
        start_idx = np.random.randint(0, num_rows - self.seq_len + 1)
        sample['data'] = sample['data'][start_idx:start_idx+self.seq_len]
        return sample



class PKPreprocess(object):
    """
    sample['data']: (N,4) array, each row is ["TIME", "TAD", "AMT", "DV"]
    TIME: use difference between consecutive time points instead of absolute value
    DV  : re-scale by scale_dv set to 100 (mg/L)
    AGE : re-scale by scale_age set to 100 (yrs)
    WT  : re-scale by scale_wt set to 100 (kg)
    """
    def __init__(self, scale_dv=100, scale_age=100, scale_wt=100):
        self.scale_dv = scale_dv
        self.scale_age = scale_age
        self.scale_wt = scale_wt
    def __call__(self, sample):
        # # use difference between consecutive time points
        # sample['data'][1:,0] = torch.diff(sample['data'][:,0])
        # sample['data'][0, 0] = 0.0
        # rescale DV, AGE, WT
        sample['data'][:,3] = sample['data'][:,3] / self.scale_dv
        sample['meta'][1] = sample['meta'][1] / self.scale_age
        sample['meta'][2] = sample['meta'][2] / self.scale_wt
        return sample




if __name__ == "__main__":
    base = Path(__file__).parent
    path = base / 'dataset/valid'
    
    transform = transforms.Compose([
        ConsecutiveSampling(seq_len=240),
        PKPreprocess(scale_dv=200.0)
    ])
    
    dataset = PKDataset(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # get one sample
    data = next(iter(dataloader))
    print(data['ptid'])
    print(data['meta'])
    print(data['data'].shape)