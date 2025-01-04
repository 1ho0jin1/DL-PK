import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


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
        self.transform = transform

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
            'data': data_tensor,  # shape: (num_rows, 3)
            'meta': meta_tensor   # shape: (4,) in your example
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    


if __name__ == "__main__":
    base = Path(__file__).parent
    path = base / 'dataset/valid'
    dataset = PKDataset(path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # get one sample
    data = next(iter(dataloader))
    print(data['ptid'])
    print(data['meta'])
    print(data['data'].shape)


    # load models and test
    from models.gru import GRUPK
    from models.lstm import LSTMPK
    from models.transformer import TransformerPK
    model1 = GRUPK()
    model2 = LSTMPK()
    model3 = TransformerPK()
    print()