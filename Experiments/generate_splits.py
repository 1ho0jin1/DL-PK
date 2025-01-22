import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils.general import set_random_seed



def main(args):
    # set random seed
    set_random_seed()

    base = Path(__file__).parent / 'dataset'
    source = Path(args.source_dir)

    id_list = os.listdir(source)
    id_list = np.random.permutation(id_list).tolist()

    # split by 4:3:3
    n = len(id_list) // 10
    train_list = id_list[:4*n]
    valid_list = id_list[4*n:7*n]
    test_list = id_list[7*n:]

    # create folders for each split and create symlinks
    for split, lst in zip(['train', 'valid', 'test'], [train_list, valid_list, test_list]):
        dest = base / split
        os.makedirs(dest, exist_ok=True)
        for id_ in tqdm(lst):
            os.symlink(source / id_, dest / id_)
            # shutil.copytree(source / id_, dest / id_) # if you want to copy instead of symlink


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='dataset')
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    main(args)