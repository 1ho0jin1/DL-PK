import os
import shutil
from tqdm import tqdm
from pathlib import Path

base = "/home/hj/DL-PK/Data-Processing/Simulation-Data/ground_truth_250110"
dest = "/home/hj/DL-PK/Data-Processing/Simulation-Data/ground_truth_250110_subsampled"

for pid in tqdm(os.listdir(base)):
    os.makedirs(f"{dest}/{pid}", exist_ok=True)
    # just copy meta.txt
    shutil.copy(f"{base}/{pid}/meta.txt", f"{dest}/{pid}/meta.txt")

    with open(f"{base}/{pid}/data.txt", 'r') as fp:
        lines = fp.readlines()
    
    with open(f"{dest}/{pid}/data.txt", 'w') as fp:
        for i, l in enumerate(lines):
            TIME = float(l.split()[0])
            if int(TIME) == TIME and int(TIME) % 1 == 0:  # sample by 1 hours
                if i < len(lines) - 1 and TIME == float(lines[i+1].split()[0]):
                    continue
                fp.write(l)