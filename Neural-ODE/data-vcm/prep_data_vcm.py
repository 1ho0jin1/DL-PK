import argparse
import numpy as np
import pandas as pd
from pathlib import Path
 


def main(args):
    # target columns : follows the example for TDM1
    # TODO: use better features specifically for VCM
    columns=["STUD", "DSFQ", "PTNM", "CYCL", "AMT", "TIME", "TFDS", "DV"]
    
    source = pd.read_csv(args.read_path)
    columns_src = list(source.columns)
    prep = pd.DataFrame(columns=columns)
    src = source.to_numpy()
    
    # AMT
    idx = columns_src.index("AMT")
    mask = src[:, idx] == '.'
    src[mask, idx] = 0
    src[:, idx] = src[:, idx].astype(float)
    prep["AMT"] = src[:, idx]
    
    # TIME
    idx = columns_src.index("TIME")
    prep["TIME"] = src[:, idx]
    
    # DV
    idx = columns_src.index("DV")
    mask = src[:, idx] == '.'
    src[mask, idx] = float("NaN")
    src[:,idx] = src[:,idx].astype(float)
    prep["DV"] = src[:, idx]
    
    # STUD & PTNM
    idx = columns_src.index("ID")
    cnt1, cnt2 = 0.0, 0.0
    STUD, PTNM = [], []
    idreg1, idreg2 = None, None
    for id_ in src[:,idx]:
        if id_.startswith('hd_'):
            if idreg1 != id_:
                idreg1 = id_
                cnt1 += 1
            STUD.append(1000.0)
            PTNM.append(cnt1)
        elif id_.startswith('n_'):
            if idreg2 != id_:
                idreg2 = id_
                cnt2 += 1
            STUD.append(2000.0)
            PTNM.append(cnt2)
    prep["STUD"] = STUD
    prep["PTNM"] = PTNM
    
    # DSFQ : set all to 1.0
    prep["DSFQ"] = 3.0
    
    # CYCL
    ptnm_reg = -1
    for i, ptnm in enumerate(prep["PTNM"]):
        if ptnm_reg != ptnm:  # new patient, reset CYCL
            ptnm_reg = ptnm
            CYCL = 0.0
        if prep["AMT"][i] > 0.0:
            CYCL += 1
        prep["CYCL"][i] = CYCL
    
    # TFDS
    prep["TFDS"] = source["TAD"]
    
    
    # write to csv
    prep.to_csv(args.write_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("neural ODE model")
    parser.add_argument('--read_path', type=str, default=r"C:\Users\qkrgh\Jupyter\DL-PK\Neural-ODE\data-vcm\DATA_VCM.csv")
    parser.add_argument('--write_path', type=str, default=r"C:\Users\qkrgh\Jupyter\DL-PK\Neural-ODE\data.csv")
    args = parser.parse_args()
    main(args)