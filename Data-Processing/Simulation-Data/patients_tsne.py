import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE

base = Path(__file__).parent
data = base / 'observation_241231.csv'
df = pd.read_csv(data)

print(df.head())
