import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

base = '/home/hj/DL-PK/Experiments/runs/inference'
models = ['gru_250219_aug', 'gru_250219_noaug', 'node_250219_aug', 'node_250219_noaug']
dosage = ['0000', '0101', '0202', '0303', '0404',\
        '1010', '1111', '1212', '1313', '1414',\
        '2020', '2121', '2222', '2323', '2424',\
        '3030', '3131', '3232', '3333', '3434',\
        '4040', '4141', '4242', '4343', '4444']


results = np.zeros((len(models), len(dosage)))
for i,m in enumerate(models):
    for j,d in enumerate(dosage):
        path = f"{base}/{m}/{d}/test/loss.txt"
        with open(path, 'r') as fp:
            lines = fp.readlines()
        mae = float(lines[0].strip().split()[-1])
        results[i,j] = mae

sns.set()
# heatmap of MAE
plt.figure(figsize=(32,4))
sns.heatmap(results, annot=True, xticklabels=dosage, yticklabels=models, cmap='viridis')
plt.xlabel('Dosage')
plt.ylabel('Model')
plt.title('MAE of different models on different dosages')
plt.show()
plt.savefig('mae_heatmap.png', bbox_inches='tight')



results = np.zeros((len(models), len(dosage)))
for i,m in enumerate(models):
    for j,d in enumerate(dosage):
        path = f"{base}/{m}/{d}/test/loss.txt"
        with open(path, 'r') as fp:
            lines = fp.readlines()
        mse = float(lines[1].strip().split()[-1])
        results[i,j] = mse

sns.set()
# heatmap of MSE
plt.figure(figsize=(32,4))
sns.heatmap(results, annot=True, xticklabels=dosage, yticklabels=models, cmap='viridis')
plt.xlabel('Dosage')
plt.ylabel('Model')
plt.title('MSE of different models on different dosages')
plt.show()
plt.savefig('mse_heatmap.png', bbox_inches='tight')