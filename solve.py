from model.env import Environment
import pandas as pd
from env.finance_env import FinanceEnv
from model.net import Net, DuelNet, TimeNet
import matplotlib.pyplot as plt
from model.util import calculate_index, EWMA

import torch
torch.manual_seed(0)

df = pd.read_csv('data/nikkei.csv', index_col=0)
df = df[['Close']]

ema, var = EWMA(df['Close'].values)
tmp = df[['Close']]
tmp['mean'] = ema
tmp['var'] = var
tmp['std'] = tmp['var']**(1/2)
tmp = tmp.loc[tmp['mean'] != 0]
normal = (tmp['Close'] - tmp['mean'])/tmp['std']

df['diff'] = normal.diff()

for i in range(60):
    df[f'return_{i}'] = df['diff'].shift(i)
df = df.loc[~df.isnull().any(axis=1)]
    
df.drop(columns = ['diff'], inplace=True)
df = df.head(5000)
data = df.to_numpy()
return_data = df['return_0'].to_numpy()
close_data = df['Close'].to_numpy()
train_data = df.iloc[:,1:].to_numpy()

env =Environment(FinanceEnv, train_data, close_data, return_data, TimeNet, repeat=30)
env.online_run()