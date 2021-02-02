from model.env import Environment
import pandas as pd
from env.finance_env import FinanceEnv
from model.net import Net, DuelNet, TimeNet
import matplotlib.pyplot as plt
from model.util import calculate_index


df = pd.read_csv('data/nikkei.csv', index_col=0)
df = df[['Close']]
df['diff'] = df.diff()

for i in range(60):
    df[f'return_{i}'] = df['diff'].shift(i)
df = df.loc[~df.isnull().any(axis=1)]
df /= 1000                         ##########工夫が必要########

df.drop(columns = ['diff'], inplace=True)
data = df.to_numpy()
return_data = df['return_0'].to_numpy()


env = Environment(FinanceEnv, data, return_data, TimeNet)
env.online_run()

calculate_index(env.reward_memory)