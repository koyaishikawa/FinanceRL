from model.env import Environment
import pandas as pd
from env.finance_env import FinanceEnv


df = pd.read_csv('data/AAPL.csv', index_col=0)
data = df[['close','macd','rsi_30','cci_30','dx_30']].to_numpy()
data = data/data.max(axis=0)

# main クラス
cartpole_env = Environment(FinanceEnv, data)
cartpole_env.run()
cartpole_env.plot_reward()