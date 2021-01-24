from model.env import Environment
import pandas as pd
from env.finance_env import FinanceEnv


df = pd.read_csv('data/AAPL.csv', index_col=0)

return_data = df['close'].diff().to_numpy()
data = df[['close','macd','rsi_30','cci_30','dx_30']].to_numpy()
data = data/data.max(axis=0)
# data['close']は一番最初のカラムにする

# main クラス
env = Environment(FinanceEnv, data, return_data)

env.run()
#env.plot_reward()

env.evaluation()
env.plot_reward()