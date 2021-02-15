from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


def calculate_index(reward_memory):
    reward_memory = np.array(reward_memory)
    # 平均利益
    return_mean = reward_memory[reward_memory != 0].mean()

    # 標準偏差
    return_std = reward_memory[reward_memory != 0].std()

    # DD_mean
    dd_mean = reward_memory[reward_memory < 0].mean()

    # DD_std
    dd_std = reward_memory[reward_memory < 0].std()

    # シャープレシオ
    sharp_ratio = return_mean / return_std

    # MDD
    mdd = reward_memory.min()

    # 勝率
    win_rate = (reward_memory > 0).sum() / (reward_memory < 0).sum()

    # トレード回数
    trade_num = len(reward_memory[reward_memory != 0])

    # 平均トレードステップ数

    # プロット用
    reward_plot = reward_memory.cumsum()
    plt.xlabel('step')
    plt.ylabel('profit')
    plt.plot(reward_plot)

    print(f'return_mean:{return_mean}\nreturn_std:{return_std} \n DD_mean:{dd_mean} \n DD_std:{dd_std} \n sharp_ratio:{sharp_ratio} \n MDD:{mdd} \n win_rate{win_rate} \n Trade_Num:{trade_num}')

def EWMA(data, windows=60):
    ema_mean = np.zeros(data.shape[0])
    ema_var = np.zeros(data.shape[0])
    alpha = 2 / (windows + 1)
    prev_ema = ema_mean[:windows].mean()
    prev_var = ema_mean[:windows].var()
    ema_mean[windows-1] = prev_ema
    ema_var[windows-1] = prev_var
    for idx in range(windows, data.shape[0]):
        delta = data[idx] - prev_ema
        present_ema = prev_ema + alpha * delta
        present_var = (1-alpha)*(prev_var) + alpha * (delta ** 2)
        ema_mean[idx] = present_ema
        ema_var[idx] = present_var
        prev_ema = present_ema
        prev_var = present_var
    return ema_mean, ema_var

def make_env_data(data):
    """
    data[pd.DataFrame] : closeのみ

    --> train_data, close_data, return_data　[np.ndarray]
        normalized              normalized

    """
    ema_mean, ema_var = EWMA(data.values)
    train_data = data.copy()
    train_data['mean'] = ema_mean
    train_data['var'] = ema_var
    train_data['std'] = train_data['var']**(1/2)
    train_data = train_data.loc[train_data['mean'] != 0]

    # normalized
    normal = (train_data['Close'] - train_data['mean']) / train_data['std']
    train_data['diff'] = normal.diff()

    for i in range(60):
        train_data[f'return_{i}'] = train_data['diff'].shift(i)
    train_data = train_data.loc[~train_data.isnull().any(axis=1)]
        
    train_data.drop(columns = ['diff', 'mean', 'var', 'std'], inplace=True)
    return_data = train_data['return_0'].to_numpy()
    close_data = train_data['Close'].to_numpy()
    train_data = train_data.iloc[:,1:].to_numpy()

    return train_data, close_data, return_data
