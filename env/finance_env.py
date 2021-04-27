import gym
import numpy as np


# 離散環境
class FinanceEnv:
    def __init__(self, train_data, close_data, return_data, max_price, cost_rate):
        """
        train_data[np.ndarray] : 学習データ
        close_data[np.array] : 実際のcloseデータ 利益計算用
        return_data[np.array] : 標準化されたreturnデータ 報酬関数用
        invest_amount[int] : 仮想的に取引をした際の推移を計算する
        cost_rate[float] : 購入金額の何パーセント分のコストがかかるのか

        """

        self.train_data = train_data
        self.close_data = close_data
        self.return_data = return_data
        self.time = 0
        self.length = train_data.shape[0]
        self.done = False
        self.share = 0
        self.prev_action = 0

        self.share_info = np.array([self.prev_action])  # [前回アクション]
        self.observation = np.append(self.train_data[self.time], self.share_info) 
        self.reward = 0
        self.cost = cost_rate / max_price

        self.profit_list = []  # rewardはモデル学習用で標準化されたデータを使う


    def step(self, action):
        self.time += 1
        self.done = False

        self.reward = self.return_data[self.time] * action - self.cost * abs((action - self.prev_action))

        if action == 1:
            self.done = True

        self.prev_action = action 
        self.observation = self.train_data[self.time]
        self.share_info = np.array([self.prev_action])

        if self.time == self.length:
            self.done = True
        
        self.state = np.append(self.observation, self.share_info)
        self.profit_list.append(self.reward)
        return self.state, np.array(self.reward).reshape(1,1), self.done, {}
        
    def reset(self):
        self.time = 0
        self.done = False
        self.prev_action = 0
        self.share_info = np.array([self.prev_action])
        self.reward = 0

        self.profit_list = []
        
        self.state = np.append(self.train_data[self.time], self.share_info)

        return self.state
