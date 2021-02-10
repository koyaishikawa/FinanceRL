import gym
import numpy as np


# 離散環境
class FinanceEnv:
    def __init__(self, train_data, close_data, return_data, invest_amount=1000000, cost_rate=0.01):
        """
        train_data[np.ndarray] : 学習データ（returnのみ）
        close_data[np.array] : 実際のcloseデータ
        return_data[np.array] : 標準化されたreturnデータ

        """

        self.train_data = train_data
        self.close_data = close_data
        self.return_data = return_data
        self.t = 0
        self.length = train_data.shape[0]
        self.terminal = False
        self.share = 0
        self.prev_action = 0
        self.cum_return = 0
        state_prev_action = 0.3 * (self.prev_action + 2)  # 0.3 0.6 0.9 
        self.share_info = np.array([state_prev_action, self.cum_return])  # [self.prev_action, return]
        self.observation = np.append(self.train_data[self.t], self.share_info) 
        self.reward = 0
        self.cost_rate = cost_rate

        # 可視化用データ（実際に取引した時のデータ）
        self.share_amount = 0                # 所持株数
        self.invest_amount = invest_amount   # 所持金
        self.cost = self.invest_amount * self.cost_rate
        self.profit = 0   # profitは本来の利益を格納する（可視化用）
        self.profit_list = []  # rewardはモデル学習用で標準化されたデータを使う


    def step(self, action):
        self.t += 1
        if self.prev_action - action == 0:
            self.reward = 0
            self.profit = 0
            
            # [0,0]
            if action == 0:
                self.cum_return = 0
                self.profit_start = 0

            # [1,1],[-1,-1]
            else:
                self.cum_return += self.return_data[self.t]

        else:
            self.reward = self.cum_return - self.cost_rate * abs(self.prev_action)
            self.profit = (self.close_data[self.t - 1] - self.profit_start) * self.share_amount - self.cost * abs(self.prev_action)    
            self.profit /= self.invest_amount        

            if action == 1:
                self.share_amount = (self.invest_amount // self.close_data[self.t - 1])
                self.cum_return += self.return_data[self.t]
                self.profit_start = self.close_data[self.t - 1]

            elif action == -1:
                self.share_amount = (self.invest_amount // self.close_data[self.t - 1]) * (-1)
                self.cum_return += self.return_data[self.t]
                self.profit_start = self.close_data[self.t - 1]

            else:
                self.share_amount = 0
                self.cum_return = 0 
                self.profit_start = 0 

        self.prev_action = action
        self.observation = self.train_data[self.t]
        state_prev_action = 0.3 * (self.prev_action + 2)
        self.share_info = np.array([state_prev_action, self.cum_return])
        if self.t == self.length:
            self.terminal = True
        
        self.state = np.append(self.observation, self.share_info)

        self.profit_list.append(self.profit)

        return self.state, np.array(self.reward).reshape(1,1), self.terminal, {}
        
    def reset(self):
        self.t = 0
        self.terminal = False
        self.share = 0
        self.share_num = 0
        self.terminal = False
        self.share = 0
        self.share_amount = 0
        self.cum_return = 0
        self.prev_action = 0
        state_prev_action = 0.3 * (self.prev_action + 2)
        self.share_info = np.array([state_prev_action, self.cum_return])
        self.reward = 0

        self.profit = 0
        self.profit_start = 0
        self.profit_list = []
        
        self.state = np.append(self.train_data[self.t], self.share_info)

        return self.state
