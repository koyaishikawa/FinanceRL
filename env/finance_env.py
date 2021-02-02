import gym
import numpy as np


# 離散環境
class FinanceEnv:
    def __init__(self, data, return_data, invest_amount=1000000, cost_rate=0.01):
        self.invest_amount = invest_amount   # 所持金
        self.data = data
        self.return_data = return_data
        self.t = 0
        self.length = data.shape[0]
        self.terminal = False
        self.index = 0                       # closeのカラム
        self.share = 0
        self.share_amount = 0                # 所持株数
        self.prev_action = 0
        self.cum_return = 0
        state_prev_action = 0.3 * (self.prev_action + 2)  # 0.3 0.6 0.9 
        state_share_num = 0.01 + (self.cum_return * self.share_amount / self.invest_amount)  # 0を避ける為に0.01を加える
        self.share_info = np.array([state_prev_action, state_share_num])  # [self.prev_action, 利益]
        self.observation = np.append(self.data[self.t], self.share_info) 
        self.reward = 0
        
        self.cost = self.invest_amount * cost_rate

    def step(self, action):
        if self.prev_action - action == 0:
            self.reward = 0  # 最初と最後にしか利益がないハード環境
            # [0,0]
            if action == 0:
                self.cum_return = 0

            # [1,1],[-1,-1]
            else:
                self.cum_return += self.return_data[self.t]

        else:
            self.reward = self.cum_return * self.share_amount - self.cost * abs(self.prev_action)
            self.reward /= self.invest_amount
            self.cum_return = 0

            if action == 1:
                self.share_amount = (self.invest_amount // self.observation[self.index])
            elif action == -1:
                self.share_amount = (self.invest_amount // self.observation[self.index]) * (-1)
            else:
                self.share_amount = 0

        self.prev_action = action
        self.t += 1
        self.observation = self.data[self.t]
        state_prev_action = 0.3 * (self.prev_action + 2)
        state_share_num = 0.01 + (self.cum_return * self.share_amount / self.invest_amount)
        self.share_info = np.array([state_prev_action, state_share_num])
        if self.t == self.length:
            self.terminal = True
        
        self.state = np.append(self.observation, self.share_info)

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
        state_share_num = 0.01 + (self.cum_return * self.share_amount / self.invest_amount)
        self.share_info = np.array([state_prev_action, state_share_num])
        self.reward = 0
        

        self.state = np.append(self.data[self.t], self.share_info)

        return self.state
