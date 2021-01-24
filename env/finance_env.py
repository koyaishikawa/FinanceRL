import gym
import numpy as np


# 離散環境
class FinanceEnv(gym.Env):
    def __init__(self, data, return_data, invest_amount=100):
        self.invest_amount = invest_amount
        self.data = data
        self.return_data = return_data
        self.t = 0
        self.length = data.shape[0]
        self.terminal = False
        self.index = 0  # closeのカラム
        self.share = 0
        self.share_amount = 0
        self.share_info = np.array([0,0])  # [self.prev_action, 利益]
        self.observation = np.append(self.data[self.t], self.share_info) 
        self.reward = 0
        self.prev_action = 0
        self.cost = 0
        self.cum_return = 0

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
            self.reward = self.cum_return * self.share_amount + self.cost * abs(self.prev_action - action)
            self.reward /= 1000
            self.cum_return = 0

            if action == 1:
                self.share_amount = (self.invest_amount / self.observation[self.index])
            elif action == -1:
                self.share_amount = (self.invest_amount / self.observation[self.index]) * (-1)
            else:
                self.share_amount = 0

        self.prev_action = action
        self.t += 1
        self.observation = self.data[self.t]
        self.share_info = np.array([self.prev_action, self.cum_return * self.share_amount / 1000])
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
        self.share_info = np.array([0,0])  # [self.prev_action, 利益]
        self.observation = np.append(self.data[self.t], self.share_info) 
        self.reward = 0
        self.prev_action = 0

        self.state = np.append(self.data[self.t], self.share_info)

        return self.state
