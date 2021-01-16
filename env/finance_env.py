import gym
import numpy as np


# 離散環境
class FinanceEnv(gym.Env):
    def __init__(self, data, index=0):
        self.data = data
        self.t = 0
        self.length = data.shape[0]
        self.index = index
        self.terminal = False
        self.share = 0
        self.share_num = 0
        self.state = self.data[self.t]
        self.reward = 0
        
    def step(self, action):
        if action == 0:
            self.reward = 0
        
        elif action == 1: #buy
            if self.share_num < 0:
                self.reward = self.share + self.state[self.index]*self.share_num
                self.share = 0
                self.share_num = 0
                 
            else:
                self.reward = 0
                self.share_num += 1
                self.share  += self.state[self.index]
                  
        elif action == 2: #sell
            if self.share_num > 0:
                self.reward = self.state[self.index]*(self.share_num) - self.share
                self.share = 0
                self.share_num = 0
                
            else:
                self.reward = 0
                self.share_num += -1
                self.share += self.state[self.index]
                
        self.t += 1
        self.state = self.data[self.t]
        if self.t == self.length:
            self.terminal = True
                
        return self.state, np.array(self.reward).reshape(1,1), self.terminal, {}
        
    def reset(self):
        self.t = 0
        self.terminal = False
        self.share = 0
        self.share_num = 0

        return self.data[self.t]
