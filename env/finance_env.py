import gym


# 離散環境
class FinanceEnv(gym.Env):
    def __init__(self, data, index, price_name="close"):
        self.data = data
        self.price_name = price_name
        self.t = 0
        self.length = data.shape[0]
        self.index = index
        self.terminal = False
        self.share = 0
        self.share_num = 0
        self.state = self.data.loc[self.t]
        
    def step(self, action):
        if action == 0:
            reward = 0
        
        elif action == 1:
            if self.share_num < 0:
                reward = self.share + self.state[self.price_name]*self.share_num
                self.share = 0
                self.share_num = 0
                 
            else:
                reward = 0
                self.share_num += 1
                self.share  += self.state[self.price_name]
                  
        elif action == -1:
            if self.share_num > 0:
                reward = self.state[self.price_name]*(self.share_num) - self.share
                self.share = 0
                self.share_num = 0
                
            else:
                reward = 0
                self.share_num += -1
                self.share += self.state[self.price_name]
                
        self.t += 1
        self.state = self.data.loc[self.t]
        if self.t == self.length:
            self.terminal = True
                
        return self.state, reward, self.terminal, {}
        
    def reset(self):
        self.t = 0
        self.terminal = False
        self.share = 0
        self.share_num = 0
