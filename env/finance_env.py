import gym
import numpy as np


# 離散環境
class FinanceEnv:
    def __init__(self, train_data, close_data, return_data, invest_amount, cost_rate):
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
        self.cum_return = 0

        self.share_info = np.array([self.prev_action, self.cum_return])  # [前回アクション, トレード中の収益]
        self.observation = np.append(self.train_data[self.time], self.share_info) 
        self.reward = 0
        self.cost_rate = cost_rate

        # 可視化用データ（実際に取引した時のデータ）
        self.share_amount = 0                # 所持株数
        self.invest_amount = invest_amount   # 所持金
        self.profit = 0   # profitは本来の利益を格納する（可視化用）
        self.profit_list = []  # rewardはモデル学習用で標準化されたデータを使う


    def step(self, action):
        self.time += 1
        self.done = False

        # アクション変化なし
        if self.prev_action - action == 0:
            self.reward = 0
            self.profit = 0
            
            # no trade
            if action == 0:
                self.cum_return = 0
                self.profit_start = 0

            # during buy or sell 
            else:
                self.reward = self.cum_return * self.prev_action - self.cost_rate * 5 * abs(self.prev_action)
                self.cum_return += self.return_data[self.time]

        # 新しいアクションが執行
        else:
            # reward: model用
            self.reward = self.cum_return * self.prev_action - self.cost_rate * 5 * abs(self.prev_action)
            # profit: 可視化用（実際の取引収益）
            self.profit = (self.close_data[self.time - 1] - self.profit_start) * self.share_amount - (self.cost_rate * self.invest_amount) * abs(self.prev_action)    
            self.profit /= self.invest_amount        

            # buy
            if action == 1:
                self.share_amount = (self.invest_amount // self.close_data[self.time - 1])
                self.cum_return += self.return_data[self.time]
                self.profit_start = self.close_data[self.time - 1]

            # sell
            elif action == -1:
                self.share_amount = (self.invest_amount // self.close_data[self.time - 1]) * (-1)
                self.cum_return += self.return_data[self.time]
                self.profit_start = self.close_data[self.time - 1]

            # excute(決済)
            else:
                self.done = True
                self.share_amount = 0
                self.cum_return = 0 
                self.profit_start = 0 

        self.prev_action = action 
        self.observation = self.train_data[self.time]
        self.share_info = np.array([self.prev_action, self.cum_return])

        if self.time == self.length:
            self.done = True
        
        self.state = np.append(self.observation, self.share_info)
        self.profit_list.append(self.profit)

        return self.state, np.array(self.reward).reshape(1,1), self.done, {}
        
    def reset(self):
        self.time = 0
        self.share = 0
        self.share_num = 0
        self.done = False
        self.share = 0
        self.share_amount = 0
        self.cum_return = 0
        self.prev_action = 0
        self.share_info = np.array([self.prev_action, self.cum_return])
        self.reward = 0

        self.profit = 0
        self.profit_start = 0
        self.profit_list = []
        
        self.state = np.append(self.train_data[self.time], self.share_info)

        return self.state
