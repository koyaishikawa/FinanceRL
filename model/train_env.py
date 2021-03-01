import gym
from model.agent import Agent
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model.net import Net, DuelNet, TimeNet
import plotly.graph_objects as go

class Environment:

    def __init__(self,
                 Env,
                 train_data,
                 close_data,
                 return_data,
                 Model,
                 repeat,
                 num_actions,
                 invest_amount,
                 cost_rate,
                 num_episodes,
                 update_rate,
                 use_GPU,
                 capacity,
                 lr,
                 batch_size,
                 gamma):
        """
        Env : シュミレーション環境
        train_data : 学習データ
        close_data : 終値（標準化なし）
        return_data : 価格差
        Model : 深層学習モデル
        repeat : 一回行動が進行した際に何回エポックを回すか
        """
        # 特徴量 + [現在の利益 + 前回のアクション]
        num_states = train_data.shape[1] + 2 
        length = train_data.shape[0]

        # 長期取引をしてくれやすいエージェントが動く環境
        self.explorer_train_data = train_data
        self.explorer_close_data = close_data
        self.explorer_return_data = return_data
        self.env_explorer = Env(self.explorer_train_data, self.explorer_close_data, self.explorer_return_data, invest_amount, cost_rate) 

        # greedyに動く環境
        self.trade_train_data = train_data
        self.trade_close_data = close_data
        self.trade_return_data = return_data
        self.env_trade = Env(self.trade_train_data, self.trade_close_data, self.trade_return_data, invest_amount, cost_rate)            

        self.agent = Agent(num_states, num_actions, Model, use_GPU, capacity, lr, batch_size, gamma)
        self.repeat = repeat
        self.action_memory = []

        self.num_episodes = num_episodes
        self.update_rate = update_rate

    def online_run(self):
        writer = SummaryWriter(log_dir="./logs")
        for episode in tqdm(range(self.num_episodes)):  # 試行数分繰り返す
            state_explorer = self.env_explorer.reset()
            state_trade = self.env_trade.reset()

            self.action_memory = []
            self.total_reward = 0

            # numpy変数をPyTorchのテンソルに変換
            state_explorer = torch.from_numpy(state_explorer).type(torch.FloatTensor)  
            state_trade = torch.from_numpy(state_trade).type(torch.FloatTensor)           
            
            # size nをsize 1xnに変換
            state_explorer = torch.unsqueeze(state_explorer, 0)  
            state_trade = torch.unsqueeze(state_trade, 0)

            for step in tqdm(range(self.env_trade.length - 1)):
                    
                state_trade, action_trade = self._play(self.env_trade, state_trade, mode="trade")
                state_explorer, _ = self._play(self.env_explorer, state_explorer, mode="explorer")

                # Experience ReplayでQ関数を更新する
                for _ in range(self.repeat):
                    self.agent.update_q_function()
                
                # trade環境が取引していない時にパラメーターを更新する
                if action_trade.item() == 0:
                    self.agent.update_trade_q_function()
                
                # targetfunctionの更新
                if step % self.update_rate == 0:
                    self.agent.update_target_q_function()

            writer.add_scalar("main/total_reward", self.total_reward, episode)

        writer.close()

    
    def plot_reward(self):
        plt.plot(np.cumsum(self.env_trade.profit_list))
        plt.show()


    def _play(self, env, state, mode):
        action = self.agent.get_action(state, mode)
        observation_next, reward, done, _ = env.step(action.item())
        if done:
            state_next = None 

        # trade環境だけアクション履歴を残しておく
        if mode == 'trade':
            self.total_reward += reward.item()
            self.action_memory.append(action.item())

        reward = torch.from_numpy(reward).type(torch.FloatTensor)
        state_next = observation_next
        state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換

        reward = self.utillity_function(reward)
        done = torch.tensor(done).view(1,1)

        # メモリに経験を追加
        self.agent.memorize(state, action, state_next, reward, done)

        # 観測の更新
        state = state_next

        return state, action

    def utillity_function(self, reward):
        #f = lambda x: np.log(x+10) - np.log(10)
        f = lamnda x: x**2 if x > 0 else (-1) * (x**2)
        reward = f(reward)
        return reward

    def load_model(self, path):
        self.agent.load_model(path)

    def _create_excute_list(self, actions):
        prev = 0
        buy = []
        sell = []
        neutral = []
        for i,action in enumerate(actions):
            if prev != action:
                if action == 1:
                    buy.append(i)
                elif action == 0:
                    neutral.append(i)
                else:
                    sell.append(i)

                prev = action
                
        return buy, sell, neutral

    def plot_trade(self):
        buy, sell, neutral = self._create_excute_list(self.action_memory)
        data = self.trade_close_data
        fig = go.Figure(data=[
            go.Scatter(x=list(range(data.shape[0])), y=data, name="price"),
            go.Scatter(x=buy, y=data[buy]-0.1, name='buy',mode='markers',marker_symbol='triangle-up',marker_color="red", marker_size=7),
            go.Scatter(x=sell, y=data[sell]+0.1,name='sell', mode='markers',marker_symbol='triangle-down',marker_color="blue", marker_size=7),
            go.Scatter(x=neutral, y=data[neutral],name='neutral', mode='markers',marker_symbol='x',marker_color="orange", marker_size=7)
        ])
        fig.show()
