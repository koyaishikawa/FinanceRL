import gym
from model.agent import Agent
import setting
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model.net import Net, DuelNet, TimeNet

class Environment:

    def __init__(self, Env, data, return_data, Model):
        length = data.shape[0]
        self.data_train = data
        self.return_data_train = return_data
        self.data_trade = self.data_train.copy()
        self.return_data_trade = self.return_data_train.copy()
        self.data_eval = data[8*(length)//10:]
        self.return_data_eval = return_data[8*(length)//10:]

        self.env_random = Env(self.data_train, self.return_data_train)  
        self.env_trade = Env(self.data_trade, self.return_data_trade)  
        self.env_eval = Env(self.data_eval, self.return_data_eval)
        num_states = data.shape[1] + 2 # 特徴量 + [現在の利益 + シェア数]
        num_actions = 3  
        self.agent = Agent(num_states, num_actions, Model)
        self.reward_memory = []
        self.repeat = 30

    def online_run(self):
        writer = SummaryWriter(log_dir="./logs")
        for episode in tqdm(range(setting.NUM_EPISODES)):  # 試行数分繰り返す
            state_trade = self.env_trade.reset()
            state_random = self.env_random.reset()

            self.reward_memory = []
            self.total_reward = 0

            state_trade = torch.from_numpy(state_trade).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            state_trade = torch.unsqueeze(state_trade, 0)  # size 4をsize 1x4に変換

            state_random = torch.from_numpy(state_random).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            state_random = torch.unsqueeze(state_random, 0)  # size 4をsize 1x4に変換

            for step in tqdm(range(self.env_trade.length - 1)):  # 1エピソードのループ
                    
                state_trade, action_trade = self.play(self.env_trade, state_trade, mode="trade")
                state_random, _ = self.play(self.env_random, state_random, mode="random")

                # Experience ReplayでQ関数を更新する
                for _ in range(self.repeat):
                    self.agent.update_q_function()
                
                if action_trade.item() == 0:
                    self.agent.update_trade_q_function()

                if step % setting.update_rate == 0:
                    self.agent.update_target_q_function()

            writer.add_scalar("main/total_reward", self.total_reward, episode)

        writer.close()


    def run(self):
        writer = SummaryWriter(log_dir="./logs")
        for episode in tqdm(range(setting.NUM_EPISODES)):  # 試行数分繰り返す
            state = self.env.reset()  # 環境の初期化

            self.reward_memory = []

            state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

            for step in tqdm(range(self.env.length - 1)):  # 1エピソードのループ
                    
                action = self.agent.get_action(state, episode)
                observation_next, reward, done, _ = self.env.step(action.item())

                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    state_next = None  # 次の状態はないので、Noneを格納

                self.reward_memory.append(reward.item())
                reward = torch.from_numpy(reward).type(torch.FloatTensor)  # 普段は報酬0
                state_next = observation_next  # 観測をそのまま状態とする
                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
                state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換

                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)

                # Experience ReplayでQ関数を更新する
                self.agent.update_q_function()

                # 観測の更新
                state = state_next

                if step % setting.update_rate == 0:
                    self.agent.update_target_q_function()

            writer.add_scalar("main/total_reward", self.total_reward, episode)

        writer.close()
    
    def plot_reward(self):
        plt.plot(self.reward_memory)
        plt.show()

    def evaluation(self):
        observation = self.env_eval.reset()  # 環境の初期化

        self.reward_memory = []
        self.total_reward = 0

        state = observation  # 観測をそのまま状態sとして使用
        state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

        for step in range(self.env_eval.length - 1):  # 1エピソードのループ

            action = self.agent.get_action(state, mode="trade")
            observation_next, reward, done, _ = self.env_eval.step(action.item())

            if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                state_next = None  # 次の状態はないので、Noneを格納

            self.total_reward += reward.item()
            self.reward_memory.append(reward.item())
            state_next = observation_next  # 観測をそのまま状態とする
            state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換

            # 観測の更新
            state = state_next

    def play(self, env, state, mode):
        action = self.agent.get_action(state, mode)
        observation_next, reward, done, _ = env.step(action.item())
        if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
            state_next = None  # 次の状態はないので、Noneを格納

        if mode == 'trade':
            self.total_reward += reward.item()
            self.reward_memory.append(reward.item())

        reward = torch.from_numpy(reward).type(torch.FloatTensor)  # 普段は報酬0
        state_next = observation_next  # 観測をそのまま状態とする
        state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換

        reward = self.utillity_function(reward)

        # メモリに経験を追加
        self.agent.memorize(state, action, state_next, reward)

        # 観測の更新
        state = state_next

        return state, action

    def utillity_function(self, reward):
        f = lambda x: np.log(x+10) - np.log(10)
        reward = f(reward)
        return reward
