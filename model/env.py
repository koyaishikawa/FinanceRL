import gym
from model.agent import Agent
import setting
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
 


class Environment:

    def __init__(self, Env, data):
        length = data.shape[0]
        self.data_train = data[:8*(length)//10]
        self.data_eval = data[8*(length)//10:]

        self.env = Env(self.data_train)  # 実行する課題を設定
        self.env_eval = Env(self.data_eval)
        num_states = 5  # 課題の状態と行動の数を設定
        num_actions = 3  # CartPoleの行動（右に左に押す）の2を取得
        self.agent = Agent(num_states, num_actions)
        self.reward_memory = []


    def run(self):
        for episode in tqdm(range(setting.NUM_EPISODES)):  # 試行数分繰り返す
            observation = self.env.reset()  # 環境の初期化

            self.reward_memory = []
            self.total_reward = 0

            state = observation  # 観測をそのまま状態sとして使用
            state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

            for step in range(self.env.length - 1):  # 1エピソードのループ
                    
                action = self.agent.get_action(state, episode)
                observation_next, reward, done, _ = self.env.step(action.item())

                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    state_next = None  # 次の状態はないので、Noneを格納

                self.total_reward += reward.item()
                self.reward_memory.append(self.total_reward)
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
    
    def plot_reward(self):
        plt.plot(self.reward_memory)
        plt.show()

    def eval(self):
        observation = self.env_eval.reset()  # 環境の初期化

        self.reward_memory = []
        self.total_reward = 0

        state = observation  # 観測をそのまま状態sとして使用
        state = torch.from_numpy(state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

        for step in range(self.env_eval.length - 1):  # 1エピソードのループ

            action = self.agent.get_action(state, episode=1, eval=True)
            observation_next, reward, done, _ = self.env_eval.step(action.item())

            if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                state_next = None  # 次の状態はないので、Noneを格納

            self.total_reward += reward.item()
            self.reward_memory.append(self.total_reward)
            state_next = observation_next  # 観測をそのまま状態とする
            state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
            state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換
            
            # 観測の更新
            state = state_next
