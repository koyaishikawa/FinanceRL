from model.memory import ReplayMemory
from model.util import Transition
# from model.net import Net, DuelNet, TimeNet
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class Brain:
    def __init__(self, num_states, num_actions, Model, use_GPU, capacity, lr, batch_size, gamma):
        self.num_actions = num_actions 
        self.memory = ReplayMemory(capacity)
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma

        if use_GPU:
            self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.dev = torch.device('cpu')
        print(f'run type:{self.dev}')

        n_in, n_mid, n_out = num_states, 120, num_actions
        self.main_q_network = Model(n_in, n_mid, n_out).to(self.dev)
        self.target_q_network = Model(n_in, n_mid, n_out).to(self.dev)
        self.trade_q_network = Model(n_in, n_mid, n_out).to(self.dev)
        print(self.main_q_network)

        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=lr)
            
    def replay(self):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        if len(self.memory) < self.batch_size:
            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states, self.done_batch = self.make_minibatch()

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()

    def decide_action(self, state, mode, alpha=1):
        '''現在の状態に応じて、行動を決定する'''
        if mode == "trade":
            self.trade_q_network.eval()
            with torch.no_grad():
                action = self.trade_q_network(state.to(self.dev)).max(1)[1].view(1, 1)
            return action - 1

        elif mode == 'explorer':
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]).to(self.dev)
            return action - 1

        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]).to(self.dev)
            return action - 1

    def make_minibatch(self):
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.dev)
        action_batch = torch.cat(batch.action).to(self.dev)
        reward_batch = torch.cat(batch.reward).to(self.dev)
        done_batch = torch.cat(batch.done).to(self.dev)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states, done_batch

    def get_expected_state_action_values(self):

        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(
            self.state_batch.to(self.dev)).gather(1, self.action_batch + 1)

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        next_state_values = torch.zeros(self.batch_size).to(self.dev)

        a_m = torch.zeros(self.batch_size).type(torch.LongTensor).to(self.dev)

        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states.to(self.dev)).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states.to(self.dev)).gather(1, self.action_batch + 1).detach().squeeze()

        expected_state_action_values = self.reward_batch +  (~self.done_batch) * self.gamma * next_state_values.unsqueeze(1)
        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 結合パラメータの更新'''

        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values)
        #print(f'loss:{loss}')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        '''Target Q-NetworkをMainと同じにする'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_trade_q_network(self):
        '''Target Q-NetworkをMainと同じにする'''
        self.trade_q_network.load_state_dict(self.main_q_network.state_dict())

    def load_model(self, path):
        self.main_q_network.load_state_dict(torch.load(path))
        self.trade_q_network .load_state_dict(torch.load(path))
        self.target_q_network .load_state_dict(torch.load(path))
