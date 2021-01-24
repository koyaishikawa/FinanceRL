from model.memory import ReplayMemory
from model.util import Transition
# from model.net import Net, DuelNet, TimeNet
import torch
from torch import optim
import torch.nn.functional as F
import setting
import random
import numpy as np



class Brain:
    def __init__(self, num_states, num_actions, Model):
        self.num_actions = num_actions 
        self.memory = ReplayMemory(setting.CAPACITY)

        #self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dev = torch.device('cpu')

        n_in, n_mid, n_out = num_states, 10, num_actions
        self.main_q_network = Model(n_in, n_mid, n_out).to(self.dev) # Netクラスを使用
        self.target_q_network = Model(n_in, n_mid, n_out).to(self.dev)  # Netクラスを使用
        print(self.main_q_network)  # ネットワークの形を出力

        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=setting.lr)
            
    def replay(self):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        if len(self.memory) < setting.BATCH_SIZE:
            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()

    def decide_action(self, state, episode, evl):
        '''現在の状態に応じて、行動を決定する'''
        if evl:
            self.target_q_network.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.target_q_network(state.to(self.dev)).max(1)[1].view(1, 1)
            return action - 1

        else:
            epsilon = 0.5 * (1 / (episode//5 + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.main_q_network(state.to(self.dev)).max(1)[1].view(1, 1)

        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]).to(self.dev)  # 0,1の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action - 1

    def make_minibatch(self):
        '''2. ミニバッチの作成'''

        # 2.1 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(setting.BATCH_SIZE)

        # 2.2 各変数をミニバッチに対応する形に変形
        # transitionsは1stepごとの(state, action, state_next, reward)が、BATCH_SIZE分格納されている
        # つまり、(state, action, state_next, reward)×BATCH_SIZE
        # これをミニバッチにしたい。つまり
        # (state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))

        # 2.3 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がBATCH_SIZE分並んでいるのですが、
        # それを torch.FloatTensor of size BATCH_SIZEx4 に変換します
        # 状態、行動、報酬、non_finalの状態のミニバッチのVariableを作成
        # catはConcatenates（結合）のことです。
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).to(self.dev)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''3. 教師信号となるQ(s_t, a_t)値を求める'''

        # 3.1 ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t)を求める
        # self.model(state_batch)は、右左の両方のQ値を出力しており
        # [torch.FloatTensor of size BATCH_SIZEx2]になっている。
        # ここから実行したアクションa_tに対応するQ値を求めるため、action_batchで行った行動a_tが右か左かのindexを求め
        # それに対応するQ値をgatherでひっぱり出す。
        self.state_action_values = self.main_q_network(
            self.state_batch.to(self.dev)).gather(1, self.action_batch + 1)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意。

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        # まずは全部0にしておく
        next_state_values = torch.zeros(setting.BATCH_SIZE).to(self.dev)

        a_m = torch.zeros(setting.BATCH_SIZE).type(torch.LongTensor).to(self.dev)

        # 次の状態での最大Q値の行動a_mをMain Q-Networkから求める
        # 最後の[1]で行動に対応したindexが返る
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states.to(self.dev)).detach().max(1)[1]

        # 次の状態があるものだけにフィルターし、size 32を32×1へ
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態があるindexの、行動a_mのQ値をtarget Q-Networkから求める
        # detach()で取り出す
        # squeeze()でsize[minibatch×1]を[minibatch]に。
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states.to(self.dev)).gather(1, a_m_non_final_next_states.to(self.dev)).detach().squeeze()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = self.reward_batch + setting.GAMMA * next_state_values.unsqueeze(1)
        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 結合パラメータの更新'''

        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values)

        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # ここおおおおおおおおおおおおおおおお

    def update_target_q_network(self):  # DDQNで追加
        '''Target Q-NetworkをMainと同じにする'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())
