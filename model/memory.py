from collections import namedtuple
import random
from .sumtree import SumTree
import numpy as np

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)


class PrioritizedExperienceReplayBuffer:
    def __init__(self,
                 batch_size,
                 capacity,
                 epsilon=0.0001,
                 alpha=0.6,
                 beta0=0.5,
                 n_iteration=5000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta0 = beta0
        self.iterations = n_iteration
        self.length = 0

    def _get_Priority(self, td_error):
        return (abs(td_error) + self.epsilon) ** self.alpha

    def put(self, state, action, reward, next_state, done):
        self.length += 1
        transition = [state, action, reward, next_state, done]
        priority = self.tree.max()
        if priority <= 0:
            priority = 1
        self.tree.add(priority, transition)

    def sample(self):
        sample = []
        indexes = []
        for rand in np.random.uniform(0, self.tree.total(), self.batch_size):
            index, _, data = self.tree.get(rand)
            sample.append(data)
            indexes.append(index)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        return states, actions, rewards, next_states, done, indexes

    def importance_sample(self, episode):
        sample = []
        indexes = []
        weights = np.empty(self.length, dtype='float32')
        total = self.tree.total()
        beta = self.beta0 + (1 - self.beta0) * episode / self.iterations
        for i, rand in enumerate(np.random.uniform(0, total, self.batch_size)):
            index, priority, data = self.tree.get(rand)
            sample.append(data)
            indexes.append(index)
            weights[i] = (self.capacity * priority / total) ** (-beta)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        weights = weights / weights.max()
        return states, actions, rewards, next_states, done, indexes, weights

    def update(self, idx, td_error):
        priority = self._get_Priority(td_error)
        self.tree.update(idx, priority)

    def size(self):
        return self.length

