from collections import namedtuple, deque
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = deque(maxlen=CAPACITY)  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        #'''transition = (state, action, state_next, reward)をメモリに保存する'''
        #if len(self.memory) < self.capacity:
        #    self.memory.append(None)
        #self.memory[self.index] = Transition(state, action, state_next, reward)
        #self.index = (self.index + 1) % self.capacity

        self.memory.append(Transition(state, action, state_next, reward))

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)
