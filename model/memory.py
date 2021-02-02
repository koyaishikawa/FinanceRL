from collections import namedtuple, deque
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = deque(maxlen=CAPACITY)
        self.index = 0 

    def push(self, state, action, state_next, reward):
        self.memory.append(Transition(state, action, state_next, reward))

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)
