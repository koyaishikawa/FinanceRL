from model.brain import Brain


class Agent:
    def __init__(self, num_states, num_actions, Model, use_GPU, capacity, lr, batch_size, gamma):
        self.brain = Brain(num_states, num_actions, Model, use_GPU, capacity, lr, batch_size, gamma) 

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, mode):
        action = self.brain.decide_action(state, mode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

    def update_trade_q_function(self):
        self.brain.update_trade_q_network()

    def load_model(self, path):
        self.brain.load_model(path)
