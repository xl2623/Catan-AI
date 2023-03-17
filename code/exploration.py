import random
import numpy as np

class EpsilonGreedyExploration():
    def __init__(self, epsilon, decay_rate = 1.0):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
    
    def action(self, model, s, usable_actions):
        if random.uniform(0,1) < self.epsilon:
            action = random.choice(usable_actions)
        else:
            action = model.A[np.argmax(np.array([model.Q(model.theta,s,a) if a in usable_actions else -np.inf for a in model.A]))]

        return action
    
    def decay_epsilon(self):
        self.epsilon = self.decay_rate * self.epsilon