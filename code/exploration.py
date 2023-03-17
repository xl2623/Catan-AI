import random
import numpy as np

class EpsilonGreedyExploration():
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def action(self, model, s, usable_actions):
        if random.uniform(0,1) < self.epsilon:
            return random.choice(usable_actions)
        
        return model.A[np.argmax(np.array([model.Q(model.theta,s,a) if a in usable_actions else -np.inf for a in model.A]))]