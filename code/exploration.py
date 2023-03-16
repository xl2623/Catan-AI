from random import uniform
import numpy as np

class EpsilonGreedyExploration():
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def action(self, model, s):
        if uniform(0,1) < self.epsilon:
            # Need to check available action space
            return model.random_action(s)
        
        return model.A[np.argmax(np.array([model.Q(s,a) for a in model.A]))]