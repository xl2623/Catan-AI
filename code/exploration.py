from random import uniform
import numpy as np

class EpsilonGreedyExploration():
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def action(self, model, s, usable_actions):
        if uniform(0,1) < self.epsilon:
            # TODO: Make this selection in here instead of going outside of this
            return model.random_action(s)
        
        return model.A[np.argmax(np.array([model.Q(model.theta,s,a) if a in usable_actions else -np.inf for a in model.A]))[0]]