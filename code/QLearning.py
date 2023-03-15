import numpy as np

def scale_gradient(grad, L2_max):
    return np.min([L2_max/np.linalg.norm(grad), 1]) * grad

class GradientQLearning():
    def __init__(self, A, gamma, Q, gradQ, theta, alpha):
        self.A = A
        self.gamma = gamma 
        self.Q = Q
        self.gradQ = gradQ
        self.theta = theta
        self.alpha = alpha

    def lookahead(self, s, a):
        return self.Q(s,a)

    def update(self, s, a, r, sp):
        u = np.max([self.Q(self.theta, sp, ap) for ap in self.A])
        grad = (r + self.gamma*u - self.Q(self.theta,s,a))*self.gradQ(self.theta, s, a)
        self.theta += self.alpha*scale_gradient(grad, 1)