import numpy as np


"""
    Q Learning
"""
class QLearning():
    def __init__(self, S, A, gamma, Q, alpha):
        self.S = S
        self.A = A
        self.gamma = gamma 
        self.Q = Q
        self.alpha = alpha

    def lookahead(self, s, a):
        return self.Q[s,a]

    def update(self, s, a, r, sp):
        self.Q[s,a] = self.Q[s,a] + self.alpha*(r + self.gamma*np.max(self.Q[sp,:] - self.Q[s,a]))

"""
    Gradient Q Learning
"""

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

def scale_gradient(grad, L2_max):
    return np.min([L2_max/np.linalg.norm(grad), 1]) * grad

def compute_beta(s, a):
    return np.array([s, s^2, a, a^2, 1])

def compute_Q(theta, s, a):
    return np.dot(theta, compute_beta(s,a))

def compute_gradQ(theta, s, a):
    return compute_beta(s,a)

def compute_Q_NN(theta, s, a):
    # TODO: This may need to change for Catan
    input = np.array([s,a])

    return predict(theta, input)

def compute_gradQ_NN(theta, s, a):
    # TODO: This may need to change for Catan
    input = np.array([s,a])

    return grad_predict(theta, input)

"""
    Neural Network
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(weights, input):
    layer_1 = np.dot(input, weights)
    layer_2 = sigmoid(layer_1)

    return layer_2

def grad_predict(weights, input):
    dot_prod = np.dot(input, weights)
    return input * np.exp(dot_prod) / np.power(dot_prod + 1, 2)

"""
    Helper functions
"""

def compute_QLearning(infile):
    data = np.loadtxt(infile, delimiter=',', skiprows=1, dtype=int)

    # Q-Learning model
    S = np.arange(0,100,1,dtype=int)
    A = np.arange(0,4,1,dtype=int)
    gamma = 0.95
    Q = np.zeros((np.shape(S)[0], np.shape(A)[0]))
    alpha = 0.2

    model = QLearning(S,A,gamma,Q,alpha)

    for sample in data:
        s,a,r,sp = sample
        s = s-1; a = a-1; sp = sp-1
        model.update(s,a,r,sp)

def compute_GradQLearning(infile):
    data = np.loadtxt(infile, delimiter=',', skiprows=1, dtype=int)

    # Gradient Q-Learning model
    A = np.arange(0,4,1,dtype=int)
    gamma = 0.95
    Q = compute_Q
    gradQ = compute_gradQ
    theta = np.random.rand(5,)
    alpha = 0.2

    model = GradientQLearning(A, gamma, Q, gradQ, theta, alpha)

    for sample in data:
        s,a,r,sp = sample
        s = s-1; a = a-1; sp = sp-1
        model.update(s,a,r,sp)

def compute_GradQLearning_NN(infile):
    data = np.loadtxt(infile, delimiter=',', skiprows=1, dtype=int)

    # Gradient Q-Learning model
    A = np.arange(0,4,1,dtype=int)
    gamma = 0.95
    Q = compute_Q_NN
    gradQ = compute_gradQ_NN
    theta = np.random.rand(2,)
    alpha = 0.2

    model = GradientQLearning(A, gamma, Q, gradQ, theta, alpha)

    for sample in data:
        s,a,r,sp = sample
        s = s-1; a = a-1; sp = sp-1
        model.update(s,a,r,sp)

if __name__=='__main__':
    infile = './data/small.csv'
    outfile = 'small'

    compute_QLearning(infile)
    compute_GradQLearning(infile)
    compute_GradQLearning_NN(infile)