from tqdm import tqdm
import numpy as np
import random

from QLearning import GradientQLearning
from exploration import EpsilonGreedyExploration
from AIGame_Wrapper import AIGame

"""
    Gradient Q Learning functions
"""

def compute_Q_NN(theta, s, a):
    input = np.array(s + [a])

    return predict(theta, input)

def compute_gradQ_NN(theta, s, a):
    input = np.array(s + [a])

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
    return input * np.exp(dot_prod) / np.power(np.exp(dot_prod) + 1, 2)


"""
    Helper functions
"""

def random_placement_policy(board, possibleVertices):
    vertexToBuild = random.choice(list(possibleVertices.keys()))

    return vertexToBuild

def simulate(game, model, exploration_policy, k):
    init_theta0 = model.theta
    init_theta = init_theta0
    wins  = 0

    for i in tqdm(range(0,k+1)):
        # print(i)
        # Get initial board state
        s = game.start()
        
        # 1st placement
        usable_actions = game.get_usable_action_space()
        a = exploration_policy.action(model, s, usable_actions)
        sp, r  = game.play(a)
        usable_actions = game.get_usable_action_space()
        model.update(s, a, r, s, usable_actions)
        
        # 2nd placement
        ap = exploration_policy.action(model, sp, usable_actions)
        spp, rp = game.play(ap)
        if rp >= 0:
            model.update(sp, ap, rp, spp, ignore_expected_util=True)
                
        # Reset Catan game
        game.reset()

        wins = wins + 1 if rp == 10 else wins
        # print()

        if i % 10000 == 0:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Wins: ' + str(wins))
            print('Change in theta: ' + str(np.linalg.norm(init_theta - model.theta)))
            print('Exploration rate: ' + str(exploration_policy.epsilon))
            wins = 0
            init_theta = model.theta
            exploration_policy.decay_epsilon()
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    print(init_theta0)
    print(model.theta)
    print(np.linalg.norm(init_theta0 - model.theta))
    print(np.linalg.norm(init_theta0))
    print(np.linalg.norm(model.theta))


def main():
    # Game class
    game = AIGame()

    # TODO: Add roads
    state_space_size = 55
    action_space_full_size = 54
     
    # Gradient Q-Learning model
    A = np.arange(0,action_space_full_size,1,dtype=int)
    gamma = 0.95
    Q = compute_Q_NN
    gradQ = compute_gradQ_NN
    theta0 = np.random.rand(state_space_size + 1,)/100.0
    alpha = 0.2

    model = GradientQLearning(A, gamma, Q, gradQ, theta0, alpha)

    # Exploration-Exploitation model
    epsilon = 1.0
    decay_rate = 0.9
    Pi = EpsilonGreedyExploration(epsilon, decay_rate)

    # Simulation
    k = 100000      # number of games to simulate
    simulate(game, model, Pi, k)

if __name__ == "__main__":
    # placement_policy = random_placement_policy
    # winner, turns = play_game(placement_policy)
    # print(winner)
    
    main()
