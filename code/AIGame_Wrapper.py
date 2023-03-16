from AIGame import catanAIGame
import numpy as np
from heuristicAIPlayer import *
import pygame
from QLearning import GradientQLearning
from exploration import EpsilonGreedyExploration

"""
    Catan Game Wrappers
"""

def play_game(placement_policy):
    # Create game object
    special_player_name = 0
    catan_game = catanAIGame(ifprint=False, ifGUI=False, specialPlayerName=special_player_name, selfstart=False)

    # 1st placement
    player_list = catan_game.create_player_list(special_placement_type="learning")
    random.shuffle(player_list)

    player_order = 1
    for player_i in player_list:
        # Establish playing order
        player_i.placementOrder = player_order
        player_order += 1

        # Place settlement
        if player_i.name == special_player_name:
            player_i.initial_setup(catan_game.board, placement_policy)
        else:
            player_i.initial_setup(catan_game.board)
        pygame.event.pump()
    
    # 2nd placement
    player_list.reverse()

    for player_i in player_list:
        # Place settlement
        if player_i.name == special_player_name:
            player_i.initial_setup(catan_game.board, placement_policy)
        else:
            player_i.initial_setup(catan_game.board)
        pygame.event.pump()
        
    # Resource allocation for players
    catan_game.allocate_initial_resources(player_list)

    # Play the game
    winner, turns = catan_game.playCatan()

    return winner, turns

def simulate(game, model, exploration_policy, k, s=None):
    # TODO: Deal w initial state
    for i in range(0,k):
        # I think we may want to do two playes here
        # Where two playes are equivalent to one game
        a = exploration_policy(model, s)
        sp, r  = game.play(s, a)
        model.update(s, a, r, sp)
        s = sp

"""
    Gradient Q Learning functions
"""

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

def random_placement_policy(board, possibleVertices):
    vertexToBuild = random.choice(list(possibleVertices.keys()))

    return vertexToBuild

def main():
    # Gradient Q-Learning model
    A = np.arange(0,37,1,dtype=int)
    gamma = 0.95
    Q = compute_Q_NN
    gradQ = compute_gradQ_NN
    theta0 = np.random.rand(37,)
    alpha = 0.2

    model = GradientQLearning(A, gamma, Q, gradQ, theta0, alpha)

    # Exploration-Exploipation model
    epsilon = 0.1
    Pi = EpsilonGreedyExploration(epsilon)

    # Simulation
    k = 20      # number of steps to simulate
    simulate(play_game, model, Pi, k)

if __name__ == "__main__":
    placement_policy = random_placement_policy

    winner, turns = play_game(placement_policy)

    print(winner)
