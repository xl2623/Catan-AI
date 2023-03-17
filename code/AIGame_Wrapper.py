from AIGame import catanAIGame
import numpy as np
from heuristicAIPlayer import *
import pygame
from QLearning import GradientQLearning
from exploration import EpsilonGreedyExploration
from tqdm import tqdm

"""
    Catan Game Wrappers
"""

class AIGame():
    def __init__(self):
        # Create game object
        self.special_player_name = '1'
        self.round = 1
        self.player_order = 1
        self.special_player_order = -1
        self.player_name_list = []
        
        self.reset()
    
    def start(self):
        # TODO: Randomize player order        
        self.player_list = self.catan_game.create_player_list(special_placement_type="learning")
        random.shuffle(self.player_list)
        
        player_order = 1
        for player_i in self.player_list:
            # Establish playing order
            player_i.placementOrder = player_order
            if player_i.name == self.special_player_name:
                self.special_player_order = player_order
            self.player_name_list.append(player_i.name)
            
            player_order += 1

        # Play until we reach the agent
        self.player_order = 1
        for player_i in self.player_list:
            if player_i.name == self.special_player_name:
                self.increment_player_order()
                break
            else:
                player_i.initial_setup(self.catan_game.board)
                pygame.event.pump()
                self.increment_player_order()
                
        # Return game state
        return self.get_state()
          
    def increment_player_order(self):
        if self.player_order < 4:
            self.player_order += 1
        else:
            self.player_order = 1
    
    def get_state(self):
        return self.catan_game.tostate_simple(self.special_player_name, self.player_name_list) 
    
    def get_current_action_space(self):
        return self.catan_game.board.get_setup_settlements(None)

    def get_usable_action_space(self):
        # Settlement action space
        usable_action_space = self.get_current_action_space()

        # Assumes actions are in a range
        full_usable_action_space = [idx for idx, action in enumerate(self.full_action_space) if action in usable_action_space]

        # TODO: also get road actionable space
        return full_usable_action_space
    
    def play(self, action):
        action = self.translate_action_to_sim(action)
        
        # Find player & play the player action
        self.player_list[self.special_player_order-1].initial_setup(self.catan_game.board, action)
        pygame.event.pump()

        # Round 1: 1st placement
        if self.round == 1:
            self.round = 2
            
            # Still some initial placements to be done
            if self.player_order != 1:
                for idx, player_i in enumerate(self.player_list):
                    if idx + 1 >= self.player_order:
                        player_i.initial_setup(self.catan_game.board)
                        self.increment_player_order()
                        pygame.event.pump()
            
            # Do the next set of placements
            self.player_list.reverse()
            
            self.player_order = 1
            for player_i in self.player_list:
                if player_i.name == self.special_player_name:
                    self.increment_player_order()
                    break
                else:
                    player_i.initial_setup(self.catan_game.board)
                    self.increment_player_order()
                    pygame.event.pump()
            
            # Return 0 reward and the state
            return self.get_state(), 0.0
        
        # Round 2: 2nd placement
        else:
            # Still some second placements to be done
            if self.player_order != 1:
                for idx, player_i in enumerate(self.player_list):
                    if idx + 1 >= self.player_order:
                        player_i.initial_setup(self.catan_game.board)
                        self.increment_player_order()
                        pygame.event.pump()
        
            # Resource allocation for players
            self.catan_game.allocate_initial_resources(self.player_list)

            # Run thru the game
            try:
                vic_points = self.catan_game.playCatan()
            except:
                # Sometimes the sim crashes
                # Hard to debug since it's literally 1/10,000 times.
                # Just ignore those sims...
                vic_points = -1
            
            return self.get_state(), vic_points   
    
    def translate_action_to_sim(self, action):
        # TODO: need to map this to the appropiate settlement vertex
        return self.catan_game.board.vertex_index_to_pixel_dict[action]
    
    def reset(self):
        self.player_order = 1
        self.round = 1
        self.catan_game = catanAIGame(ifprint=False, ifGUI=False, specialPlayerName=self.special_player_name, selfstart=False)
        self.full_action_space = self.get_current_action_space()

"""
    Gradient Q Learning functions
"""

def compute_Q_NN(theta, s, a):
    # TODO: This may need to change for Catan
    input = np.array(s + [a])

    return predict(theta, input)

def compute_gradQ_NN(theta, s, a):
    # TODO: This may need to change for Catan
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
