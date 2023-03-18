from AIGame import catanAIGame
import pygame
import random
from board import *
import random
from gameView import *
from player import *
from heuristicAIPlayer import *
import signal

"""
    Timeout class
"""

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

"""
    Catan Game Wrappers
"""

class AIGame():
    def __init__(self, specialPlayerName='1', other_player_type="heuristic"):
        # Create game object
        self.special_player_name = specialPlayerName
        self.round = 1
        self.player_order = 1
        self.special_player_order = -1
        self.player_name_list = []
        self.other_player_type = other_player_type
        
        self.reset()
    
    def start(self):
        self.player_list = self.catan_game.create_player_list(special_placement_type="learning", other_player_type=self.other_player_type)
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
        # get all possible vertices that you can build, in pixel coordinate
        possibleVertices = self.catan_game.board.get_setup_settlements(self)
        # get mapping from pixal coord to vertex indeices
        pixel_to_vertex_index_dict = dict((v, k) for k, v in self.catan_game.board.vertex_index_to_pixel_dict.items())
        # convert possibleVertices from pixel coord to indices
        possibleVertices_index = [pixel_to_vertex_index_dict[key] for key in pixel_to_vertex_index_dict if key in possibleVertices]
        # edge bank
        edgeBank = [(Point(x=580.0, y=400.0), Point(x=540.0, y=330.72)), (Point(x=580.0, y=400.0), Point(x=540.0, y=469.28)), 
                    (Point(x=580.0, y=400.0), Point(x=660.0, y=400.0)), (Point(x=540.0, y=330.72), Point(x=460.0, y=330.72)), 
                    (Point(x=540.0, y=330.72), Point(x=580.0, y=261.44)), (Point(x=460.0, y=330.72), Point(x=420.0, y=400.0)), 
                    (Point(x=460.0, y=330.72), Point(x=420.0, y=261.44)), (Point(x=420.0, y=400.0), Point(x=460.0, y=469.28)), 
                    (Point(x=420.0, y=400.0), Point(x=340.0, y=400.0)), (Point(x=460.0, y=469.28), Point(x=540.0, y=469.28)), 
                    (Point(x=460.0, y=469.28), Point(x=420.0, y=538.56)), (Point(x=540.0, y=469.28), Point(x=580.0, y=538.56)), 
                    (Point(x=580.0, y=261.44), Point(x=540.0, y=192.15)), (Point(x=580.0, y=261.44), Point(x=660.0, y=261.44)), 
                    (Point(x=540.0, y=192.15), Point(x=460.0, y=192.15)), (Point(x=540.0, y=192.15), Point(x=580.0, y=122.87)), 
                    (Point(x=460.0, y=192.15), Point(x=420.0, y=261.44)), (Point(x=460.0, y=192.15), Point(x=420.0, y=122.87)), 
                    (Point(x=420.0, y=261.44), Point(x=340.0, y=261.44)), (Point(x=700.0, y=330.72), Point(x=660.0, y=261.44)), 
                    (Point(x=700.0, y=330.72), Point(x=660.0, y=400.0)), (Point(x=700.0, y=330.72), Point(x=780.0, y=330.72)), 
                    (Point(x=660.0, y=261.44), Point(x=700.0, y=192.15)), (Point(x=660.0, y=400.0), Point(x=700.0, y=469.28)), 
                    (Point(x=700.0, y=469.28), Point(x=660.0, y=538.56)), (Point(x=700.0, y=469.28), Point(x=780.0, y=469.28)), 
                    (Point(x=580.0, y=538.56), Point(x=660.0, y=538.56)), (Point(x=580.0, y=538.56), Point(x=540.0, y=607.85)), 
                    (Point(x=660.0, y=538.56), Point(x=700.0, y=607.85)), (Point(x=420.0, y=538.56), Point(x=460.0, y=607.85)), 
                    (Point(x=420.0, y=538.56), Point(x=340.0, y=538.56)), (Point(x=460.0, y=607.85), Point(x=540.0, y=607.85)), 
                    (Point(x=460.0, y=607.85), Point(x=420.0, y=677.13)), (Point(x=540.0, y=607.85), Point(x=580.0, y=677.13)), 
                    (Point(x=340.0, y=400.0), Point(x=300.0, y=469.28)), (Point(x=340.0, y=400.0), Point(x=300.0, y=330.72)), 
                    (Point(x=300.0, y=469.28), Point(x=340.0, y=538.56)), (Point(x=300.0, y=469.28), Point(x=220.0, y=469.28)), 
                    (Point(x=340.0, y=538.56), Point(x=300.0, y=607.85)), (Point(x=340.0, y=261.44), Point(x=300.0, y=330.72)), 
                    (Point(x=340.0, y=261.44), Point(x=300.0, y=192.15)), (Point(x=300.0, y=330.72), Point(x=220.0, y=330.72)), 
                    (Point(x=580.0, y=122.87), Point(x=540.0, y=53.59)), (Point(x=580.0, y=122.87), Point(x=660.0, y=122.87)), 
                    (Point(x=540.0, y=53.59), Point(x=460.0, y=53.59)), (Point(x=460.0, y=53.59), Point(x=420.0, y=122.87)), 
                    (Point(x=420.0, y=122.87), Point(x=340.0, y=122.87)), (Point(x=700.0, y=192.15), Point(x=660.0, y=122.87)), 
                    (Point(x=700.0, y=192.15), Point(x=780.0, y=192.15)), (Point(x=820.0, y=261.44), Point(x=780.0, y=192.15)), 
                    (Point(x=820.0, y=261.44), Point(x=780.0, y=330.72)), (Point(x=780.0, y=330.72), Point(x=820.0, y=400.0)), 
                    (Point(x=820.0, y=400.0), Point(x=780.0, y=469.28)), (Point(x=780.0, y=469.28), Point(x=820.0, y=538.56)), 
                    (Point(x=820.0, y=538.56), Point(x=780.0, y=607.85)), (Point(x=700.0, y=607.85), Point(x=780.0, y=607.85)), 
                    (Point(x=700.0, y=607.85), Point(x=660.0, y=677.13)), (Point(x=580.0, y=677.13), Point(x=660.0, y=677.13)), 
                    (Point(x=580.0, y=677.13), Point(x=540.0, y=746.41)), (Point(x=420.0, y=677.13), Point(x=460.0, y=746.41)), 
                    (Point(x=420.0, y=677.13), Point(x=340.0, y=677.13)), (Point(x=460.0, y=746.41), Point(x=540.0, y=746.41)), 
                    (Point(x=300.0, y=607.85), Point(x=340.0, y=677.13)), (Point(x=300.0, y=607.85), Point(x=220.0, y=607.85)), 
                    (Point(x=220.0, y=469.28), Point(x=180.0, y=538.56)), (Point(x=220.0, y=469.28), Point(x=180.0, y=400.0)), 
                    (Point(x=180.0, y=538.56), Point(x=220.0, y=607.85)), (Point(x=220.0, y=330.72), Point(x=180.0, y=400.0)), 
                    (Point(x=220.0, y=330.72), Point(x=180.0, y=261.44)), (Point(x=300.0, y=192.15), Point(x=220.0, y=192.15)), 
                    (Point(x=300.0, y=192.15), Point(x=340.0, y=122.87)), (Point(x=220.0, y=192.15), Point(x=180.0, y=261.44))]
                    # start from the 0 index of state
        # get all possible edges that you can build assoicated with each vertex        
        possibleRoads = []
        for j in range(0,len(possibleVertices_index)):
            currVertex = self.catan_game.board.boardGraph[[self.catan_game.board.vertex_index_to_pixel_dict[vertex_index] for vertex_index in possibleVertices_index][j]]
            v_neighbor = currVertex.edgeList
            my_pixelcoor = currVertex.pixelCoordinates
            listofroads = [(my_pixelcoor, one_of_the_neighbor) for one_of_the_neighbor in v_neighbor]
            curr_possible_road_index = []
            for i in range(0, len(listofroads)):
                try:
                    curr_possible_road_index.append(edgeBank.index((listofroads[i][0], listofroads[i][1])))
                except:
                    curr_possible_road_index.append(edgeBank.index((listofroads[i][1], listofroads[i][0])))
            
            possibleRoads.append(curr_possible_road_index)

        # possibleRoads: stores at index k, all the possible edge index that you can build a road, at possibleVertices_index[k] in a list
        # possibleVertices_index: store all the possible vertices that you can build a settlement
        
        # assemble action space
        full_usable_action_space = []
        for i in range(0, len(possibleVertices_index)):
            for road in possibleRoads[i]:
                curr_action = []
                curr_action.append(possibleVertices_index[i])
                curr_action.append(road)
                full_usable_action_space.append(curr_action)

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
            except Exception as e:
                # Sometimes the sim crashes
                # Hard to debug since it's literally 1/10,000 times.
                # Just ignore those sims...
                # print(e)
                vic_points = -1
            
            return self.get_state(), vic_points-10 
    
    def translate_action_to_sim(self, action):
        # sim directly take action, as long as special agent is set to learning
        return action
    
    def reset(self):
        self.player_order = 1
        self.round = 1
        self.catan_game = catanAIGame(ifprint=False, ifGUI=False, specialPlayerName=self.special_player_name, selfstart=False)
        self.full_action_space = self.get_current_action_space()

"""
    Run game with given policy
    Return relevant states, actions and rewards
"""

def play_game_with_policy(placement_policy, other_player_type="heuristic"):
    try:
        with timeout(10):
            game      = AIGame(specialPlayerName="1", other_player_type=other_player_type)

            s         = game.start()
            # 1st placement
            usable_a  = game.get_usable_action_space() # returns a list of size [# of possible action], each element of the list is a two-element list [a1, a2]
            a         = placement_policy(s, usable_a) # assume a is list that looks like [a1, a2] 
            sp, r     = game.play(a)

            # 2nd placement
            usable_ap = game.get_usable_action_space()
            ap        = placement_policy(sp, usable_ap)
            spp, rp   = game.play(ap)

            # Note if rp=-11, something went wrong and you should just skip this round

            return s, usable_a, a, r, sp, usable_ap, ap, rp, spp
    except:
        return [], [], [], [], [], [], [], -11, []

def example():
    for iter in range(1000):
        game      = AIGame()
        s         = game.start()
        # 1st placement
        usable_a  = game.get_usable_action_space() # returns a list of size [# of possible action], each element of the list is a two-element list [a1, a2]
        sp, r     = game.play(random.choice(usable_a))

        # 2nd placement
        usable_ap = game.get_usable_action_space()
        spp, rp   = game.play(random.choice(usable_ap))

        print(s)
        print(r)
        print(sp)
        print(rp)
        print(spp)
   
if __name__=="__main__": 
    game      = AIGame()
    s         = game.start()
    print(game.player_list[0].name)
    print(game.get_usable_action_space())
    print(len(game.get_usable_action_space()))