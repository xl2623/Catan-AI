from AIGame import catanAIGame
import pygame
import random


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
            except Exception as e:
                # Sometimes the sim crashes
                # Hard to debug since it's literally 1/10,000 times.
                # Just ignore those sims...
                # print(e)
                vic_points = -1
            
            return self.get_state(), vic_points-10   
    
    def translate_action_to_sim(self, action):
        return self.catan_game.board.vertex_index_to_pixel_dict[action]
    
    def reset(self):
        self.player_order = 1
        self.round = 1
        self.catan_game = catanAIGame(ifprint=False, ifGUI=False, specialPlayerName=self.special_player_name, selfstart=False)
        self.full_action_space = self.get_current_action_space()

"""
    Run game with given policy
    Return relevant states, actions and rewards
"""

def play_game_with_policy(placement_policy):
    game      = AIGame()

    s         = game.start()
    # 1st placement
    usable_a  = game.get_usable_action_space()
    a         = placement_policy(s, usable_a)
    sp, r     = game.play(a)

    # 2nd placement
    usable_ap = game.get_usable_action_space()
    ap        = placement_policy(sp, usable_ap)
    spp, rp   = game.play(ap)

    # Note if rp=-11, something went wrong and you should just skip this round

    return s, usable_a, a, r, sp, usable_ap, ap, rp, spp