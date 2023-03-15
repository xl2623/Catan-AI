from AIGame import catanAIGame
import numpy as np
from heuristicAIPlayer import *
import pygame

def play_game(placement_policy):
    # Create game object
    special_player_name = 0
    catan_game = catanAIGame(ifprint=False, ifGUI=False, special_player_name=special_player_name, selfstart=False)

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
            player_i.inital_setup(catan_game.board)
        pygame.event.pump()
    
    # 2nd placement
    player_list.reverse()

    for player_i in player_list:
        # Place settlement
        if player_i.name == special_player_name:
            player_i.initial_setup(catan_game.board, placement_policy)
        else:
            player_i.inital_setup(catan_game.board)
        pygame.event.pump()
        
    # Resource allocation for players
    catan_game.allocate_initial_resources(player_list)

    # Play the game
    winner, turns = catan_game.playCatan()

    return winner, turns
