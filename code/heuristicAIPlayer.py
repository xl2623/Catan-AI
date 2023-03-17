#Settlers of Catan
#Heuristic AI class implementation

from board import *
from player import *
import numpy as np
import random

'''
    Allowable initial placement functions
    
    - heuristic
    - random
'''

#Class definition for an AI player
class heuristicAIPlayer(player):
    
    #Update AI player flag and resources
    def updateAI(self): 
        self.isAI = True
        self.setupResources = [] #List to keep track of setup resources
        #Initialize resources with just correct number needed for set up
        self.resources = {'ORE':0, 'BRICK':4, 'WHEAT':2, 'WOOD':4, 'SHEEP':2} #Dictionary that keeps track of resource amounts
        if self.ifprint:
            print("Added new AI Player:", self.name)

    '''
    Set of initial placement functions
    '''
    def initial_setup(self, board, action=None, state=None):
        if self.init_placement_type == 'heuristic':
            self.initial_heuristic_setup(board)
        elif self.init_placement_type == 'random':
            self.initial_random_setup(board)
        elif self.init_placement_type == 'learning':
            self.initial_learning_setup(board, action, state)
        else:
            raise NameError("HeuristicPlayer: initial setup type is not recognized")

    def build_settlement_at_vertex(self, vertexToBuild, board):
        #Add to setup resources
        for adjacentHex in board.boardGraph[vertexToBuild].adjacentHexList:
            resourceType = board.hexTileDict[adjacentHex].resource.type
            if(resourceType not in self.setupResources and resourceType != 'DESERT'):
                self.setupResources.append(resourceType)

        self.build_settlement(vertexToBuild, board)

    def build_rand_road(self, board):
        # Build random road
        possibleRoads = board.get_setup_roads(self)
        randomEdge = np.random.randint(0, len(possibleRoads.keys()))
        self.build_road(list(possibleRoads.keys())[randomEdge][0], list(possibleRoads.keys())[randomEdge][1], board)

    def initial_random_setup(self, board):
        possibleVertices = board.get_setup_settlements(self)
        
        # Build random settlement
        vertexToBuild = random.choice(list(possibleVertices.keys()))
        self.build_settlement_at_vertex(vertexToBuild, board)
        
        # Build random road
        self.build_rand_road(board)

    #Function to build an initial settlement based on heuristics
    def initial_heuristic_setup(self, board):
        possibleVertices = board.get_setup_settlements(self)
        
        #Simple heuristic for choosing initial spot
        diceRoll_expectation = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1, None:0}
        vertexValues = []

        #Get the adjacent hexes for each hex
        for v in possibleVertices.keys():
            vertexNumValue = 0
            resourcesAtVertex = []
            #For each adjacent hex get its value and overall resource diversity for that vertex
            for adjacentHex in board.boardGraph[v].adjacentHexList:
                resourceType = board.hexTileDict[adjacentHex].resource.type
                if(resourceType not in resourcesAtVertex):
                    resourcesAtVertex.append(resourceType)
                numValue = board.hexTileDict[adjacentHex].resource.num
                vertexNumValue += diceRoll_expectation[numValue] #Add to total value of this vertex

            #basic heuristic for resource diversity
            vertexNumValue += len(resourcesAtVertex)*2
            for r in resourcesAtVertex:
                if(r != 'DESERT' and r not in self.setupResources):
                    vertexNumValue += 2.5 #Every new resource gets a bonus
            vertexValues.append(vertexNumValue)
        
        # Choose best vertex based on heuristics
        vertexToBuild_index = vertexValues.index(max(vertexValues))
        vertexToBuild = list(possibleVertices.keys())[vertexToBuild_index]
        self.build_settlement_at_vertex(vertexToBuild, board)

        # Build random road
        self.build_rand_road(board)
    
    def initial_learning_setup(self, board, action, state):
        # goal is to extra possible action based on current state
        possibleVertices = board.get_setup_settlements(self)
        pixel_to_vertex_index_dict = dict((v, k) for k, v in board.vertex_index_to_pixel_dict.items())
        possibleVertices_index = [pixel_to_vertex_index_dict[key] for key in pixel_to_vertex_index_dict if key in possibleVertices]
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
                
        possibleRoads = []
        for j in range(0,len(possibleVertices_index)):
            currVertex = board.boardGraph[[board.vertex_index_to_pixel_dict[vertex_index] for vertex_index in possibleVertices_index][j]]
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

        #print(possibleRoads)
        #print(possibleVertices_index)
        # assemble action space
        A = []
        for i in range(1, len(possibleVertices_index)):
            for road in possibleRoads[i]:
                curr_action = []
                curr_action.append(possibleVertices_index[i])
                curr_action.append(road)
                A.append(curr_action)
        
        # possible action space is A
        # current state is state
        # print(A)
        # print(state)

        SApair = []
        for element in A:
            currSA = state+element
            SApair.append(currSA)

        # # Build settlement
        # vertexToBuild = action
        # self.build_settlement_at_vertex(vertexToBuild, board)

        # # Build random road
        # self.build_rand_road(board)

    '''
    '''
    
    def move(self, board):
        if self.ifprint:
            print("AI Player {} playing...".format(self.name))
        #Trade resources if there are excessive amounts of a particular resource
        self.trade()
        #Build a settlements, city and few roads
        possibleVertices = board.get_potential_settlements(self)
        if(possibleVertices != {} and (self.resources['BRICK'] > 0 and self.resources['WOOD'] > 0 and self.resources['SHEEP'] > 0 and self.resources['WHEAT'] > 0)):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            self.build_settlement(list(possibleVertices.keys())[randomVertex], board)

        #Build a City
        possibleVertices = board.get_potential_cities(self)
        if(possibleVertices != {} and (self.resources['WHEAT'] >= 2 and self.resources['ORE'] >= 3)):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            self.build_city(list(possibleVertices.keys())[randomVertex], board)

        #Build a couple roads
        for i in range(2):
            if(self.resources['BRICK'] > 0 and self.resources['WOOD'] > 0):
                possibleRoads = board.get_potential_roads(self)
                if len(possibleRoads.keys()) != 0:
                    randomEdge = np.random.randint(0, len(possibleRoads.keys()))
                    self.build_road(list(possibleRoads.keys())[randomEdge][0], list(possibleRoads.keys())[randomEdge][1], board)

        #Draw a Dev Card with 1/3 probability
        devCardNum = np.random.randint(0, 3)
        if(devCardNum == 0):
            self.draw_devCard(board)
        
        return

    #Wrapper function to control all trading
    def trade(self):
        for r1, r1_amount in self.resources.items():
            if(r1_amount >= 6): #heuristic to trade if a player has more than 5 of a particular resource
                for r2, r2_amount in self.resources.items():
                    if(r2_amount < 1):
                        self.trade_with_bank(r1, r2)
                        break

    
    #Choose which player to rob
    def choose_player_to_rob(self, board):
        '''Heuristic function to choose the player with maximum points.
        Choose hex with maximum other players, Avoid blocking own resource
        args: game board object
        returns: hex index and player to rob
        '''
        #Get list of robber spots
        robberHexDict = board.get_robber_spots()

        hexToRob_index = random.choice([key for key in robberHexDict])
        
        #Choose a hexTile with maximum adversary settlements
        maxHexScore = 0 #Keep only the best hex to rob
        for hex_ind, hexTile in robberHexDict.items():
            #Extract all 6 vertices of this hexTile
            vertexList = polygon_corners(board.flat, hexTile.hex)

            hexScore = 0 #Heuristic score for hexTile
            playerToRob_VP = 0
            playerToRob = None
            for vertex in vertexList:
                playerAtVertex = board.boardGraph[vertex].state['Player']
                if playerAtVertex == self:
                    hexScore -= self.victoryPoints
                elif playerAtVertex != None: #There is an adversary on this vertex
                    hexScore += playerAtVertex.visibleVictoryPoints
                    #Find strongest other player at this hex, provided player has resources
                    if playerAtVertex.visibleVictoryPoints >= playerToRob_VP and sum(playerAtVertex.resources.values()) > 0:
                        playerToRob_VP = playerAtVertex.visibleVictoryPoints
                        playerToRob = playerAtVertex
                else:
                    pass

            if hexScore >= maxHexScore and playerToRob != None:
                hexToRob_index = hex_ind
                playerToRob_hex = playerToRob
                maxHexScore = hexScore

        return hexToRob_index, playerToRob_hex


    def heuristic_move_robber(self, board):
        '''Function to control heuristic AI robber
        Calls the choose_player_to_rob and move_robber functions
        args: board object
        '''
        #Get the best hex and player to rob
        hex_i, playerRobbed = self.choose_player_to_rob(board)

        #Move the robber
        self.move_robber(hex_i, board, playerRobbed)

        return


    # def heuristic_play_dev_card(self, board):
    #     '''Heuristic strategies to choose and play a dev card
    #     args: board object
    #     '''
    #     #Check if player can play a devCard this turn
    #     if self.devCardPlayedThisTurn != True:
    #         #Get a list of all the unique dev cards this player can play
    #         devCardsAvailable = []
    #         for cardName, cardAmount in self.devCards.items():
    #             if(cardName != 'VP' and cardAmount >= 1): #Exclude Victory points
    #                 devCardsAvailable.append((cardName, cardAmount))

    #         if(len(devCardsAvailable) >=1):
                #If a hexTile is currently blocked, try and play a Knight

                #If expansion needed, try road-builder

                #If resources needed, try monopoly or year of plenty


    def resources_needed_for_settlement(self):
        '''Function to return the resources needed for a settlement
        args: player object - use self.resources
        returns: list of resources needed for a settlement
        '''
        resourcesNeededDict = {}
        for resourceName in self.resources.keys():
            if resourceName != 'ORE' and self.resources[resourceName] == 0:
                resourcesNeededDict[resourceName] = 1

        return resourcesNeededDict


    def resources_needed_for_city(self):
        '''Function to return the resources needed for a city
        args: player object - use self.resources
        returns: list of resources needed for a city
        '''
        resourcesNeededDict = {}
        if self.resources['ORE'] < 3:
            resourcesNeededDict['ORE'] = 3 - self.resources['ORE']

        if self.resources['WHEAT'] < 2:
            resourcesNeededDict['ORE'] = 2 - self.resources['WHEAT']

        return resourcesNeededDict

    def heuristic_discard(self):
        '''Function for the AI to choose a set of cards to discard upon rolling a 7
        '''
        return

    #Function to propose a trade -> give r1 and get r2
    #Propose a trade as a dictionary with {r1:amt_1, r2: amt_2} specifying the trade
    #def propose_trade_with_players(self):
    

    #Function to accept/reject trade - return True if accept
    #def accept_trade(self, r1_dict, r2_dict):
        

    #Function to find best action - based on gamestate
    def get_action(self):
        return

    #Function to execute the player's action
    def execute_action(self):
        return




