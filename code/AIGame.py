#Settlers of Catan
#Gameplay class with pygame with AI players

from board import *
from gameView import *
from player import *
from heuristicAIPlayer import *
import queue
import numpy as np
import sys, pygame
import matplotlib.pyplot as plt

#Class to implement an only AI
class catanAIGame():
    #Create new gameboard
    def __init__(self, ifprint = False, ifGUI = False, specialPlayerName=0, selfstart=True):
        # Display options
        self.ifprint = ifprint
        self.ifGUI = ifGUI
        self.specialPlayerName = specialPlayerName
        
        if self.ifprint:
            print("Initializing Settlers of Catan with only AI Players...")
        self.board = catanBoard()
        
        #Qlearning stuff
        self.q = {}

        #Game State variables
        self.gameOver = False
        self.maxPoints = 10
        self.numPlayers = 0

        #Dictionary to keep track of dice statistics
        self.diceStats = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
        self.diceStats_list = []

        while(self.numPlayers not in [3,4]): #Only accept 3 and 4 player games
            try:
                if self.ifprint:
                    self.numPlayers = int(input("Enter Number of Players (3 or 4):"))
                else:
                    self.numPlayers = 4
            except:
                print("Please input a valid number")
        if self.ifprint:
            print("Initializing game with {} players...".format(self.numPlayers))
            print("Note that Player 1 goes first, Player 2 second and so forth.")
        
        #Initialize blank player queue and initial set up of roads + settlements
        self.playerQueue = queue.Queue(self.numPlayers)
        self.gameSetup = True #Boolean to take care of setup phase

        #Initialize boardview object
        if self.ifGUI:
            self.boardView = catanGameView(self.board, self)
        self.currdiceNum = 0
        self.pixel_to_vertex_index_dict = {y: x for x, y in self.board.vertex_index_to_pixel_dict.items()}
        if selfstart:
            #Functiont to go through initial set up
            self.build_initial_settlements() 

            self.playCatan()

        #Plot diceStats histogram
        if self.ifGUI:
            plt.hist(self.diceStats_list, bins = 11)
            plt.show()

        return None
    
    # compute the state of the current board
    def tostate(self, playerName):
        # extract hex state
        # 0: number of the dice
        self.state = []
        self.state.append(self.currdiceNum)
        ######################### below are static state ###############################
        # 1-19: number on the hex
        for key, value in self.board.hexTileDict.items():
            if value.resource.num == None:
                self.state.append(0)
            else:
                self.state.append(value.resource.num)

        # 20-38: resource on the hex
        Resource_Dict = {'DESERT':0, 'ORE':1, 'BRICK':2, 'WHEAT':3, 'WOOD':4, 'SHEEP':5}
        for key, value in self.board.hexTileDict.items():
            self.state.append(Resource_Dict[value.resource.type])
            if value.robber:
                 robberLoc = value.index
        
        # 38-39: location of the robber
        self.state.append(robberLoc)

        # 40-58: locatioon of the port
        # 18 indices of the vertices that connects to port
        for key, value in self.board.boardGraph.items():
            if value.port:
                self.state.append(value.vertexIndex)

        ######################### below are dynamic state ###############################
        # 59-113: what's on each vertex
        # for each vertex, if nothing is on it, 0
        #                  if myself is on it, if city -> 2
        #                                      if settlement -> 1
        #                  if opponent is on it, if city -> -2
        #                                        if settlement -> -1
        for key, value in self.board.boardGraph.items():
            if value.state["Player"] == None:
                self.state.append(0)
            elif value.state["Player"].name == playerName:
                if value.state["Settlement"]:
                    self.state.append(1)
                elif value.state['City']:
                    self.state.append(2)
            elif value.state["Player"].name != playerName:
                if value.state["Settlement"]:
                    self.state.append(-1)
                elif value.state['City']:
                    self.state.append(-2)

        # 114-185 for each edge, if nothing is on it, 0
        #                if myself is on it, 1
        #                if opponent is on it, -1

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

        # edge bank needs to be kept constant for all future training
        playerList  = list(self.playerQueue.queue)
        roads = [(thePlayer.buildGraph["ROADS"], thePlayer.name) for thePlayer in playerList]
        myRoads = [road[0] for road in roads if road[1]==playerName]
        oppRoads = [road[0] for road in roads if road[1]!=playerName]
        #TODO: need to add logic to check direct order and reverse order of the edge!!!
        for edge in edgeBank:
            if edge in myRoads:
                self.state.append(1)
            elif edge in oppRoads:
                self.state.append(-1)
            else:
                self.state.append(0)


        '''# this code is used to extract edge bank
        for key, value in self.board.boardGraph.items():
            source = value.pixelCoordinates
            if len(value.edgeList) == 3:
                currEdge1 = (source, value.edgeList[0])
                currEdge2 = (source, value.edgeList[1])
                currEdge3 = (source, value.edgeList[2])
                currEdge1eq = (value.edgeList[0], source)
                currEdge2eq = (value.edgeList[1], source)
                currEdge3eq = (value.edgeList[2], source)
                if (currEdge1 not in edgeBank) and (currEdge1eq not in edgeBank):
                    edgeBank.append(currEdge1)
                if (currEdge2 not in edgeBank) and (currEdge2eq not in edgeBank):
                    edgeBank.append(currEdge2)
                if (currEdge3 not in edgeBank) and (currEdge3eq not in edgeBank):
                    edgeBank.append(currEdge3)
            elif len(value.edgeList) == 2:
                currEdge1 = (source, value.edgeList[0])
                currEdge2 = (source, value.edgeList[1])
                currEdge1eq = (value.edgeList[0], source)
                currEdge2eq = (value.edgeList[1], source)
                if (currEdge1 not in edgeBank) and (currEdge1eq not in edgeBank):
                    edgeBank.append(currEdge1)
                if (currEdge2 not in edgeBank) and (currEdge2eq not in edgeBank):
                    edgeBank.append(currEdge2)

        # visualizae edgeBank order
        self.boardView= catanGameView(self.board, self)
        for i in range(0,72):
            self.boardView.draw_road(edgeToDraw = edgeBank[i], roadColor=(255, 0,0))
            pygame.display.update()
            pygame.time.delay(100)'''
        
        # 186-233 for each player, stores if is myself, 
        #                                       road built, 
        #                                       cities built, 
        #                                       settlement built, 
        #                                       knightsPlayed, 
        #                                       maxRoadLength,
        #                                       visibleVictoryPoints
        #                           and all resources
        
        for player in playerList:
            if player.name == playerName:
                self.state.append(1)
            else:
                self.state.append(0)
            self.state.append(15-player.roadsLeft)
            self.state.append(4-player.citiesLeft)
            self.state.append(4-player.settlementsLeft)
            self.state.append(player.knightsPlayed)
            self.state.append(player.maxRoadLength)
            self.state.append(player.visibleVictoryPoints)
            # resource
            self.state.append(player.resources['ORE'])
            self.state.append(player.resources['BRICK'])
            self.state.append(player.resources['WHEAT'])
            self.state.append(player.resources['WOOD'])
            self.state.append(player.resources['SHEEP'])

    # compute the state of the current board
    # simplified for initial placement
    # playerName is an integer that describes the current player
    # playerNameList is a list containing each player's name in the order of placement
    def tostate_simple(self, playerName, playerNameList):
        # extract hex state
        # 0: the order at check player is placed
        self.state = []
        self.state.append(playerNameList.index(playerName))
        ######################### below are static state ###############################
        # 1-19: number on the hex
        for key, value in self.board.hexTileDict.items():
            if value.resource.num == None:
                self.state.append(0)
            else:
                self.state.append(value.resource.num)

        # 20-38: resource on the hex
        Resource_Dict = {'DESERT':0, 'ORE':1, 'BRICK':2, 'WHEAT':3, 'WOOD':4, 'SHEEP':5}
        for key, value in self.board.hexTileDict.items():
            self.state.append(Resource_Dict[value.resource.type])
            if value.robber:
                 robberLoc = value.index
        
        # # 39-57: locatioon of the port
        # #        indices of the vertices that connects to port
        # for key, value in self.board.boardGraph.items():
        #     if value.port:
        #         self.state.append(value.vertexIndex)
        # # 39-57: locatioon of the port
        # #        indices of the vertices that connects to port
        # for key, value in self.board.boardGraph.items():
        #     if value.port:
        #         self.state.append(value.vertexIndex)

        ######################### below are dynamic state ###############################
        # 60-68: settlement placement
        # 60, 61 (my settlement placement)
        #        if placed, (vertexindex)
        #        if not placement, (-1)
        # 62, 63 (oppoent #1's settlement placement)
        #        if placed, (vertexindex)
        #        if not placement, -1
        # 64, 65 (oppoent #2's settlement placement)
        #        if placed, (vertexindex)
        #        if not placement, -1
        # 66, 67 (oppoent #3's settlement placement)
        #        if placed, (vertexindex)
        #        if not placement, -1
        playerList  = list(self.playerQueue.queue)
        start = int(playerName)-1                  # start from the 0 index of state
        for i in range(start, start+len(playerList)):
            currPlayer = playerList[i%len(playerList)]
            settlements = currPlayer.buildGraph['SETTLEMENTS']
            if len(settlements) == 0:
                self.state.append(-1)
                self.state.append(-1)
            elif len(settlements) == 1:
                self.state.append(self.pixel_to_vertex_index_dict[settlements[0]])
                self.state.append(-1)
            else:
                self.state.append(self.pixel_to_vertex_index_dict[settlements[0]])
                self.state.append(self.pixel_to_vertex_index_dict[settlements[1]])

        # 68-76: road placement
        # 60, 61 (my road placement)
        #        if placed, (edgeindex)
        #        if not placement, (-1)
        # 62, 63 (oppoent #1's road placement)
        #        if placed, (edgeindex)
        #        if not placement, -1
        # 64, 65 (oppoent #2's raod placement)
        #        if placed, (edgeindex)
        #        if not placement, -1
        # 66, 67 (oppoent #3's road placement)
        #        if placed, (edgeindex)
        #        if not placement, -1
        #
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
        for i in range(start, start+len(playerList)):
            currPlayer = playerList[i%len(playerList)]
            roads = currPlayer.buildGraph['ROADS']
            if len(roads) == 0:
                self.state.append(-1)
                self.state.append(-1)
            elif len(roads) == 1:
                try:
                    self.state.append(edgeBank.index((roads[0][0], roads[0][1])))
                except:
                    self.state.append(edgeBank.index((roads[0][1], roads[0][0])))
                self.state.append(-1)
            else:
                try:
                    self.state.append(edgeBank.index((roads[0][0], roads[0][1])))
                except:
                    self.state.append(edgeBank.index((roads[0][1], roads[0][0])))
                
                try:
                    self.state.append(edgeBank.index((roads[1][0], roads[1][1])))
                except:
                    self.state.append(edgeBank.index((roads[1][1], roads[1][0])))

        return self.state

    def toaction_simple(self, playerName, s, ns):
        # action space is coupled with state space
        # but this is not a problem. This shuold be dealt with duing deployment phase
        # only allow action in set {a1, a2, ... an} \in allowable(s)
        # here the maximum action will be considered
        # there are 72 possible edges, and 54 possible vertices
        self.action = []
        diff = [ns[i]-s[i] for i in range(39, 55)]# extract difference between current state and next state
        diff = [ns[i]-s[i] for i in range(57-18, 73-18)]# extract difference between current state and next state
        if diff[0] != 0:
            self.action.append(diff[0])
        else:
            self.action.append(diff[1])
        if diff[8] != 0:
            self.action.append(diff[8])
        else:
            self.action.append(diff[9])
        return self.action

    def create_player_list(self, special_placement_type="random"):
        #Initialize new players with names and colors
        playerColors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        for i in range(self.numPlayers):
            if self.ifprint:
                playerNameInput = input("Enter AI Player {} name: ".format(i+1))
                newPlayer = heuristicAIPlayer(playerNameInput, playerColors[i])
                newPlayer.updateAI()
                self.playerQueue.put(newPlayer)
            else:
                playerNameInput = ["1", "2", "3", "4"]
                if playerNameInput[i] == self.specialPlayerName:
                    newPlayer = heuristicAIPlayer(playerNameInput[i], playerColors[i], init_placement_type=special_placement_type)
                else:
                    newPlayer = heuristicAIPlayer(playerNameInput[i], playerColors[i])
                    
                newPlayer.updateAI()
                self.playerQueue.put(newPlayer)

        playerList = list(self.playerQueue.queue)

        return playerList

    def allocate_initial_resources(self, playerList):
        for player_i in playerList:
            #Initial resource generation
            #check each adjacent hex to latest settlement
            for adjacentHex in self.board.boardGraph[player_i.buildGraph['SETTLEMENTS'][-1]].adjacentHexList:
                resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                if(resourceGenerated != 'DESERT'):
                    player_i.resources[resourceGenerated] += 1

    #Function to initialize players + build initial settlements for players
    def build_initial_settlements(self):
        playerList = self.create_player_list("heuristic")
        # choose from heuristic, random, learning

        #Build Settlements and roads of each player forwards
        random.shuffle(playerList)
        playerNameList=[thePlayer.name for thePlayer in playerList]
        order = 1
        for player_i in playerList:
            # extract state
            s = self.tostate_simple(player_i.name, playerNameList)
            # print(self.state[0])
            # print(self.state[57:73])
            player_i.placementOrder = order
            order += 1
            player_i.initial_setup(self.board)
            pygame.event.pump()
            if self.ifGUI:
                self.boardView.displayGameScreen()
            if self.ifprint:
                pygame.time.delay(1000)
            # extract next state
            ns = self.tostate_simple(player_i.name, playerNameList)
            # print(self.state[0])
            # print(self.state[57:73])
            a = self.toaction_simple(player_i.name, s, ns)
            # print(s[57:73])
            # print(ns[57:73])
            self.q[player_i.name] = {"s1":s, "a1":a}


        #Build Settlements and roads of each player reverse
        playerList.reverse()
        for player_i in playerList: 
            s = self.tostate_simple(player_i.name, playerNameList)
            # print(self.state[0])
            # print(self.state[57:73])
            player_i.initial_setup(self.board)
            pygame.event.pump()
            if self.ifGUI:
                self.boardView.displayGameScreen()
            if self.ifprint:
                pygame.time.delay(1000)
            ns = self.tostate_simple(player_i.name, playerNameList)
            # print(self.state[0])
            # print(self.state[57:73])
            a = self.toaction_simple(player_i.name, s, ns)
            # print(s[57:73])
            # print(ns[57:73])
            # print(len(self.state))
            if self.ifprint:
                print("Player {} starts with {} resources".format(player_i.name, len(player_i.setupResources)))
            self.q[player_i.name]["a2"] = a
            self.q[player_i.name]["s2"] = s
            self.q[player_i.name]["s3"] = ns

            #Initial resource generation
            #check each adjacent hex to latest settlement
            for adjacentHex in self.board.boardGraph[player_i.buildGraph['SETTLEMENTS'][-1]].adjacentHexList:
                resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                if(resourceGenerated != 'DESERT'):
                    player_i.resources[resourceGenerated] += 1
                    if self.ifprint:
                        print("{} collects 1 {} from Settlement".format(player_i.name, resourceGenerated))

        

        if self.ifprint:
            pygame.time.delay(10000)
        self.gameSetup = False

    #Function to roll dice 
    def rollDice(self):
        dice_1 = np.random.randint(1,7)
        dice_2 = np.random.randint(1,7)
        diceRoll = dice_1 + dice_2
        if self.ifprint:
            print("Dice Roll = ", diceRoll, "{", dice_1, dice_2, "}")

        return diceRoll

    #Function to update resources for all players
    def update_playerResources(self, diceRoll, currentPlayer):
        if(diceRoll != 7): #Collect resources if not a 7
            #First get the hex or hexes corresponding to diceRoll
            hexResourcesRolled = self.board.getHexResourceRolled(diceRoll)
            #print('Resources rolled this turn:', hexResourcesRolled)

            #Check for each player
            for player_i in list(self.playerQueue.queue):
                #Check each settlement the player has
                for settlementCoord in player_i.buildGraph['SETTLEMENTS']:
                    for adjacentHex in self.board.boardGraph[settlementCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 1
                            if self.ifprint:
                                print("{} collects 1 {} from Settlement".format(player_i.name, resourceGenerated))
                
                #Check each City the player has
                for cityCoord in player_i.buildGraph['CITIES']:
                    for adjacentHex in self.board.boardGraph[cityCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 2
                            if self.ifprint:
                                print("{} collects 2 {} from City".format(player_i.name, resourceGenerated))

                if self.ifprint:
                    print("Player:{}, Resources:{}, Points: {}".format(player_i.name, player_i.resources, player_i.victoryPoints))
                    #print('Dev Cards:{}'.format(player_i.devCards))
                    #print("RoadsLeft:{}, SettlementsLeft:{}, CitiesLeft:{}".format(player_i.roadsLeft, player_i.settlementsLeft, player_i.citiesLeft))
                    print('MaxRoadLength:{}, Longest Road:{}\n'.format(player_i.maxRoadLength, player_i.longestRoadFlag))
        
        else:
            if self.ifprint:
                print("AI using heuristic robber...")
            currentPlayer.heuristic_move_robber(self.board)

    #function to check if a player has the longest road - after building latest road
    def check_longest_road(self, player_i):
        if(player_i.maxRoadLength >= 5): #Only eligible if road length is at least 5
            longestRoad = True
            for p in list(self.playerQueue.queue):
                if(p.maxRoadLength >= player_i.maxRoadLength and p != player_i): #Check if any other players have a longer road
                    longestRoad = False
            
            if(longestRoad and player_i.longestRoadFlag == False): #if player_i takes longest road and didn't already have longest road
                #Set previous players flag to false and give player_i the longest road points
                prevPlayer = ''
                for p in list(self.playerQueue.queue):
                    if(p.longestRoadFlag):
                        p.longestRoadFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.longestRoadFlag = True
                player_i.victoryPoints += 2

                if self.ifprint:
                    print("Player {} takes Longest Road {}".format(player_i.name, prevPlayer))

    #function to check if a player has the largest army - after playing latest knight
    def check_largest_army(self, player_i):
        if(player_i.knightsPlayed >= 3): #Only eligible if at least 3 knights are player
            largestArmy = True
            for p in list(self.playerQueue.queue):
                if(p.knightsPlayed >= player_i.knightsPlayed and p != player_i): #Check if any other players have more knights played
                    largestArmy = False
            
            if(largestArmy and player_i.largestArmyFlag == False): #if player_i takes largest army and didn't already have it
                #Set previous players flag to false and give player_i the largest points
                prevPlayer = ''
                for p in list(self.playerQueue.queue):
                    if(p.largestArmyFlag):
                        p.largestArmyFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.largestArmyFlag = True
                player_i.victoryPoints += 2

                if self.ifprint:
                    print("Player {} takes Largest Army {}".format(player_i.name, prevPlayer))

    #Function that runs the main game loop with all players and pieces
    def playCatan(self):
        #self.board.displayBoard() #Display updated board
        numTurns = 0
        while (self.gameOver == False):
            if numTurns > 500:
                self.gameOver == True
                # print("gameover triggered")
                # self.triggered == True
            #Loop for each player's turn -> iterate through the player queue
            for currPlayer in self.playerQueue.queue:
                # self.tostate(currPlayer.name)
                # TODO: add self.action
                # print(self.state)
                # currPlayer.sarn.append([self.state, ])
                numTurns += 1
                
                if self.ifprint:
                    print("---------------------------------------------------------------------------")
                    print("Current Player:", currPlayer.name)

                turnOver = False #boolean to keep track of turn
                diceRolled = False  #Boolean for dice roll status
                
                #Update Player's dev card stack with dev cards drawn in previous turn and reset devCardPlayedThisTurn
                currPlayer.updateDevCards()
                currPlayer.devCardPlayedThisTurn = False

                while(turnOver == False):

                    #TO-DO: Add logic for AI Player to move
                    #TO-DO: Add option of AI Player playing a dev card prior to dice roll
                    
                    #Roll Dice and update player resources and dice stats
                    pygame.event.pump()
                    self.currdiceNum = self.rollDice()
                    diceNum = self.currdiceNum
                    diceRolled = True
                    self.update_playerResources(diceNum, currPlayer)
                    self.diceStats[diceNum] += 1
                    self.diceStats_list.append(diceNum)

                    currPlayer.move(self.board) #AI Player makes all its moves
                    #Check if AI player gets longest road and update Victory points
                    self.check_longest_road(currPlayer)
                    if self.ifprint:
                        print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))

                    if self.ifGUI:
                        self.boardView.displayGameScreen()#Update back to original gamescreen
                    if self.ifprint:
                        pygame.time.delay(300)
                    turnOver = True
                    
                    #Check if game is over
                    if currPlayer.victoryPoints >= self.maxPoints:
                        for player in list(self.playerQueue.queue):
                            continue
                            self.q[player.name]['r'] = player.victoryPoints-self.maxPoints
                        self.gameOver = True
                        self.turnOver = True
                        if self.ifprint:
                            print("====================================================")
                            print("PLAYER {} WINS IN {} TURNS!".format(currPlayer.name, int(numTurns/4)))
                            print(self.diceStats)
                            print("Exiting game in 5 seconds...")
                            pygame.time.delay(5000)
                        else:
                            # print("====================================================")
                            # print("PLAYER {} WINS IN {} TURNS!".format(currPlayer.name, int(numTurns/4)))

                            # return currPlayer.name, int(numTurns/4)
                            special_player = [player for player in self.playerQueue.queue if player.name == self.specialPlayerName][0]
                            return special_player.victoryPoints
                        break

                if(self.gameOver):
                    # startTime = pygame.time.get_ticks()
                    # runTime = 0
                    # while(runTime < 5000): #5 second delay prior to quitting
                    #     runTime = pygame.time.get_ticks() - startTime

                    special_player = [player for player in self.playerQueue.queue if player.name == self.specialPlayerName][0]
                    
                    return special_player.victoryPoints
                    # break

def strnice(input):
    result = ""
    for i in input:
        result += str(i)+" "
    return result

if __name__=="__main__":
    newGame_AI = catanAIGame(specialPlayerName="1")
    # iterations = 180
    # f1 = open("data.txt", "a")
    # for i in range(0,iterations):
    #     newGame_AI = catanAIGame()
    #     playerList = list(newGame_AI.playerQueue.queue)
    #     for player in playerList:
    #         f1.write(strnice(newGame_AI.q[player.name]["s1"]+newGame_AI.q[player.name]["a1"]+newGame_AI.q[player.name]["s2"]+newGame_AI.q[player.name]["a2"]+newGame_AI.q[player.name]["s3"])+str(newGame_AI.q[player.name]["r"])+'\n')
    #     print(i)
