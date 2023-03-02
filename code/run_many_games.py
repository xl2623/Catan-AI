from AIGame import catanAIGame
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]
    
def get_data(filename, numberOfGames = 5000):
    file = open(filename+".txt", "a")
    for i in range(0, numberOfGames):
        game = catanAIGame()
        playerList = [game.playerQueue.get() for i in range(1, game.numPlayers+1)]
        playerVP = [thePlayer.victoryPoints for thePlayer in playerList]
        placementOrder = [int(thePlayer.placementOrder) for thePlayer in playerList]
        index = argmax(playerVP)
        content = "{} {}\n".format(index+1, placementOrder[index])
        file.write(content)
        print(i)
    

def select_random_player(filename, rand_player_name, numberOfGames=5000):
    file = open(filename, "a")
    for i in range(0, numberOfGames):
        game = catanAIGame(specialPlayerName=rand_player_name)
        playerList = [game.playerQueue.get() for i in range(1, game.numPlayers+1)]
        playerVP = [thePlayer.victoryPoints for thePlayer in playerList]
        placementOrder = [int(thePlayer.placementOrder) for thePlayer in playerList]
        index = argmax(playerVP)
        content = "{} {}\n".format(index+1, placementOrder[index])
        file.write(content)
        print(i)
    file.close()

def compute_distribution(fileName):
    # Text file data converted to integer data type
    File_data = np.loadtxt(fileName, dtype=int)
    return File_data

def plotfunc():
    index = 1
    D1 = compute_distribution("data_notrand.txt")
    p1 = plt.hist(D1[:,index], bins = 4)
    plt.title("data_notrand")
    plt.show()  
    D2 = compute_distribution("data_1rand.txt")
    p2 = plt.hist(D2[:,index], bins = 4)
    plt.title("data_1rand")
    plt.show()  
    D3 = compute_distribution("data_2rand.txt")
    p3 = plt.hist(D3[:,index], bins = 4)
    plt.title("data_2rand")
    plt.show()  
    D4 = compute_distribution("data_3rand.txt")
    p4 = plt.hist(D4[:,index], bins = 4)
    plt.title("data_3rand")
    plt.show()  
    D5 = compute_distribution("data_4rand.txt")
    p5 = plt.hist(D5[:,index], bins = 4)
    plt.title("data_4rand")
    plt.show()  


if __name__ == "__main__":
    # filename = "data4rand.txt"
    # select_random_player(filename, '4', 20000-10874)
    # select_random_player(filename)
    plotfunc()

    