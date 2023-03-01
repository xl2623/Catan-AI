from AIGame import catanAIGame

for i in range(1,5):
    # Name of the player that will use the random policy
    rand_player_name = str(i)
    print("Random player: " + rand_player_name)
    game = catanAIGame(specialPlayerName=rand_player_name)
    print("\n")