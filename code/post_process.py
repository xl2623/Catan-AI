import pandas as pd
import matplotlib.pyplot as plt
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

filename = 'results_alpha_1e-2_epsilon_1_decay_0.9_rate_500_same_board_iter_100k_heuristic.csv'

filepath = os.path.join(dir_path, 'data/', filename)

# Load data
data = pd.read_csv(filepath)

# Rolling averages
avg_size = 100
data["avg_theta_change"] = data["tot_theta_change"].rolling(avg_size).mean()
data["avg_win"] = data["win"].rolling(avg_size).mean()
data["avg_reward"] = data["reward"].rolling(avg_size).mean()

# Plots galore
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(data["iteration"], data["avg_theta_change"])
ax.set_title("Rolling average of the change in the parameters since initialization \n" + r'$\gamma$ = 0.99 $\alpha$ = 0.01 $\epsilon$ = 1.0 decay = 0.9 rate = 500')
ax.set_xlabel("Iteration #")
ax.set_ylabel(r'Rolling average of $\theta - \theta_0$')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(data["iteration"], data["avg_win"])
ax.set_title("Rolling average of the win rate \n" + r'$\gamma$ = 0.99 $\alpha$ = 0.01 $\epsilon$ = 1.0 decay = 0.9 rate = 500')
ax.set_xlabel("Iteration #")
ax.set_ylabel(r'Rolling average of the win rate')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(data["iteration"], data["avg_reward"])
ax.set_title("Rolling average of the reward \n" + r'$\gamma$ = 0.99 $\alpha$ = 0.01 $\epsilon$ = 1.0 decay = 0.9 rate = 500')
ax.set_xlabel("Iteration #")
ax.set_ylabel(r'Rolling average of the reward')
plt.grid()
plt.show()