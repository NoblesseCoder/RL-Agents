import gym
import gym_karmedbandits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



sns.set(color_codes=True, style="whitegrid")

# Initialize gym Environment
env = gym.make("KArmedBandits-v0")
env.reset()

global_reward_list = []
episodes = 1000
n_actions = 10

# Reinforcement Learning Training 
for action in range(n_actions):
	local_reward_list = []
	for episode in range(episodes):
		while True:
			next_state, reward, done, _ = env.step(action)
			if done:
				break
		local_reward_list.append(reward)
	
	env.close()
	global_reward_list.append(local_reward_list)


# Plotting the reward distributions of k actions
plt.title("Reward distributions of " + str(n_actions) + " actions for MAB problem")
plt.xlabel("Action")
plt.ylabel("Reward Distribution")
sns.violinplot(data=np.array(global_reward_list).T)
plt.savefig("MAB_reward_dist_plot.jpg")
plt.show()
